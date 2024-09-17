# -*- coding: utf-8 -*-

import re
import sh
import os
import json
import io
import zstandard as zstd
import codecs
import re
import base64
import numpy as np
import time
import random
import traceback
import pyarrow
from pyarrow import hdfs
from collections import defaultdict, namedtuple

from .ad_gpu_ps_pb2 import GpuPsFeature64
from .util import logger, PsBalance, get_lines
from .util import dctx, cctx


def get_shard_from_v2(filename: str) -> dict:
    p = re.compile(r'embedding_(\d+).*prod-ps-(\d+)')
    d = {}
    with open(filename) as f:
        for line in f:
            if line.find('embedding_') < 0 or line.find('prod-ps') < 0:
                continue
            arr = p.findall(line)
            if len(arr) > 0:
                key = int(arr[0][0])
                ps_index = int(arr[0][1])
                if key not in d:
                    d[key] = {'load': 0.0, 'shard': set()}
                d[key]['shard'].add(ps_index)

    for key in d:
        d[key]['shard'] = list(sorted(d[key]['shard']))

    return d

def get_shard_from_v3(filename):
    p_embedding = re.compile(r'embedding_(\d+)')
    p_index = re.compile(r'prod-ps-(\d+)')
    d = {}
    with open(filename) as f:
        for line in open(filename):
            if line.find('embedding_') < 0 or line.find('prod-ps') < 0:
                continue
            arr_embedding = p_embedding.findall(line)
            arr_index = p_index.findall(line)
            if len(arr_embedding) > 0 and len(arr_index) > 0:
                key = int(arr_embedding[0])
                if key not in d:
                    d[key] = {'load': 1.0, 'shard': set()}

                ps_index = set([int(x) for x in arr_index])

                d[key]['shard'] = d[key]['shard'].union(ps_index)

    for key in d:
        d[key]['shard'] = list(sorted(d[key]['shard']))

    return d

def get_shard_from_ps_name(d: dict) -> dict:
    new_d = {}
    p = re.compile(r'prod-ps-([0-9]+).')

    for key in d:
        arr = []
        for name in d[key]['shard']:
            match = p.findall(name)
            if len(match) == 1:
                arr.append(int(match[0]))
        new_d[int(key)] = {
            'load': d[key]['load'],
            'shard': arr
        }

    return new_d

class LockBalance(PsBalance):
    def __init__(self, limit=100):
        self._has_debug_info = None
        self._sparse_features = []
        self._enable_format_opt = True
        self._cnt_map = {}
        self._median_load = [0.0 for i in range(5)]
        self._mean_load = 0.0
        self._limit = limit

    def parse_dp(self, dp):
        idx = 0
        nums = 4
        hh = np.fromstring(dp[idx: idx + nums * 4], dtype=np.int32)

        batch_size = hh[0]
        # logger.info('batch_size', batch_size)
        dense_total_size = hh[1]
        # logger.info('dense_total_size:', dense_total_size)
        field_count = hh[2]
        # logger.info('field_count:', field_count)
        sparse_value_num = hh[3]
        # logger.info('sparse_value_num:', sparse_value_num)
        # print(batch_size, dense_total_size, field_count, sparse_value_num)

        idx = idx + nums * 4
        nums = batch_size
        labels = np.fromstring(dp[idx: idx + nums * 4], dtype=np.int32)
        idx = idx + nums * 4
        nums = batch_size * dense_total_size
        dense_values = np.fromstring(dp[idx: idx + nums * 4], dtype=np.float32)
        dense_values = dense_values.reshape((batch_size, dense_total_size))

        idx = idx + nums * 4
        nums = field_count
        sparse_field_sizes = np.fromstring(
            dp[idx: idx + nums * 4], dtype=np.int32)

        idx = idx + nums * 4
        nums = field_count
        sparse_field_offsets = np.fromstring(
            dp[idx: idx + nums * 4], dtype=np.int32)
        idx = idx + nums * 4

        if self._has_debug_info == None:
            self._has_debug_info = self._guess_debug_info(
                dp, sparse_field_offsets, idx)
            logger.info('guess_debug_info: %s', self._has_debug_info)
        if self._has_debug_info == True:
            sparse_field_offsets = np.append(
                sparse_field_offsets, np.fromstring(dp[idx: idx + 4], dtype=np.int32))
            idx += 4

        ret = {}
        for i in range(field_count):
            if i not in self._cnt_map:
                self._cnt_map[i] = {
                    'total_cnt': 0,
                    'unique': set()
                }

            if i not in ret:
                ret[i] = {
                    'total_cnt': 0,
                    'unique': set()
                }

            start = sparse_field_offsets[i]
            if i != len(sparse_field_offsets) - 1:
                ps_fea_str = dp[idx + start: idx + sparse_field_offsets[i + 1]]
            else:
                if self._has_debug_info:
                    ps_fea_str = dp[idx + start: idx +
                                    sparse_field_offsets[i + 1]]
                else:
                    ps_fea_str = dp[idx + start:]

            ps_fea = dctx.decompress(ps_fea_str)
            GpuPsfea = GpuPsFeature64()
            GpuPsfea.ParseFromString(ps_fea)
            assert len(GpuPsfea.features) == len(GpuPsfea.item_indices)
            cnt = len(GpuPsfea.features)
            fea = 0
            unique_cnt = 0
            total_cnt = 0
            cur_fea = float('inf')
            for j in range(cnt):
                fea += GpuPsfea.features[j]
                if cur_fea != fea:
                    self._cnt_map[i]['unique'].add(cur_fea)
                    self._cnt_map[i]['total_cnt'] += 1
                    cur_fea = fea
                    ret[i]['total_cnt'] += 1
                    ret[i]['unique'].add(cur_fea)

        for i in ret:
            ret[i]['unique_cnt'] = len(ret[i]['unique'])

        return ret

    def compute_load(self, path: str, compute_type: str = '', share_dict: dict = {}) -> dict:
        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(path):
            raise Exception('path: %s not exists' % (path))

        n = 300
        info = fs.ls(path)
        d = defaultdict(float)
        batch_loads = defaultdict(float)
        total_loads = defaultdict(float)
        interleaving_loads = defaultdict(float)
        one_file = [
            x for x in info if os.path.basename(x) != '_SUCCESS'
            and not os.path.basename(x).startswith('.')
        ][0]
        logger.info('one_file_name:{}'.format(one_file))
        try:
            loads = []
            for line in get_lines(one_file, limit=self._limit):
                logger.info('len(line): %d', len(line))
                loads.append(self.process(line))

            for key in self._cnt_map:
                capacity = self._sparse_features[int(key)].capacity if key < len(
                    self._sparse_features) else 1000000
                self._cnt_map[key]['unique_cnt'] = len(
                    self._cnt_map[key]['unique'])
                load = int(self._cnt_map[key]['total_cnt'] / 10) * (
                    self._cnt_map[key]['unique_cnt'] / capacity)
                total_loads[int(key)] += load

                logger.info('key: %d, unique_cnt: %d, total_cnt: %d, load: %f', key,
                            self._cnt_map[key]['unique_cnt'], self._cnt_map[key]['total_cnt'], load)

            for i, one_batch_load in enumerate(loads):
                for key in one_batch_load:
                    capacity = self._sparse_features[int(key)].capacity if key < len(
                        self._sparse_features) else 1000000
                    load = int(one_batch_load[key]['total_cnt'] / 10) * \
                        one_batch_load[key]['unique_cnt'] / capacity

                    batch_loads[key] += load

            for key in self._cnt_map:
                if int(key) in share_dict:
                    key = share_dict[int(key)]

                logger.info('key: %d, v: %s', key, total_loads[key])
                d[key] = total_loads[key]

        except sh.ErrorReturnCode_1 as e:
            logger.info(e)
        except Exception as e1:
            logger.info(e1)
            traceback.print_exc()

        load_values = list(sorted(d.values()))
        self._mean_load = sum(load_values) / len(load_values)

        logger.info('done, filename: %s, get %d load, mean_load: %f, median_load: %s',
                    one_file, len(d), self._mean_load, str(self._median_load))
        logger.info(d)
        return d

    def alloc_ps_num(self, load, ps_max_num, loads_eps):
        shard_num = 0
        if load >= self._mean_load:
            shard_num = min(8, ps_max_num)
        else:
            shard_num = 1

        sorted_index = self.get_sorted_ps_load_index(loads_eps)
        shard_load = load / shard_num
        for i in range(shard_num):
            loads_eps[sorted_index[i]] += shard_load

        return sorted_index[:shard_num]


def relocate_shard(ps_shard: dict, real_load: list, limit: int = 3) -> dict:
    logger.info(json.dumps(real_load, indent=4))
    LoadInfo = namedtuple("LoadInfo", "ps_index, load")
    new_real_load = list([LoadInfo(
        *(x)) for x in sorted(enumerate(real_load), key=lambda x: x[1], reverse=True)])
    n = len(real_load)
    max_ps_index = new_real_load[0].ps_index
    min_ps_index = new_real_load[-1].ps_index
    logger.info(json.dumps(new_real_load, indent=4))

    d = {}

    for i in range(limit):
        d[new_real_load[i].ps_index] = {
            'diff': (new_real_load[i].load - new_real_load[n - i - 1].load) / 2,
            'target': new_real_load[n - i - 1].ps_index,
            'arr': [],
            'total_load': 0.0,
            'load': new_real_load[i].load,
            'relocated_load': 0.0
        }

    for index in ps_shard:
        shard_list = ps_shard[index]['shard']
        for big_ps_index in d:
            if big_ps_index in shard_list:
                d[big_ps_index]['total_load'] += ps_shard[index]['load'] / \
                    len(shard_list)
            if big_ps_index in shard_list and d[big_ps_index]['target'] not in shard_list:
                d[big_ps_index]['arr'].append(index)

    logger.info(json.dumps(d, indent=4))

    for ps_index in d:
        cur_load = 0.0
        relocated_load = d[ps_index]['total_load'] * \
            (d[ps_index]['diff'] / d[ps_index]['load'])

        for index in d[ps_index]['arr']:
            shard_list = ps_shard[index]['shard']
            ps_shard[index]['shard'][shard_list.index(
                ps_index)] = d[ps_index]['target']
            cur_load += ps_shard[index]['load'] / len(shard_list)
            if cur_load >= relocated_load:
                break

    return ps_shard
