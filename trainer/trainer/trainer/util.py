# -*- coding: utf-8 -*-
# Author: dongxing@kuaishou.com
import os
import sys
import sh
import io
import re
import json
import logging
import psutil
import tensorflow as tf
from datetime import datetime, timedelta
import zstandard as zstd
import codecs
import base64
import numpy as np
import time
import random
import traceback
import pyarrow

import pyarrow
from pyarrow import hdfs
from collections import defaultdict

from . import dist

dctx = zstd.ZstdDecompressor()
cctx = zstd.ZstdCompressor(level=22)

LOG_FORMAT = "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s - %(funcName)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def get_root_dir():
    return os.path.abspath(os.path.join(os.path.abspath(__file__), '../../'))


class FeatureInfo:

    def __init__(
        self,
        class_name: str = '',
        category: str = '',
        field: int = -1,
        size: int = -1,
        capacity: int = -1
    ) -> None:
        self.class_name = class_name
        self.category = category
        self.field = field
        self.size = size
        self.capacity = capacity


def get_variable(
    name: str,
    shape: tuple,
    initializer,
    trainable: bool = True,
    is_cpu_ps: bool = True
):
    """Helper to create a Variable stored on CPU memory."""
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        if is_cpu_ps:
            with tf.device('/cpu:0'):
                v = tf.get_variable(name,
                                    shape,
                                    initializer=initializer,
                                    trainable=trainable)
        else:
            v = tf.get_variable(name,
                                shape,
                                initializer=initializer,
                                trainable=trainable)
    return v


def get_dense_vars():
    var_list = [
        v for v in tf.global_variables()
        if v.name.find('embedding_') < 0 and v.name.find('global_step') < 0 and
        v.name.find('out_of_range_ind') < 0 and v.shape.ndims > 0
    ]

    return var_list


def get_restore_vars(restore_mode):
    var_list = []
    if restore_mode == 0:
        # load hash model
        var_list = [
            v for v in tf.global_variables() if v.name.find('global_step') < 0
        ]
    elif restore_mode == 1:
        # load no hash model
        var_list = [
            v for v in tf.global_variables() if v.name.find('embedding_') < 0
        ]
    else:
        logger.critical("restore mode error! restore failed!")
    return var_list


def get_tf_session_config(
    per_process_gpu_memory_fraction: float = 0.7,
    tf_intra_op_parallelism_threads: int= 0,
    tf_inter_op_parallelism_threads: int= 0,
    use_xla: bool = True
):
    if tf_intra_op_parallelism_threads > 0 or tf_inter_op_parallelism_threads > 0:
        sess_config = tf.ConfigProto(
            device_count={"CPU": 15},
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=tf_intra_op_parallelism_threads,
            inter_op_parallelism_threads=tf_inter_op_parallelism_threads)
    else:
        sess_config = tf.ConfigProto(device_count={"CPU": 15},
                                     allow_soft_placement=True,
                                     log_device_placement=False)
    if use_xla:
        sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    sess_config.gpu_options.visible_device_list = str(dist.local_rank())
    sess_config.gpu_options.force_gpu_compatible = True

    return sess_config


def check_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info("create dir: %s", path)


def get_cmd_hadoop():
    try:
        return sh.Command("/home/hadoop/software/hadoop/bin/hadoop")
    except Exception as e:
        logger.info('no such command, %s', e)
        return None


cmd_hadoop = get_cmd_hadoop()


def get_calc_load():
    try:
        return sh.Command(os.path.join(get_root_dir(), 'calc_load'))
    except Exception as e:
        logger.info('no such command, %s', e)
        return None


cmd_calc_load = get_calc_load()


def set_trainer_cpu_affinity():
    local_size = dist.local_size()
    local_rank = dist.local_rank()

    p = psutil.Process()  # current process
    all_cpus = list(range(psutil.cpu_count()))
    cpus_per_trainer = int(len(all_cpus) / local_size)

    alloc_cpus = all_cpus[(local_rank *
                           cpus_per_trainer):(local_rank * cpus_per_trainer +
                                              cpus_per_trainer)]
    p.cpu_affinity(alloc_cpus)

    trainer_cpu_affinity = p.cpu_affinity()
    trainer_cpu_affinity = [str(x) for x in trainer_cpu_affinity]
    logger.info("trainer%d run at cpus[%s]", local_rank,
                ",".join(trainer_cpu_affinity))


def get_lines(
    filename: str,
    host: str = "viewfs://hadoop-lt-cluster",
    limit: int = 100,
    user: str = "ad"
) -> str:
    fs = hdfs.connect(host=host, user=user)
    logger.info('filename: %s', filename)
    with fs.open(filename, 'rb') as f:
        try:
            reader = io.BufferedReader(f, 1024 * 1024 * 8)
            file_finished = False
            i = 0
            while (i < limit and not file_finished):
                logger.info('readline:{}'.format(i))
                line = reader.readline()
                if not line:
                    file_finished = True
                    continue
                i += 1
                yield line
        except Exception as e:
            file_finished = True
            logger.info(e)


def wrap_ips(ips, port):
    # type: (str, int) -> str
    return ['%s:%s' % (x, port) for x in ips.split(',') if len(x) > 0]


class PsBalance:
    def __init__(
        self,
        limit: int = 100,
        debug: bool = False
    ):
        self._has_debug_info = None
        self._sparse_features = []
        self._enable_format_opt = True
        self._limit = limit
        self._debug = debug

    def _guess_debug_info(
        self,
        dp: str,
        sparse_field_offsets: [int],
        idx: int
    ) -> bool:
        end = None if len(sparse_field_offsets) == 1 else idx + \
            sparse_field_offsets[1]
        try:
            ps_fea = dctx.decompress(dp[idx + sparse_field_offsets[0]:end])
            return False
        except Exception as e:
            try:
                end = None if end == None else end + 4
                ps_fea = dctx.decompress(dp[idx + 4 +
                                            sparse_field_offsets[0]:end])
                return True
            except Exception as e:
                raise Exception('wrong batch format!!!')

    def set_sparse_features(self, features: [str]):
        self._sparse_features = features

    def parse_simple_features(self, dp):
        """Parse SimpleFeatures.
        """
        # TODO: parse SimpleFeatures.
        feat = KafkaFeatureLog()
        feat.ParseFromString(dp)

        # ignore filter
        log_features = feat.log_info.log_features

        stat = {}
        for item_info in feat.item_infos:
            featnames = [x.class_name for x in self._sparse_features]
            for i, featname in enumerate(featnames):
                if featname in log_features:
                    val = log_features[featname]
                elif featname in item_info.item_features:
                    val = item_info.item_features[featname]
                else:
                    return {}

                if len(val.id) != len(val.val):
                    return {}

                for sign in val.id:
                    if i not in stat:
                        stat[i] = {}

                    if sign not in stat[i]:
                        stat[i][sign] = 0

                    stat[i][sign] += 1

        ret = {}
        for i, data in stat.items():
            total_cnt = 0
            for cnt in data.values():
                total_cnt += cnt
            ret[i] = {'unique_cnt': len(data), 'total_cnt': total_cnt}

        return ret

    def print_debug_info(
        self,
        dp: bytes,
        idx: int,
        field_count: int,
        batch_size: int,
        sparse_field_offsets: [int]
    ):
        if self._has_debug_info == True:
            idx += sparse_field_offsets[field_count]
            debug_sizes = np.fromstring(dp[idx:idx + batch_size * 4],
                                        dtype=np.int32)
            idx += batch_size * 4
            debug_info = []
            logger.info('debug_sizes: %s', debug_sizes)
            for i in range(batch_size):
                logger.info('i: %s, %s', i, dp[idx:idx + debug_sizes[i]])
                debug_info.append(dp[idx:idx + debug_sizes[i]])
                idx += debug_sizes[i]

    def process(self, line):
        dp = base64.b64decode(line)
        return self.parse_simple_features(dp)

    def compute_load(
        self,
        path: str,
        compute_type: str = 'mean',
        share_dict: dict = {}
    ) -> dict:
        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(path):
            raise Exception('path: %s not exists' % (path))

        n = 0
        info = fs.ls(path)
        d = defaultdict(int)
        one_file = [
            x for x in info if os.path.basename(x) != '_SUCCESS' and
            not os.path.basename(x).startswith('.')
        ][0]
        logger.info('one_file_name:{}'.format(one_file))
        d = defaultdict(int)
        try:
            for line in get_lines(one_file, limit=self._limit):
                logger.info('len(line): %d', len(line))
                load_dict = self.process(line)
                for index in load_dict:
                    unique_cnt = load_dict[index]["unique_cnt"]
                    if int(index) in share_dict:
                        index = share_dict[int(index)]
                    if compute_type == 'mean':
                        d[int(index)] += int(unique_cnt)
                    elif compute_type == 'max':
                        d[int(index)] = max(unique_cnt, d[int(index)])
                    elif compute_type == 'last':
                        d[int(index)] = unique_cnt
                    else:
                        raise Exception(
                            'unsupported compute_type: %s, candidates: sum, max, last!'
                            % (compute_type))
                n += 1
        except sh.ErrorReturnCode_1 as e:
            logger.info(e)
        except Exception as e1:
            logger.info(e1)
            traceback.print_exc()

        sum = 0
        for k in d:
            sum += d[k]
        mean = sum * 1.0 / n
        logger.info('stat_info: total_sparse_id[%d]|batches[%d]|mean[%f]', sum,
                    n, mean)
        if compute_type == 'mean':
            for k in d:
                d[k] = d[k] * 1.0 / n
        logger.info('done, filename: %s, get %d load, load: %s', one_file,
                    len(d), json.dumps(d, indent=4))
        return d

    def get_sorted_ps_load_index(self, loads_eps: [int]):
        return [
            i[0] for i in sorted(list(loads_eps.items()), key=lambda x: x[1])
        ]

    def alloc_ps_num(
        self,
        load: [int],
        ps_max_num: int,
        loads_eps: [int]
    ) -> [int]:
        shard_num = 0
        if (load >= 40000.0):
            shard_num = min(8, ps_max_num)
        elif (load >= 20000.0):
            shard_num = min(4, ps_max_num)
        elif (load >= 10000.0):
            shard_num = min(2, ps_max_num)
        else:
            shard_num = 1
        sorted_index = self.get_sorted_ps_load_index(loads_eps)
        shard_load = max(1.0, load / shard_num)
        for i in range(shard_num):
            loads_eps[sorted_index[i]] += shard_load

        return sorted_index[:shard_num]

    def get_squared_max_ps(self, ps_nums: int) -> int:
        squared_max_ps = 1
        while squared_max_ps < ps_nums:
            if squared_max_ps * 2 > ps_nums:
                break
            squared_max_ps *= 2

        return squared_max_ps

    def get_ps_shard(
        self,
        path: str,
        compute_type: str,
        share_dict: dict
    ) -> dict:
        d = self.compute_load(path, compute_type, share_dict)

        loads_eps = {}
        all_ps = wrap_ips(os.environ.get('PS', ''), 5800)
        ps_nums = len(all_ps)
        for i in range(ps_nums):
            loads_eps[i] = 0.0

        squared_max_ps = self.get_squared_max_ps(ps_nums)

        ps_shard = {}
        for x in d:
            ps_shard[x] = {
                'load': d[x],
                'shard': self.alloc_ps_num(d[x], squared_max_ps, loads_eps)
            }

        logger.info('ps_shard: %s', json.dumps(ps_shard, indent=4))
        shard_detail = {}
        for ps_name in all_ps:
            shard_detail[ps_name] = {
                'ps_name': ps_name,
                'total_load': 0.0,
                'emb_table': []
            }
        for x in ps_shard:
            for p in ps_shard[x]['shard']:
                shard_detail[
                    all_ps[p]]['total_load'] += ps_shard[x]['load'] / len(
                        ps_shard[x]['shard'])
                shard_detail[all_ps[p]]['emb_table'].append({
                    'name': x,
                    'load': ps_shard[x]['load'] / len(ps_shard[x]['shard'])
                })

        for ps_name in shard_detail:
            shard_detail[ps_name]['emb_table'].sort(key=lambda x: x['load'],
                                                    reverse=True)

        shard_detail_arr = list(shard_detail.values())
        shard_detail_arr.sort(key=lambda x: x['total_load'], reverse=True)

        logger.info('shard_detail: %s', json.dumps(shard_detail_arr, indent=4))

        return ps_shard


class AdaptPsBalance(PsBalance):
    def __init__(
        self,
        ratio_unique_cnt: float,
        ratio_total_cnt: float,
        limit: int = 100
    ):
        self._ratio_unique_cnt = ratio_unique_cnt
        self._ratio_total_cnt = ratio_total_cnt
        self._has_debug_info = None
        self._mean_load = 0.0
        self._limit = limit

    def compute_load(
        self,
        path: str,
        compute_type: str = '',
        share_dict: dict = {}
    ) -> dict:
        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(path):
            raise Exception('path: %s not exists' % (path))

        n = 100
        info = fs.ls(path)
        d = defaultdict(int)
        one_file = [
            x for x in info if os.path.basename(x) != '_SUCCESS' and
            not os.path.basename(x).startswith('.')
        ][0]
        logger.info('one_file_name:{}'.format(one_file))
        try:
            for line in get_lines(one_file, limit=self._limit):
                logger.info('len(line): %d', len(line))
                load_dict = self.process(line)
                for index in load_dict:
                    if int(index) in share_dict:
                        index = share_dict[int(index)]
                    load = self._ratio_unique_cnt * \
                        load_dict[index]['unique_cnt'] + \
                        self._ratio_total_cnt * load_dict[index]['total_cnt']
                    if compute_type == 'mean':
                        d[int(index)] += int(load)
                    elif compute_type == 'max':
                        d[int(index)] = max(load, d[int(index)])
                    elif compute_type == 'last':
                        d[int(index)] = load
                    else:
                        raise Exception(
                            'unsupported compute_type: %s, candidates: sum, max, last!'
                            % (compute_type))
        except sh.ErrorReturnCode_1 as e:
            logger.info(e)
        except Exception as e1:
            logger.info(e1)

        if compute_type == 'mean':
            for k in d:
                d[k] = d[k] * 1.0 / n

        self._mean_load = sum(list(d.values())) / len(d)

        logger.info('done, filename: %s, get %d load', one_file, len(d))
        logger.info(d)

        return d

    def alloc_ps_num(
        self,
        load: float,
        ps_max_num: int,
        loads_eps: [float]
    ) -> [int]:
        shard_num = 0
        if load >= self._mean_load:
            shard_num = ps_max_num
        else:
            shard_num = 1
        sorted_index = self.get_sorted_ps_load_index(loads_eps)
        shard_load = max(1.0, load / shard_num)
        for i in range(shard_num):
            loads_eps[sorted_index[i]] += shard_load

        return sorted_index[:shard_num]


def transfer_load_json(
    json_path: str,
    new_json_path: str,
    key_ps_num: int,
    other_ps_lists: [int]
):
    """Transfer variables from ps whose load is large to those ps whose load is low.

    Select the variables from most stressful ps, then transfer them to the least stressful ps.

    Args:
        json_path: the old shard json.
        new_json_path: the new shard json.
        key_ps_num: index of the stressul ps.
        other_ps_lists: the target ps.

    Returns:
        None.
    """
    emb_tab = json.load(codecs.open(json_path, 'r', 'utf-8'))
    index = [[], [], [], [], [], [], [], []]

    for i in emb_tab:
        if len(emb_tab[i]['shard']) == 1:
            index[int(emb_tab[i]['shard'][0])].append(i)

    le = len(index[key_ps_num])

    # Shuffle the embedding which is assigned to most stressful ps.
    shuffle_index = [
        random.choice(other_ps_lists + [key_ps_num]) for _ in range(le)
    ]

    logger.info(shuffle_index)

    map_c = defaultdict(list)
    for index, j in enumerate(index[key_ps_num]):
        emb_tab[j]['shard'] = [shuffle_index[index]]
        map_c[str(shuffle_index[index])].append(j)

    logger.info("map key_ps_num_{}:{}".format(key_ps_num, map_c))

    json.dump(emb_tab,
              codecs.open(new_json_path, 'w', 'utf-8'),
              ensure_ascii=False,
              indent=2)


def master_rank(f):
    """Wrapper for the function that must use the json from rank 0.

    `f` must return json. When the json content is stored into file, all keys will be converted to
    string. Other rank need to convert the string key back to int while loading the json.

    Args:
        f: Funtion that will handle the json content.

    Returns:
        wrapper: The wrapper function who will return the converted json.
    """
    def wrapper(*args, **kwargs):
        func_name = f.__name__
        filename = 'data/%s_result.json' % (func_name)
        check_dir('data')

        if dist.rank() == 0:
            d = {}
            if os.path.exists(filename):
                os.remove(filename)

            d['res'] = f(*args, **kwargs)
            d['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            json.dump(d,
                      codecs.open(filename, 'w', 'utf-8'),
                      ensure_ascii=False,
                      indent=4)

            return d['res']
        else:
            for i in range(10000):
                time.sleep(5)
                if os.path.exists(filename):
                    d = json.load(open(filename))
                    if 'time' in d and 'res' in d:
                        delta = datetime.now() - \
                            datetime.strptime(d['time'], '%Y-%m-%d %H:%M:%S')
                        if (delta.days * 24 * 3600 + delta.seconds) < 10:
                            res = d['res']
                            if type(res) == dict:
                                for k in list(res.keys()):
                                    if str(k).isdigit():
                                        res[int(k)] = res[k]

                            return res
            raise Exception('cannot get result from function %s' % (func_name))

    return wrapper


def check_file_exists(file_list: [str]) -> list:
    fs = pyarrow.HadoopFileSystem()

    res = []
    for path in file_list:
        if not fs.exists(path):
            logger.info('%s not exists, pass it', path)
            continue

        info = fs.info(path)
        if info['kind'] == 'file':
            res.append(path)
            continue
        logger.info('%s is not a file but a %s, pass it', path, info['kind'])

    return res


def get_files(
    path: str,
    start_dt: str = '',
    end_dt: str = ''
) -> list:
    """Get all hdfs filenames under path, and date in range between `start_dt` and `end_dt`.

    For example, `path/20200306`, `path/2020-03-06/1200`.
    """
    fs = pyarrow.HadoopFileSystem()

    if not fs.exists(path):
        logger.info('path: %s not exists', path)
        return []

    start_dt = str(start_dt).replace('-', '')
    end_dt = str(end_dt).replace('-', '')

    paths = [
        x for x in fs.ls(path, True)
        if not x['name'].split('/')[-1].startswith('.')
    ]

    if all([x['kind'] == 'file' for x in paths]):
        return [x['name'] for x in paths]

    assert len(start_dt
              ) >= 8, 'must provide start_dt, for example: 20200306, 2020030610'

    p_dt_hour = re.compile(r'([\-\d]{8,10})/?[^\d]*=?(\d{2,4})?')
    child = []
    for x in paths:
        if x['kind'] == 'directory':
            m = p_dt_hour.findall(x['name'].split('/')[-1])
            if len(m) >= 1 and (m[0][0].replace('-', '')[:8] >= start_dt[:8] and
                                m[0][0].replace('-', '')[:8] <= end_dt[:8]):
                logger.info('find name: %s', x['name'])
                child.append(x['name'])

    if len(child) == 0:
        return child

    res = []
    first = [
        x for x in fs.ls(child[0], True)
        if not x['name'].split('/')[-1].startswith('.')
    ]
    if all([x['kind'] == 'file' for x in first]):
        for x in child:
            res.extend(fs.ls(x, False))
        return res

    if len(first) > 0 and first[0][
            'kind'] == 'directory' and first[0]['name'].find('data.har') >= 0:
        for x in child:
            res.extend(fs.ls(x + '/data.har', False))
        return res

    if len(start_dt) == 8:
        start_dt = '%s00' % (start_dt)
    if len(end_dt) == 8:
        end_dt = '%s23' % (end_dt)

    res = []
    for date_path in child:
        tmp_paths = fs.ls(date_path, True)
        for hour_path in tmp_paths:
            m = p_dt_hour.findall('/'.join(hour_path['name'].split('/')[-2:]))
            logger.info(hour_path['name'])
            logger.info(m)
            if len(m) >= 1:
                s = '%s%s' % (m[0][0], m[0][1])
                if s.replace('-', '')[:10] >= start_dt[:10] and s.replace(
                        '-', '')[:10] <= end_dt[:10]:
                    res.extend(fs.ls(hour_path['name'], False))

    return res


def get_path_by_time(
    path: str,
    start_dt: str = '',
    end_dt: str = ''
) -> list:
    return get_files(path, start_dt, end_dt)


class PrintGateHook(tf.train.SessionRunHook):
    """Print gate hook when training.
    """
    def __init__(self, gate, print_interval=500):
        self._print_interval = print_interval
        self._gate = gate

    def begin(self):
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._print_interval == 0:
            return tf.train.SessionRunArgs([self._gate])

    def after_run(self, run_context, run_values):
        if self._step % self._print_interval == 0:
            value = run_values.results[0].reshape(-1)
            total_gate = value.shape[0]
            zero_gate = np.sum(value < 1e-3)
            logger.info("step:{}, GATE: zero rate {}/{}".format(
                self._step, zero_gate, total_gate))
            logger.info("value, {}".format(value))


class VarDecayHooks(tf.train.SessionRunHook):
    def __init__(
        self,
        print_tensor,
        interval: int = 500,
        decay_rate: int = 0.99
    ):
        """Initialize.

        Args:
            interval: Step count for each decay. Initial value of epsilon is 0.1，every interval
                decay `decay_rate`. The final value of epsilon should be `1e-3 ~ 1e-4`.
            decay_rate: A float number smaller than 1.0.
        """
        self._tensor = print_tensor
        self._interval = interval
        self.update = tf.assign(self._tensor,
                                tf.maximum(self._tensor * decay_rate, 1.0e-5))

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._interval == 0:
            return tf.train.SessionRunArgs([self.update, self._tensor])

    def after_run(self, run_context, run_values):
        if self._step % self._interval == 0:
            value = run_values.results[1]
            logger.info("epsilon g_step:{}, var_name:{}, var_value:{}".format(
                self._step, self._tensor.name, value))


def get_category_info() -> dict:
    category_info = {
        'sparse_aggregate_user': {
            'priority': 0,
            'simple': 'user'
        },
        'sparse_user': {
            'priority': 0,
            'simple': 'user'
        },
        'sparse_item': {
            'priority': 1,
            'simple': 'photo'
        },
        'sparse_combine': {
            'priority': 2,
            'simple': 'combine'
        },
        'dense_aggregate_user': {
            'priority': 3,
            'simple': 'dense_user'
        },
        'dense_user': {
            'priority': 3,
            'simple': 'dense_user'
        },
        'dense_item': {
            'priority': 4,
            'simple': 'dense_photo'
        },
        'dense_combine': {
            'priority': 5,
            'simple': 'dense_combine'
        }
    }

    return category_info


def sort_features(features: dict) -> list:
    arr = [{**(features[name]), 'name': name} for name in features]

    category_info = get_category_info()
    arr.sort(key=lambda x: (category_info[x['meta']['type']]['priority'], x[
        'meta']['feature_id']))

    return arr


def check_feature_order(features: dict) -> bool:
    arr = sort_features(features)

    sparse_start, sparse_end = -1, -1
    dense_start, dense_end = -1, -1
    for i in range(len(arr)):
        if arr[i]['meta']['type'].startswith('sparse'):
            if sparse_start == -1:
                sparse_start = i
            sparse_end = i
        if arr[i]['meta']['type'].startswith('dense'):
            if dense_start == -1:
                dense_start = i
            dense_end = i

    for (start, end) in [(sparse_start, sparse_end), (dense_start, dense_end)]:
        if start == -1:
            continue
        assert arr[start]['meta'][
            'feature_id'] == 0, 'feature_id must start from 0!'
        last_feature_id = 0
        for i in range(start + 1, end + 1):
            if arr[i]['meta']['feature_id'] != arr[i -
                                                   1]['meta']['feature_id'] + 1:
                logger.erro(
                    'feature_id must be continuous!, i: %d, feature_id: %d, name: %s',
                    (i, arr[i]['meta']['feature_id'], arr[i]['name']))
                return False

    return True


def test_batch_data(filename: str, limit: int = 1000):
    logger.info('test batch data, filename: %s', filename)
    ps_balance = PsBalance(limit, True)
    ps_balance.compute_load(filename)


def run_command(cmd: [str]):
    logger.info("cmd: %s", ' '.join(cmd))

    os.system(' '.join(cmd))
