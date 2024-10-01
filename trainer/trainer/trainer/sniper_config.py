# -*- coding: utf-8 -*-

import os
import sys
import codecs
import sh
import time
from collections import defaultdict
import yaml
import hashlib
import pyarrow
from datetime import datetime, timedelta
import random
import json
import tensorflow as tf
from tensorflow import dtypes
import io
from pyarrow import hdfs
from collections import defaultdict
import socket

from .util import check_dir, logger, get_root_dir, cmd_hadoop, cmd_calc_load
from .util import PsBalance, master_rank, wrap_ips, FeatureInfo, string_to_int_hash
from .ps_shard import LockBalance

from .dist import rank


def _load_op():
    """Load trainer_ops.so。
    """
    basename = 'trainer_ops.so'
    so_file = os.path.join(os.path.dirname(__file__), basename)
    return tf.load_op_library(so_file)


_trainer_ops_so = _load_op()


def get_base_conf():
    logger.info("root: %s", get_root_dir())

    return json.load(
        codecs.open(os.path.join(get_root_dir(), 'trainer/base_config.json')))


@master_rank
def compute_ps_shard_base(path: str,
                          compute_type: str,
                          share_dict: dict = {},
                          feature_config: str = "",
                          enable_format_opt: bool = True,
                          limit: int = 100) -> dict:
    """Main logic for computing ps shard.
    """
    ps_balance = PsBalance(limit=limit)

    if len(feature_config) > 0:
        d = parse_feature_config(feature_config, [0])

        sparse_features = []
        for sparse in d['feature_column']:
            if sparse['class'] == 'embedding_column':
                emb_table = d['embedding_table'][sparse['embedding_column']
                                                 ['emb_table']]

                feature_info = FeatureInfo(class_name=sparse['class_name'],
                                           capacity=emb_table['capacity'])

                sparse_features.append(feature_info)

        ps_balance.set_sparse_features(sparse_features)
        ps_balance.set_enable_format_opt(enable_format_opt)

    return ps_balance.get_ps_shard(path, compute_type, share_dict)


def compute_ps_shard(path: str,
                     share_dict: dict = {},
                     feature_config: str = "",
                     enable_format_opt: bool = True,
                     limit: int = 100) -> dict:
    return compute_ps_shard_lock(path, share_dict, feature_config,
                                 enable_format_opt, limit)


def compute_ps_shard_max(path: str,
                         share_dict: dict = {},
                         feature_config: str = "",
                         enable_format_opt: bool = True,
                         limit: int = 100) -> dict:
    return compute_ps_shard_base(path, "max", share_dict, feature_config,
                                 enable_format_opt, limit)


def compute_ps_shard_last(path: str,
                          share_dict: dict = {},
                          feature_config: str = "",
                          enable_format_opt: bool = True,
                          limit: int = 100) -> dict:
    return compute_ps_shard_base(path, "last", share_dict, feature_config,
                                 enable_format_opt, limit)


def compute_ps_shard_mean(path: str,
                          share_dict: dict = {},
                          feature_config: str = "",
                          enable_format_opt: bool = True,
                          limit: int = 100) -> dict:
    return compute_ps_shard_base(path, "mean", share_dict, feature_config,
                                 enable_format_opt, limit)


@master_rank
def compute_ps_shard_lock(path: str,
                          share_dict: dict = {},
                          feature_config: str = "",
                          enable_format_opt: bool = True,
                          limit: int = 100) -> dict:
    ps_balance = LockBalance(limit=limit)
    if len(feature_config) > 0:
        d = parse_feature_config(feature_config, [0])
        sparse_features = []
        for sparse in d['feature_column']:
            if sparse['class'] == 'embedding_column':
                emb_table = d['embedding_table'][sparse['embedding_column']
                                                 ['emb_table']]

                feature_info = FeatureInfo(class_name=sparse['class_name'],
                                           capacity=emb_table['capacity'])

                sparse_features.append(feature_info)

        ps_balance.set_sparse_features(sparse_features)
        ps_balance.set_enable_format_opt(enable_format_opt)

    return ps_balance.get_ps_shard(path, '', share_dict)


def fix_json_key(d: dict) -> dict:
    """Convert str key into int key in json.

    Because when we save json to file, int key will be converted to string key automatically.
    We need to convert string key back to int key when restore from json file.
    """
    new_d = {}

    for k in d:
        new_d[k] = d[k]
        if str(k).isdigit():
            new_d[int(k)] = d[k]

    return new_d


def if_set_dup(dup_set: set, field) -> bool:
    before_len = len(dup_set)
    dup_set.add(field)
    return before_len == len(dup_set)


def assert_add_set(category: str, dup_sets: list, field):
    if category == "user":
        assert not if_set_dup(
            dup_sets[0], field), "There is a duplicate sparse field %s" % field
    elif category == "reco_user":
        assert not if_set_dup(
            dup_sets[0], field), "There is a duplicate sparse field %s" % field
    elif category == "photo":
        assert not if_set_dup(
            dup_sets[0], field), "There is a duplicate sparse field %s" % field
    elif category == "reco_photo":
        assert not if_set_dup(
            dup_sets[0], field), "There is a duplicate sparse field %s" % field
    elif category == "combine":
        assert not if_set_dup(
            dup_sets[0], field), "There is a duplicate sparse field %s" % field
    elif category == "dense_user":
        assert not if_set_dup(
            dup_sets[1], field), "There is a duplicate dense field %s" % field
    elif category == "dense_photo":
        assert not if_set_dup(
            dup_sets[1], field), "There is a duplicate dense field %s" % field
    elif category == "dense_combine":
        assert not if_set_dup(
            dup_sets[1], field), "There is a duplicate dense field %s" % field


def get_ps_shard_index(ps_shard: dict, ps: []) -> dict:
    d = {}

    for x in ps_shard:
        d[x] = {
            'load': ps_shard[x]['load'],
            'shard': [ps.index(v) for v in ps_shard[x]['shard']]
        }

    return d


def parse_feature_lines(feature_file_lines: list,
                        ps: list,
                        ps_shard: dict = {},
                        embedding_size: int = 16) -> dict:
    ps_shard = fix_json_key(ps_shard)

    d = {
        'feature_column': [],
        'embedding_table': {},
        'network': {
            'inputs': {
                'sparse': [],
                'sparse_index': [],
                'emb_table': [],
                'dense': [],
                'dense_index': []
            }
        }
    }

    map_class_name = {}
    common_configs = {}
    common_configs['hash_type'] = 'djb2_hash64'
    common_configs['ps_hash'] = 0

    # slot ==> embedding_table name
    slot_configs = {}

    # sets[0] is for sparse sets[1] is for dense
    assert_dup_sets = [set(), set()]

    for line in feature_file_lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        seg = line.split(",")

        field = -1
        size = -1
        dense = False
        category = -1
        category_name = ""
        class_name = ""
        prefix = -1
        need_update = 1
        emb_size = embedding_size
        capacity = -1
        slot = -1

        for i in range(0, len(seg)):
            seg1 = seg[i].split("=")
            if seg1[0].strip() == "hash_type":
                common_configs['hash_type'] = seg1[1].strip()
            if seg1[0].strip() == "ps_hash":
                common_configs['ps_hash'] = int(seg1[1].strip())
            if seg1[0].strip() == "slot":
                slot = int(seg1[1])
            if seg1[0].strip() == "capacity":
                capacity = int(seg1[1])
            if seg1[0].strip() == "field":
                field = int(seg1[1])
            if seg1[0].strip() == "size":
                size = int(seg1[1])
            if seg1[0].strip() == "class":
                class_name = seg1[1].strip()
            if seg1[0].strip() == "prefix":
                prefix = int(seg1[1].strip())
            if seg1[0].strip() == "need_update":
                need_update = int(seg1[1].strip())
            if seg1[0].strip() == "emb_size":
                emb_size = int(seg1[1].strip())
            if seg1[0].strip() == "category":
                tmp = seg1[1].strip()
                category_name = tmp

                if tmp == "user":
                    category = 0
                elif tmp == "reco_user":
                    category = 0
                elif tmp == "photo":
                    category = 1
                elif tmp == "reco_photo":
                    category = 1
                elif tmp == "combine":
                    category = 2
                elif tmp == "dense_user":
                    category = 3
                    dense = True
                elif tmp == "dense_photo":
                    category = 4
                    dense = True
                elif tmp == "dense_combine":
                    category = 5
                    dense = True
                else:
                    logger.info("unsupported category %s" % seg)

        if len(class_name) == 0:
            continue

        assert_add_set(category_name, assert_dup_sets, field)
        if capacity == -1:
            capacity = size

        if dense:
            d['feature_column'].append({
                "class": "numeric_column",
                "class_name": class_name,
                "numeric_column": {
                    "dim": size,
                },
                "attrs": {
                    "category": category_name,
                    "prefix": prefix,
                    "field": field
                }
            })

            d['network']['inputs']['dense'].append(class_name)
            dense_index = len(d['feature_column']) - 1
            d['network']['inputs']['dense_index'].append(dense_index)
        else:
            assert not (slot != -1 and prefix != -1
                       ), "slot and prefix can not both have values"
            map_class_name[field] = class_name
            emb_table = 'embedding_%d' % (field)
            class_type = 'embedding_column'

            if slot != -1 and slot not in slot_configs:
                slot_configs[slot] = {"emb_table": emb_table, "field": field}

            share_emb_table = emb_table
            share_info = {}

            if slot != -1:
                class_type = 'seq_column'
                share_info = slot_configs[slot]
                share_emb_table = share_info['emb_table']

            d['network']['inputs']['sparse'].append(class_name)
            sparse_index = len(d['network']['inputs']['sparse']) - 1
            d['network']['inputs']['sparse_index'].append(sparse_index)

            if class_type == 'seq_column':
                d['feature_column'].append({
                    "class": class_type,
                    "class_name": class_name,
                    class_type: {
                        "emb_table": share_emb_table,
                        "meta_emb_table": emb_table
                    },
                    "attrs": {
                        "category": category_name,
                        "share_field": share_info["field"],
                        "share_slot": slot
                    },
                    "sparse_index": sparse_index,
                })
                d['network']['inputs']['emb_table'].append(share_emb_table)
            else:
                d['feature_column'].append({
                    "class": class_type,
                    "class_name": class_name,
                    class_type: {
                        "emb_table": emb_table
                    },
                    "attrs": {
                        "category": category_name,
                        "prefix": prefix,
                        "field": field
                    },
                    "sparse_index": sparse_index,
                })
                d['network']['inputs']['emb_table'].append(emb_table)

            if share_emb_table == emb_table:
                if capacity == -1:
                    capacity = size
                d['embedding_table'][emb_table] = {
                    "dim": emb_size,
                    "capacity": capacity,
                    "hash_bucket_size": size,
                    "hash_func": "default",
                    "fields": [],
                }

            d['embedding_table'][share_emb_table]["fields"].append(field)

    if len(ps_shard) > 0:
        for x in range(len(d['network']['inputs']['sparse'])):
            key = 'embedding_%s' % (x)
            if key not in d['embedding_table']:
                logger.info("embedding_table[%s] not exsits" % key)
                continue

            if x in ps_shard:
                d['embedding_table'][key]['load'] = ps_shard[x]['load']

                if any([int(v) >= len(ps) for v in ps_shard[x]['shard']]):
                    raise Exception(
                        'ps_index is bigger than ps num: %s, key: %s' %
                        (len(ps), key))

                d['embedding_table'][key]['shard'] = [
                    ps[int(v)] for v in ps_shard[x]['shard']
                ]
            else:
                d['embedding_table'][key]['load'] = 1.0

                h = int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % len(ps)

                d['embedding_table'][key]['shard'] = [ps[h]]
                ps_shard[x] = {'load': 1.0, 'shard': [h]}
    else:
        for key in d['embedding_table']:
            # h = int(hashlib.sha1(key.encode('utf-8')).hexdigest(), 16) % len(ps)
            h = string_to_int_hash(key) % len(ps)
            d['embedding_table'][key]['shard'] = [ps[h]]
            d['embedding_table'][key]['load'] = 0.0

            ps_shard[int(key.split('_')[1])] = {'load': 0.0, 'shard': [h]}
            logger.info('name: %s, idx: %d, ps: %s', key, h,
                        d['embedding_table'][key]['shard'])

    for k in d['embedding_table']:
        shard_config = d['embedding_table'][k]['shard']
        shard_num = len(shard_config)
        assert shard_num > 0, '%s do not have shard_config!'
        x = 1
        while x <= shard_num:
            if x != shard_num:
                x <<= 1
            if x == shard_num:
                break
        assert x == shard_num, 'var[%s] shard_num[%d] is not equal to 2^n' % (
            k, shard_num)

    d['final_ps_shard'] = ps_shard
    d['common_configs'] = common_configs
    d['slot_configs'] = slot_configs
    return d


def parse_feature_config(feature_config: list,
                         ps: list,
                         ps_shard: dict = {}) -> dict:
    feature_file_lines = codecs.open(feature_config, 'r', 'utf-8').readlines()
    return parse_feature_lines(feature_file_lines, ps, ps_shard)


def parse_conf_from_file(filename):
    d = {}
    d['model_name'] = os.path.basename(filename).split('.')[0]
    d['filename'] = filename

    if filename.endswith('.config'):
        for line in codecs.open(filename, 'r', 'utf-8').read().split('\n'):
            if line.startswith('--') and line.find('=') > 0:
                arr = line[2:].strip().split('=', 2)
                if len(arr) >= 2:
                    d[arr[0]] = ('='.join(arr[1:])).strip()

    elif (filename.endswith('.yaml') or filename.endswith('.yml')):
        d = yaml.load(open(filename), Loader=yaml.FullLoader)

    elif filename.endswith('.json'):
        d = json.load(open(filename))

    else:
        raise Exception(
            'config file must endswith one of .config, .yaml, .yml, .json, but is %s'
            % (filename))

    return d


def change_config(config):
    if config.get('check', True):
        check_model_name(config['model_name'], config['user_name'],
                         config['aim'])

    if config['kafka_train'] or config['is_online']:
        env_model_name = os.environ.get('MODEL_NAME', '')
        assert len(
            env_model_name) > 0, 'must provide environment variable MODEL_NAME!'
        assert env_model_name.startswith(
            config['model_name']), 'MODEL_NAME must startswith %s' % (
                config['model_name'])

        config['model_export_root_path'] = '/home/ad/model'
        logger.info('is online, set model_export_root_path = /home/ad/model')

        if config['kafka_train']:
            assert len(config['topic']) > 0, 'must provide topic'
            now = datetime.now()
            config['group_id'] = 'klearn_online_%s_%s_%s_%s' % (
                config['user_name'], config['model_name'],
                now.strftime('%Y%m%d'), now.strftime('%H_%M_%S'))
    else:

        config['model_export_root_path'] = '/home/ad/model_offline'
        config['use_btq'] = False
        logger.info(
            'offline train, set model_export_root_path = /home/ad/model_offline, use_btq = False'
        )

    return config


def contains(large='', small=''):
    d1 = dict([(x, large.count(x)) for x in large])
    d2 = dict([(x, small.count(x)) for x in small])

    for k in small:
        if k not in d1 or d1[k] < d2[k]:
            return False

    return True


def get_all_aim():
    return ["dsp_conv", "dsp_lps"]


def check_model_name(model_name, user_name, aim):
    assert len(
        model_name
    ) > 0, 'model_name should be the name of config filename without suffix, but is empty!'
    assert len(
        model_name
    ) < 32, 'model_name cannot exceeds 32 chars, but is: %s, len: %s' % (
        model_name, len(model_name))

    assert len(user_name) > 0, 'must provide user_name !'
    assert len(aim) > 0, 'must provide aim !'

    all_aim = get_all_aim()

    assert model_name.startswith(
        aim), 'wrong model_name format, must start with aim, but is %s' % (
            model_name)

    if not any([contains(user_name, x) for x in model_name.split('_')]):
        raise Exception(
            'wrong model_name format, must contains user short name, but is %s'
            % (model_name))


def is_true(x):
    return str(x).lower() == 'true' or str(x).lower() == '1'


def get_share_dict(filename: str) -> dict:
    d = parse_conf_from_file(filename)
    feature_config = d['ks_feature_config_path']
    fd = None
    try:
        fd = open(feature_config, encoding='utf-8')
    except:
        fd = open(feature_config)
    share_dict = {}
    slot_index = {}
    for line in fd:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        seg = line.split(",")

        field = -1
        slot = -1
        class_name = ""
        dense = False

        for i in range(0, len(seg)):
            seg1 = seg[i].split("=")
            if seg1[0].strip() == "slot":
                slot = int(seg1[1])
            if seg1[0].strip() == "field":
                field = int(seg1[1])
            if seg1[0].strip() == "class":
                class_name = seg1[1].strip()
            if seg1[0].strip() == "category":
                tmp = seg1[1].strip()
                if tmp.find("dense") == 0:
                    dense = True
        if len(class_name) == 0:
            continue
        if dense:
            continue
        if slot == -1 or field == -1:
            continue
        if slot not in slot_index:
            slot_index[slot] = field
        share_dict[field] = slot_index[slot]
    return share_dict


def is_hash_embedding(embedding_table: dict) -> bool:
    for emb_table in embedding_table:
        if embedding_table[emb_table]["hash_bucket_size"] > 0:
            return True

    return False


def get_local_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    return ip_address


@master_rank
def convert_config(filename: str, ps_shard: dict = {}) -> str:
    logger.info('filename: %s', os.path.join(os.getcwd(), filename))
    conf = get_base_conf()

    d = parse_conf_from_file(filename)
    d['feature_file'] = d.get('ks_feature_config_path', '')
    d['topic'] = d.get('topic_id', '')

    for k in d:
        if k in conf['trainer']:
            if type(conf['trainer'][k]) == bool:
                conf['trainer'][k] = is_true(d[k])
            else:
                conf['trainer'][k] = (type(conf['trainer'][k]))(d[k])
            logger.info('k: %s, v: %s, conf: %s', k, d[k], conf['trainer'][k])
        else:
            conf['trainer'][k] = d[k]
    if conf['trainer']["optimizer"] == 'adam':
        assert conf['trainer']["optimizer"] == 'adam' and conf['trainer'][
            "export_mode"] == 1 and conf['trainer'][
                "restore_mode"] == 1, "adam can not use restore_mode = 1 or export_mode = 1 "

    conf['trainer'] = change_config(conf['trainer'])

    conf['sample']['batch_size'] = conf['trainer']['batch_size']
    conf['sample']['label_size'] = conf['trainer'].get('label_size', 1)

    # If is local, set ps and hub to local host.
    if conf['trainer']['is_local']:
        ip_address = get_local_ip()
        conf['clusterspec']['ps'] = wrap_ips(ip_address, 34000)
        conf['clusterspec']['hub'] = wrap_ips(ip_address, 35000)
    else:
        conf['clusterspec']['ps'] = wrap_ips(os.environ.get('PS', ''), 34000)
        conf['clusterspec']['hub'] = wrap_ips(os.environ.get('HUB', ''), 35000)

    conf['trainer']['dirname'] = os.path.join('/share/ad/',
                                              conf['trainer']['user_name'],
                                              'klearn_train',
                                              conf['trainer']['model_name'])

    if conf['trainer']['dragon_pipeline'] != '':
        dragon_pipeline = json.load(
            codecs.open(conf['trainer']['dragon_pipeline'], 'r', 'utf-8'))
        processor = dragon_pipeline['pipeline_manager_config']['base_pipeline'][
            'processor']

        ad_feature_processor = ''
        for x in processor:
            if processor[x]['type_name'] == 'AdFeatureEnricher':
                ad_feature_processor = x
                break

        feature_file_lines = codecs.open(conf['trainer']['feature_file'], 'r',
                                         'utf-8').readlines()

        conf['trainer']['dragon_pipeline_str'] = json.dumps(dragon_pipeline)

        logger.info('dragon config to feature file done, dragon_pipeline: %s',
                    conf['trainer']['dragon_pipeline'])
    else:
        feature_file_lines = codecs.open(conf['trainer']['feature_file'], 'r',
                                         'utf-8').readlines()

    # for debug mode
    if conf['trainer']['debug_mode']:
        conf['clusterspec']['ps'] = ['localhost:5800'] * 100
        conf['clusterspec']['hub'] = ['localhost:5900'] * 100

    feature_detail = parse_feature_lines(feature_file_lines,
                                         conf['clusterspec']['ps'], ps_shard,
                                         conf['trainer']['embedding_size'])

    conf.update(feature_detail)
    conf['trainer'].update(feature_detail['common_configs'])

    if conf['trainer']['queue_str_size'] == 0:
        if conf['trainer']['need_batch']:
            conf['trainer']['queue_str_size'] = conf['trainer'][
                'batch_size'] * conf['trainer']['hub_worker'] * 2
        else:
            conf['trainer'][
                'queue_str_size'] = conf['trainer']['hub_worker'] * 2
        logger.info('set queue_str_size: %d', conf['trainer']['queue_str_size'])

    nodes = ['stream', 'train_log_processor', 'feed_node']
    for node in nodes:
        key_num = 'hub_%s_num' % (node)
        if conf['trainer'][key_num] == 0:
            conf['trainer'][key_num] = conf['trainer']['hub_worker']

        key_size = 'hub_%s_size' % (node)
        if conf['trainer'][key_size] == 0:
            if conf['trainer']['need_batch']:
                conf['trainer'][key_size] = conf['trainer'][
                    'batch_size'] * conf['trainer'][key_num] * 2
            else:
                conf['trainer'][key_size] = conf['trainer'][key_num]

    default_group_name = os.path.basename(
        os.path.dirname(os.path.abspath(filename)))
    conf['trainer']['group_name'] = default_group_name
    conf['trainer']['pwd'] = os.getcwd()
    check_dir(conf['trainer']['dirname'])

    if not is_hash_embedding(conf['embedding_table']):
        conf['trainer']['use_param_vector'] = False
        logger.info("no hash mode, set use_param_vector = False")

    if len(ps_shard) > 0:
        conf['trainer']['use_auto_shard'] = False
        logger.info("already has ps_shard, set use_auto_shard = false")

    if conf['trainer']['use_auto_shard']:
        num_gpu = int(os.environ.get("MY_GPU", 1))
        if num_gpu > 1:
            raise Exception("gpu num must be 1 when use_auto_shard is true!")

    check_dir('data')
    logger.info('final conf: %s', json.dumps(conf, indent=4))

    conf_file = 'data/%s_modified_klearn_%d.json' % (d['model_name'],
                                                     rank())
    json.dump(conf,
              codecs.open(conf_file, 'w', 'utf-8'),
              ensure_ascii=False,
              indent=4)
    logger.info('pwd: %s, save conf to json: %s', os.getcwd(),
                os.path.abspath(conf_file))

    shard_filename = os.path.join(conf['trainer']['dirname'], 'ps_shard.json')
    json.dump(conf['final_ps_shard'],
              codecs.open(shard_filename, 'w', 'utf-8'),
              ensure_ascii=False,
              indent=4)
    logger.info('save ps_shard to path: %s', shard_filename)

    return conf_file


class SniperConf(object):
    """All config.
    """
    def __init__(self, conf_file):
        assert len(conf_file) > 0, 'must provide conf_file'

        self.conf_file = conf_file
        self.trainer_id = rank()

        logger.info("conf_file: %s", self.conf_file)

        self.conf = json.load(codecs.open(self.conf_file, 'r', 'utf-8'))
        self.conf.update(self.conf['trainer'])
        self.__dict__.update(self.conf['trainer'])

        # clusterspec
        self.trainer_eps = self.conf['clusterspec']['trainer']
        self.ps_eps = self.conf['clusterspec']['ps']
        self.hub_eps = self.conf['clusterspec']['hub']
        self.epsilion_decay_interval = self.conf['trainer'][
            'epsilion_decay_interval']

        # feature column
        self.feature_column = self.conf['feature_column']
        self.debug_info = self.conf['trainer']['debug_info']

        # embedding table
        self.embedding_table = self.conf['embedding_table']
        self.ps_shard = {
            x: self.embedding_table[x]['shard'] for x in self.embedding_table
        }

        # network - input
        self.sparse_input = self.conf['network']['inputs']['sparse']
        self.sparse_indices = self.conf['network']['inputs']['sparse_index']
        self.dense_input = self.conf['network']['inputs']['dense']
        self.dense_indices = self.conf['network']['inputs']['dense_index']

        self.input_dense_total_size = 0
        self.input_dense_user_count = 0
        self.input_dense_size = []
        self.dense_fields = []
        for i in range(len(self.dense_input)):
            dense = self.dense_input[i]
            dense_index = self.dense_indices[i]
            dim = self.feature_column[dense_index]['numeric_column']['dim']
            self.input_dense_size.append(dim)
            self.input_dense_total_size += dim
            self.dense_fields.append(dim)
            category = self.feature_column[dense_index]["attrs"]["category"]
            if category == "dense_user":
                self.input_dense_user_count += 1

        self.exclude_dense_feature = [
            int(x)
            for x in self.conf['exclude_dense_feature'].split(',')
            if len(x) > 0
        ]
        self.exclude_dense_fields = [
            self.dense_fields[x] for x in self.exclude_dense_feature
        ]
        self.exclude_dense_set = set(self.exclude_dense_feature)
        self.real_dense_fields = [
            self.dense_fields[x]
            for x in range(len(self.dense_fields))
            if x not in self.exclude_dense_set
        ]

        self.input_sparse_total_size = 0
        self.input_sparse_user_count = 0
        self.input_sparse_emb_table_name = []
        self.meta_emb_table_name = []
        self.input_sparse_emb_table_capacity = []
        self.input_sparse_emb_size = []
        self.fields = []
        for i in range(len(self.sparse_input)):
            sparse = self.sparse_input[i]
            sparse_index = self.sparse_indices[i]
            emb_table = None
            meta_emb_table = None
            if 'embedding_column' in self.feature_column[sparse_index]:
                emb_table = self.feature_column[sparse_index][
                    'embedding_column']['emb_table']
                meta_emb_table = emb_table
            else:
                emb_table = self.feature_column[sparse_index]['seq_column'][
                    'emb_table']
                meta_emb_table = self.feature_column[sparse_index][
                    'seq_column']['meta_emb_table']
            self.input_sparse_emb_table_name.append(emb_table)
            self.meta_emb_table_name.append(meta_emb_table)
            self.input_sparse_emb_table_capacity.append(
                self.embedding_table[emb_table]['capacity'])
            dim = self.embedding_table[emb_table]['dim']
            self.input_sparse_emb_size.append(dim)
            self.input_sparse_total_size += dim
            self.fields.append(
                self.embedding_table[emb_table]['hash_bucket_size'])
            category = self.feature_column[sparse_index]["attrs"]["category"]
            if category == "reco_user" or category == "user":
                self.input_sparse_user_count += 1

        self.print_interval = int(self.conf['trainer']['print_interval'])
        self.batch_size = int(self.conf['trainer']['batch_size'])
        self.slice_size = int(self.conf['trainer']['batch_size'])

        self.total_sparse_shard = sum([
            len(self.embedding_table[x]['shard']) for x in self.embedding_table
        ])

        self.var_2_btq = self.get_var_2_btq()

        self.btq_topic_to_vars = {}
        self.var_2_emb_shard = {}
        self.btq_sharding = False

        # checkpoint
        self.ckp_save_nfs_path = self.conf['trainer']['model_export_root_path']
        # nfs full interval in seconds.
        self.ckp_save_nfs_full_interval = self.conf['trainer']['ckp_save_nfs_full_interval']
        # restore path of parameters
        self.ckp_restore_nfs_path = self.conf['trainer']['ckp_restore_nfs_path']

        if len(self.ckp_restore_nfs_path) == 0 and len(self.warmup_path) != 0:
            self.ckp_restore_nfs_path = self.warmup_path

        if (len(self.ckp_save_nfs_path) == 0
            and len(self.model_export_root_path) != 0):
            self.ckp_save_nfs_path = self.model_export_root_path

        if self.ckp_save_nfs_full_interval == 0 and self.export_interval != 0:
            self.ckp_save_nfs_full_interval = self.export_interval

        self.split_input = self.conf['trainer']['split_input']
        self.debug_mode = self.conf['trainer']['debug_mode']
        self.use_xla = self.conf['trainer']['use_xla']
        self.mixed_precision = self.conf['trainer']['mixed_precision']
        self.aim = self.conf['trainer']['aim']
        self.debug_offline = bool(self.debug and not self.kafka_train)

        self.local_ckpt_dir = os.path.join(os.getcwd(), self.model_name, 'ckpt')
        check_dir(self.local_ckpt_dir)

    def add_config(self, **kwargs):
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def get_var_2_btq(self) -> dict:
        """Variable to btq queue name mapping.

        Btq is an internal p2p transfer platform. We use only `DEFAULT_BTQ_TOPIC` to be consistent
        with the interface.
        """

        return {
            'DEFAULT_BTQ_TOPIC': self.model_name,
        }
