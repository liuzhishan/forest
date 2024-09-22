# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import dtypes
from tensorflow import data
from tensorflow import dtypes
from tensorflow.python.framework import ops

from .sniper_config import SniperConf
from .util import logger
from .util import check_file_exists

__all__ = ['TrainerOps']


def _load_op():
    basename = 'trainer_ops.so'
    so_file = os.path.join(os.path.dirname(__file__), basename)
    return tf.load_op_library(so_file)


_trainer_ops_so = _load_op()


class TrainerOpsBase(object):
    """Wrapper for ops written in c++.
    """
    def __init__(self, sniper_conf):
        self._conf_file = sniper_conf.conf_file
        logger.info('conf_file: %s', self._conf_file)
        self._trainer_id = sniper_conf.trainer_id

        self._batch_size = sniper_conf.batch_size
        self._label_size = sniper_conf.label_size
        self._dense_total_size = sniper_conf.input_dense_total_size
        self._sparse_field_count = len(sniper_conf.input_sparse_emb_size)
        self._sparse_emb_size = sniper_conf.input_sparse_emb_size

        self._input_sparse_total_size = sniper_conf.input_sparse_total_size
        # batch_id, lables, dense, debug_info, list of embeddings
        self._input_signature = [
            tf.TensorSpec([1], dtypes.as_dtype(tf.int64)),
            tf.TensorSpec([self._label_size, self._batch_size],
                          dtypes.as_dtype(tf.int32)),
            tf.TensorSpec([self._batch_size, self._dense_total_size],
                          dtypes.as_dtype(tf.float32))
        ] + [
            tf.TensorSpec([self._batch_size, emb_size],
                          dtypes.as_dtype(tf.float32))
            for emb_size in self._sparse_emb_size
        ]

        if len(sniper_conf.debug_info) > 0:
            self._input_signature.append(
                tf.TensorSpec([self._batch_size], dtypes.as_dtype(tf.string)))

    def create_embedding_table(self):
        """Send request to ps for creating embedding table.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.create_embedding_table(
                self._conf_file, self._trainer_id)

    def create_dense_var(self, var, var_name: str):
        """Send request to ps for creating dense variable.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.create_dense(
                var,
                var_name,
                self._conf_file,
                self._trainer_id,
            )

    def start_sample(
        self,
        parallel: int,
        src: str,
        file_list: [str],
        work_mode: int
    ):
        """Send request to hub for starting sample.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.start_sample(
                parallel,
                src,
                file_list,
                work_mode,
                self._conf_file,
                self._trainer_id
            )

    def prefetch_dataset(self, work_mode):
        """Prefetch data and put the result into queue.

        The data is coming in order of batch_id, label, dense, debug_info, list of embeddings.
        """
        class PrefetchDataset(data.Dataset):
            def __init__(self, conf_file, trainer_id, input_signature,
                         work_mode):
                logger.info('python PrefetchDataset start')
                self._conf_file = conf_file
                self._trainer_id = trainer_id
                self._input_signature = input_signature
                self._work_mode = work_mode

                super(PrefetchDataset, self).__init__()

            def _inputs(self):
                return []

            def _as_variant_tensor(self):
                return _trainer_ops_so.input_prefetch_dataset(
                    self._conf_file,
                    self._trainer_id,
                    self._work_mode,
                    [x.dtype for x in self._input_signature],
                    [x.shape for x in self._input_signature]
                )

            @property
            def output_classes(self):
                return tuple([tf.Tensor] * len(self._input_signature))

            @property
            def output_shapes(self):
                return tuple([x.shape for x in self._input_signature])

            @property
            def output_types(self):
                return tuple([x.dtype for x in self._input_signature])

        return PrefetchDataset(
            self._conf_file,
            self._trainer_id,
            self._input_signature,
            work_mode
        )

    def push_grad(
        self,
        batch_id: int,
        var_list: list,
        gradient: float,
        eta: float
    ):
        """Push gradient parameter to ps.
        """
        var_list_tensor = tf.convert_to_tensor(var_list, name="push_grad_param_var_list")
        eta_tensor = tf.convert_to_tensor(eta, name="push_grad_param_eta")

        with tf.device('/cpu:0'):
            return _trainer_ops_so.push_grad(
                batch_id,
                var_list_tensor,
                gradient,
                eta_tensor,
                self._conf_file,
                self._trainer_id
            )

    def push_variable(self, var_name, var_type, var, var_opt):
        """Push variable to ps.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.push_variable(
                var,
                var_opt,
                var_name,
                var_type,
                self._conf_file,
                self._trainer_id
            )

    def pull_variable(self, var_name, var_type, var, var_opt):
        """Pull variable parameters from ps to trainer.
        """
        if var_type == 1:
            return _trainer_ops_so.pull_variable(
                var,
                var_opt,
                var_name,
                var_type,
                self._conf_file,
                self._trainer_id
            )
        elif var_type == 2:
            with tf.device('/cpu:0'):
                return _trainer_ops_so.pull_variable(
                    var,
                    var_opt,
                    var_name,
                    var_type,
                    self._conf_file,
                    self._trainer_id
                )

    def save(self, version, var_name, ckp_type, ckp_target, nfs_path, queue):
        """Save to hdfs or local.
        """
        with tf.device('/cpu:0'):
            version_tensor = tf.convert_to_tensor(version, name="ckp_version")
            return _trainer_ops_so.sniper_save(
                version_tensor,
                var_name,
                ckp_type,
                ckp_target,
                nfs_path,
                queue,
                self._conf_file,
                self._trainer_id
            )

    def check_ckp(self, version, ckp_type, ckp_target):
        with tf.device('/cpu:0'):
            version_tensor = tf.convert_to_tensor(version, name="ckp_version")
            return _trainer_ops_so.check_ckp(
                version_tensor,
                ckp_type,
                ckp_target,
                self._conf_file,
                self._trainer_id
            )

    def restore(self, varname, shard_idx, shard_num, shard_nfs_weight_paths,
                shard_nfs_adagrad_paths):
        """Restore parameters from hdfs to ps.

        Blocking. Need to wait restore to finish.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.sniper_restore(
                varname,
                shard_idx,
                shard_num,
                shard_nfs_weight_paths,
                shard_nfs_adagrad_paths,
                self._conf_file,
                self._trainer_id
            )

    def freeze(self, model_name, dense_vars, dense_var_queues, sparse_vars,
               sparse_var_queues):
        logger.info('sparse_vars: %s', str(sparse_vars))
        with tf.device('/cpu:0'):
            return _trainer_ops_so.freeze(model_name, dense_vars,
                                         dense_var_queues, sparse_vars,
                                         sparse_var_queues, self._conf_file,
                                         self._trainer_id)

    def count_feature(self):
        """Count feature.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.count_feature(self._conf_file)

    def save_feature_count(self, varname, nfs_path):
        """Save feature count result to hdfs.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.save_feature_count(varname, nfs_path,
                                                     self._conf_file,
                                                     self._trainer_id)

    def restore_feature_count(self, varname, paths):
        """Restore feature count result to ps.
        """
        with tf.device('/cpu:0'):
            return _trainer_ops_so.restore_feature_count(varname, paths,
                                                        self._conf_file,
                                                        self._trainer_id)


def TrainerOps(sniper_conf):
    return TrainerOpsBase(sniper_conf)
