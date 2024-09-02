# -*- coding: utf-8 -*-
#
# Distributed training based on horovod.

import os
import sys
import json
import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow.mpi_ops import Average, Sum, Adasum

def init():
    return hvd.init()


def size():
    return hvd.size()


def local_size():
    return hvd.local_size()


def rank():
    return hvd.rank()


def local_rank():
    return hvd.local_rank()


def DistributedOptimizer(opt):
    return hvd.DistributedOptimizer(opt)


def InitHooks():
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    return hooks


def get_all_network_vars():
    """All network vars need to average.
    """
    var_list = [
        v for v in tf.global_variables() if v.name.find('embedding') < 0 and
        v.name.find('global_step') < 0 and v.name.find('out_of_range_ind') < 0
    ]
    return var_list


def make_average_variables_group_fn(
    name: str,
    device_dense: str = '',
    device_sparse: str = '',
    compression = hvd.Compression.none,
    op = Average
):
    def allreduce_vars_group(vars):
        with tf.name_scope(name + "_Allreduce"):
            return tf.group(*[
                var.assign(
                    hvd.allreduce(
                        var,
                        device_dense=device_dense,
                        device_sparse=device_sparse,
                        compression=compression,
                        op=op)
                ) if var is not None else var
                for var in vars
            ])

    return allreduce_vars_group


def allreduce_variables(
    variables: list,
    name: str,
    device_dense: str = '',
    device_sparse: str = '',
    compression = hvd.Compression.none,
    op = Average
):
    average_group = make_average_variables_group_fn(
        name,
        device_dense,
        device_sparse,
        compression,
        op
    )

    return average_group(variables)


def average_network_variables(
    name: str,
    device_dense: str = '',
    device_sparse: str = '',
    compression = hvd.Compression.none,
    op = Average
):
    var_list = get_all_network_vars()
    print(
        "########################### average_network_variables varlist: ",
        var_list
    )

    return allreduce_variables(
        var_list,
        name,
        device_dense,
        device_sparse,
        compression,
        op
    )


class PeriodAvgNetworkVaiablesHook(tf.train.SessionRunHook):
    """SessionRunHook that will average network variables at fixed steps.
    """
    def __init__(self, frequency):
        super(PeriodAvgNetworkVaiablesHook, self).__init__()
        self.freq = frequency
        self.avg_op = None

    def begin(self):
        self._step = 0
        if not self.avg_op:
            with tf.device('/cpu:0'):
                self.avg_op = average_network_variables('average_trainable_network_variables')

    def before_run(self, run_context):
        self._step += 1
        if self.freq > 0 and self._step % self.freq == 0:
            if self.avg_op is not None:
                return tf.train.SessionRunArgs([self.avg_op])
