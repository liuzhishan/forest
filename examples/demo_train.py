import tensorflow as tf
import threading
import fire
import os
import sys
import math
import json
import pyarrow
from datetime import timedelta
from datetime import datetime
from tensorflow.python.ops import variables

sys.path.insert(0, '.')

from trainer.model_base import ModelBase
from trainer.trainer import Trainer
from trainer.AUC import auc as auc_eval
from trainer import hooks as train_hook
from trainer.trainer_ops_wrapper import TrainerOps
from trainer.sniper_config import SniperConf, convert_config
from trainer import dist
from trainer import util
from trainer.util import logger, PsBalance, get_lines


class HashDnnModel(ModelBase):
    def __init__(self, sniper_conf, trainer_ops):
        self._config = sniper_conf
        self._trainer_ops = trainer_ops
        ModelBase.__init__(self, self._config, self._trainer_ops)

        self._dnn_net_size = [512, 256, 2]

        self._batch_size = self._config.batch_size
        self._print_interval = self._config.print_interval

        self._input_size = self._config.input_sparse_total_size + self._config.input_dense_total_size

    def inference(
        self,
        dnn_input,
        labels,
        is_cpu_ps: bool,
        is_train: bool,
        hooks,
        debug_info
    ):
        input_size = self._input_size
        layer = dnn_input

        for i in range(len(self._dnn_net_size)):
            with tf.variable_scope("upper_layer_{}".format(i),
                                   reuse=tf.AUTO_REUSE):
                layer_size = self._dnn_net_size[i]

                weight_name = 'w'
                w = tf.get_variable(
                    weight_name, [input_size, layer_size],
                    initializer=tf.random_normal_initializer(
                        stddev=1.0 / math.sqrt(float(input_size))),
                    trainable=True)
                bias_name = 'b'
                b = tf.get_variable(bias_name, [layer_size],
                                    initializer=tf.zeros_initializer,
                                    trainable=True)

                logger.info("%s length=%d * %d" %
                            (w.name, input_size, layer_size))
                logger.info("%s length=%d" % (b.name, layer_size))

                if i != len(self._dnn_net_size) - 1:
                    layer = tf.nn.relu(tf.add(tf.matmul(layer, w), b))
                else:
                    layer = tf.add(tf.matmul(layer, w), b)

                input_size = layer_size

        prob = tf.nn.softmax(layer, name="softmax")
        labels = tf.transpose(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=layer, name='xentropy')
        train_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        auc, prob_mean, real_mean = None, None, None

        if hooks is not None:
            if is_train:
                _, auc, reset_auc_val = auc_eval(labels,
                                                 tf.clip_by_value(
                                                     prob[:, 1],
                                                     tf.constant(0.0),
                                                     tf.constant(1.0)),
                                                 name="train_auc")
                prob_mean = tf.reduce_mean(prob[:, 1], name="prob_mean")
                real_mean = tf.reduce_mean(tf.cast(labels, tf.float32),
                                           name="real_mean")
                hooks.append(
                    train_hook.LoggerHookV3(
                        auc,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))
                hooks.append(
                    train_hook.LoggerHookV2(
                        train_loss,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))
                hooks.append(
                    train_hook.LoggerHookV2(
                        prob_mean,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))
                hooks.append(
                    train_hook.LoggerHookV2(
                        real_mean,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))

                hooks.append(
                    train_hook.SaveModelHook(
                        self, auc, train_loss, prob_mean, real_mean,
                        self._config.ckp_save_nfs_path, self._config.var_2_btq))

            else:
                prob_mean = tf.reduce_mean(prob[:, 1], name="prob_mean")
                real_mean = tf.reduce_mean(tf.cast(labels, tf.float32),
                                           name="real_mean")
                _, auc, reset_auc_val = auc_eval(labels,
                                                 tf.clip_by_value(
                                                     prob[:, 1],
                                                     tf.constant(0.0),
                                                     tf.constant(1.0)),
                                                 name="val_auc")
                hooks.append(
                    train_hook.LoggerHookV2(
                        auc,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval,
                        acc=True))
                hooks.append(
                    train_hook.LoggerHookV2(
                        train_loss,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))
                hooks.append(
                    train_hook.LoggerHookV2(
                        prob_mean,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))
                hooks.append(
                    train_hook.LoggerHookV2(
                        real_mean,
                        batch_size=self._batch_size,
                        print_interval=self._print_interval))

        return train_loss


    def get_learning_rate(self, total_train_step, cur_step, base_lr=0.05):
        if total_train_step == 0:
            return base_lr

        if cur_step < total_train_step:
            lr = base_lr * (1.1 -
                            math.exp(-cur_step * 2.33 / total_train_step))
            if lr > base_lr:
                lr = base_lr
            return lr

        return base_lr


def get_1k_path_list(begin_date_str, end_date_str):
    begin_date = datetime.strptime(begin_date_str, "%Y%m%d %H")
    end_date = datetime.strptime(end_date_str, "%Y%m%d %H")
    delta = end_date - begin_date
    path_list = []

    for i in range(int(delta.days * 24 + delta.seconds / 3600) + 1):
        date = begin_date + timedelta(seconds=3600 * i)
        path = date.strftime(
            'hdfs://default/home/ad/liuzhishan/trainer/unittest/offline_train/train_path/p_date=%Y%m%d/p_hourmin=%H00/batch_size=1024/feature_file=dsp_conv_sxb_nebula_appid_large'
        )
        files = tf.io.gfile.listdir(path)
        for f in files:
            path_list.append(path + "/" + f)

    return path_list


def get_hdfs_files(dirnames: list):
    res = []

    fs = pyarrow.HadoopFileSystem()
    for dirname in dirnames:
        for x in fs.ls(dirname):
            if x.endswith("_SUCCESS"):
                continue

            res.append(x)

    return res


def get_local_train_files():
    dirnames = ["viewfs:///home/model_strategy_e/users/liuzhishan/simple_features_conv/dsp_feature_nebula_new_v1/20240917/0000",
                # "viewfs:///home/model_strategy_e/users/liuzhishan/simple_features_conv/dsp_feature_nebula_new_v1/20240917/0100",
                # "viewfs:///home/model_strategy_e/users/liuzhishan/simple_features_conv/dsp_feature_nebula_new_v1/20240917/0200"
                ]

    return get_hdfs_files(dirnames)


def get_local_test_files():
    dirnames = ["viewfs:///home/model_strategy_e/users/liuzhishan/simple_features_conv/dsp_feature_nebula_new_v1/20240917/0300"]

    return get_hdfs_files(dirnames)


def train_and_validate(conf_file):
    # --------------- step1 ---------------
    # distributed training init
    dist.init()
    util.set_trainer_cpu_affinity()

    new_conf_file = convert_config(conf_file)
    # --------------- step2 ---------------
    # load && check conf
    sniper_conf = SniperConf(new_conf_file)

    # build model && trainer
    trainer_ops = TrainerOps(sniper_conf)
    model = HashDnnModel(sniper_conf, trainer_ops)
    trainer = Trainer(sniper_conf, trainer_ops, model)

    # --------------- step3 ---------------
    # train
    train_path = get_local_train_files()
    trainer.train(train_path)

    # --------------- step4 ---------------
    # validate
    if dist.rank() == 0:
        valid_path = get_local_test_files()
        trainer.evaluate(valid_path)

    # --------------- step5 ---------------
    # export
    # model.export()


def test_batch(filename):
    ps_balance = PsBalance()
    for line in get_lines(filename):
        logger.info('len(line): %d', len(line))
        load_dict = ps_balance.process(line)


if __name__ == "__main__":
    fire.Fire()
