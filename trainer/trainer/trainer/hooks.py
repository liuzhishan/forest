import time
import json
import codecs
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from datetime import datetime
import os

from infra.perflog import create_perf_context
from kconf.client import KConf
from kconf.exception import KConfError
from infra.perflog import create_perf_context

from .util import logger
from . import util

class TraceHook(tf.train.SessionRunHook):
    """Hook to perform Traces every N steps."""

    def __init__(self, every_step=50, trace_level=tf.RunOptions.FULL_TRACE):
        self._trace = every_step == 1
        self.trace_level = trace_level
        self.every_step = every_step

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use _TraceHook.")

    def before_run(self, run_context):
        if self._trace:
            options = tf.RunOptions(trace_level=self.trace_level)
        else:
            options = None
        return tf.train.SessionRunArgs(fetches=self._global_step_tensor,
                                       options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results - 1
        if self._trace:
            self._trace = False
            if run_values.run_metadata is None:
                raise RuntimeError(
                    "Run metadata should not be None when trace is on")
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_values.run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()

            with open('timeline' + str(global_step) + '.json', 'w') as f:
                f.write(ctf)
        if not (global_step + 1) % self.every_step:
            self._trace = True


class DatasetInitializerHook(tf.train.SessionRunHook):
    """Creates a SessionRunHook that initializes the passed iterator."""

    def __init__(self, iterator):
        self._iterator = iterator

    def begin(self):
        self._initializer = self._iterator.initializer

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)


class DSHandleHook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
        self.train_str = train_str
        self.valid_str = valid_str
        self.train_handle = None
        self.valid_handle = None

    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run(
                [self.train_str, self.valid_str])


class LoggerHook(tf.train.SessionRunHook):
    def __init__(self, print_interval, batch_size, train_auc, valid_auc):
        self.train_auc = train_auc
        self.valid_auc = valid_auc
        self.mode = "diabled"
        self.print_interval = print_interval
        self.batch_size = batch_size

    def set_mode(self, mode):
        self.mode = mode
        self._step = 0

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        if self.mode == "train":
            self._step += 1
            return tf.train.SessionRunArgs([self.train_auc])
        elif self.mode == "valid":
            self._step += 1
            return tf.train.SessionRunArgs([self.valid_auc])
        else:
            return

    def after_run(self, run_context, run_values):
        if self.mode not in ["train", "valid"]:
            logger.info("not in train / valid mode, ignore logger hook")
            return
        print_interval = self.print_interval
        results = run_values.results
        if self._step % print_interval == 0:
            # if True:
            tf_auc = results[0]
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            examples_per_sec = print_interval * self.batch_size / duration
            format_str = ('%s: step %d, tf_auc = %.4f (%.1f examples/sec)')
            logger.info(
                format_str %
                (datetime.now(), self._step, tf_auc[0], examples_per_sec))


class LoggerHookTrainAndValid(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self,
                 print_tensor,
                 batch_size,
                 print_interval=10,
                 print_op=None,
                 acc=False):
        self._print_tensor = print_tensor
        self._print_op = print_op
        self._print_interval = print_interval
        self._batch_size = batch_size
        self._acc_value_train = 0
        self._acc_value_valid = 0
        self._acc = acc

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._print_op is None:
            return tf.train.SessionRunArgs([self._print_tensor])
        else:
            return tf.train.SessionRunArgs(
                [self._print_tensor, self._print_op])

    def after_run(self, run_context, run_values):
        self._acc_value_valid += run_values.results[0]
        if self._print_op is None:
            train_res = run_context.session.run([self._print_tensor])
        else:
            train_res = run_context.session.run(
                [self._print_tensor, self._print_op])
        self._acc_value_train += train_res[0]
        if self._step % self._print_interval == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            valid_value = run_values.results[0]
            if self._acc == True:
                valid_value = self._acc_value_valid / self._step
            examples_per_sec = self._print_interval * self._batch_size * 1.0 / duration
            iteration_per_sec = self._print_interval * 1.0 / duration

            format_str = (
                '%s: step %d, %s_%s = %.8f (%.1f it/sec; %.1f examples/sec)')
            logger.info(
                format_str %
                (datetime.now(), self._step, "valid", self._print_tensor.name,
                 valid_value, iteration_per_sec, examples_per_sec))
            # train log
            train_value = train_res[0]
            if self._acc == True:
                train_value = self._acc_value_train / self._step
            logger.info(
                format_str %
                (datetime.now(), self._step, "train", self._print_tensor.name,
                 train_value, iteration_per_sec, examples_per_sec))


class LoggerHookV2(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self,
                 print_tensor,
                 batch_size,
                 print_interval=10,
                 print_op=None,
                 acc=False):
        self._print_tensor = print_tensor
        self._print_op = print_op
        self._print_interval = int(print_interval)
        self._batch_size = int(batch_size)
        self._acc_value = 0
        self._acc = bool(acc)

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._print_op is None:
            return tf.train.SessionRunArgs([self._print_tensor])
        else:
            return tf.train.SessionRunArgs(
                [self._print_tensor, self._print_op])

    def after_run(self, run_context, run_values):
        self._acc_value += run_values.results[0]
        if self._step % self._print_interval == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            loss_value = run_values.results[0]
            if self._acc == True:
                loss_value = self._acc_value / self._step
            examples_per_sec = self._print_interval * self._batch_size * 1.0 / duration
            iteration_per_sec = self._print_interval * 1.0 / duration

            format_str = (
                '%s: step %d, %s = %.8f (%.1f it/sec; %.1f examples/sec)')
            logger.info(format_str %
                        (datetime.now(), self._step, self._print_tensor.name,
                         loss_value, iteration_per_sec, examples_per_sec))


class LoggerHookV3(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, print_tensor, batch_size, print_interval=10):
        self._print_tensor = print_tensor
        self._print_interval = print_interval
        self._batch_size = batch_size

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._print_interval == 0:
            return tf.train.SessionRunArgs([self._print_tensor])

    def after_run(self, run_context, run_values):
        if self._step % self._print_interval == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            value = run_values.results[0]
            examples_per_sec = self._print_interval * self._batch_size * 1.0 / duration
            iteration_per_sec = self._print_interval * 1.0 / duration

            format_str = (
                '%s: step %d, auc = %.4f (%.1f it/sec; %.1f examples/sec)')
            logger.info(format_str % (datetime.now(), self._step, value,
                                      iteration_per_sec, examples_per_sec))


class VarHook(tf.train.SessionRunHook):
    """"print variables"""

    def __init__(self, print_tensor, batch_size, print_interval=10, limit=100):
        self._print_tensor = print_tensor
        self._print_interval = print_interval
        self._batch_size = batch_size
        self._limit = limit

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._print_interval == 0:
            return tf.train.SessionRunArgs([self._print_tensor])

    def after_run(self, run_context, run_values):
        if self._step % self._print_interval == 0:
            value = run_values.results[0]
            logger.info(
                "{date_now}: step {g_step}, var_shape={shape}, {var_name} = {var}"
                .format(date_now=datetime.now(),
                        g_step=self._step,
                        var_name=self._print_tensor.name,
                        shape=np.array(value).shape,
                        var=value.tolist()[:self._limit]))

class PsPushEmbeddingHook(tf.train.SessionRunHook):
    def __init__(self, ps_push_op):
        self._ps_push_op = ps_push_op

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self._ps_push_op)


class PreOpsHook(tf.train.SessionRunHook):
    def __init__(self, pre_ops, pre_ops_feed_dict=None):
        self._pre_ops = pre_ops
        self._pre_ops_feed_dict = pre_ops_feed_dict

    def after_create_session(self, session, coord):
        session.run(self._pre_ops, feed_dict=self._pre_ops_feed_dict)


class PrefillHook(tf.train.SessionRunHook):
    def __init__(self, prefill_op, count=1, name="unknown"):
        self._prefill_op = prefill_op
        self._name = name
        self._count = count

    def after_create_session(self, session, coord):
        for i in range(self._count):
            session.run(self._prefill_op)


class IntervalOpHook(tf.train.SessionRunHook):
    """do op after every interval"""

    def __init__(self, do_op, do_interval):
        self._do_op = do_op
        self._do_interval = do_interval

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._do_interval == 0:
            return tf.train.SessionRunArgs([self._do_op])


class ClickhouseHook(tf.train.SessionRunHook):
    """Logs loss and runtime.

    注意: perf 对性能有影响。离线训练可自行决定是否使用。在线训练必须加上。
    """

    def __init__(self,
                 aim,
                 model_name,
                 auc_tensor,
                 loss_tensor,
                 prob_mean_tensor,
                 real_mean_tensor,
                 is_online,
                 acc=False,
                 print_interval=100,
                 batch_size=1024,
                 last_result={}):
        self._aim = aim
        self._model_name = model_name
        self._auc_tensor = auc_tensor
        self._loss_tensor = loss_tensor
        self._prob_tensor = prob_mean_tensor
        self._real_tensor = real_mean_tensor
        self._acc = acc
        self._is_online = is_online
        self._print_interval = print_interval
        self._batch_size = batch_size
        self._startt_time = time.time()
        self._last_result = last_result

        self._auc_value = 0
        self._loss_value = 0
        self._prob_value = 0
        self._real_value = 0

        self._acc_loss = 0
        self._acc_prob = 0
        self._acc_real = 0

        self._step = 0
        extra = 'online' if is_online else 'offline'
        self._perf_auc = create_perf_context('klearn.train',
                                             aim,
                                             model_name,
                                             'auc',
                                             extra,
                                             biz_def='ad')
        self._perf_loss = create_perf_context('klearn.train',
                                              aim,
                                              model_name,
                                              'loss',
                                              extra,
                                              biz_def='ad')
        self._perf_prob = create_perf_context('klearn.train',
                                              aim,
                                              model_name,
                                              'prob',
                                              extra,
                                              biz_def='ad')
        self._perf_real = create_perf_context('klearn.train',
                                              aim,
                                              model_name,
                                              'real',
                                              extra,
                                              biz_def='ad')

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([
            self._auc_tensor, self._loss_tensor, self._prob_tensor,
            self._real_tensor
        ])

    def after_run(self, run_context, run_values):
        self._acc_loss += run_values.results[1]
        self._acc_prob += run_values.results[2]
        self._acc_real += run_values.results[3]

        if self._acc == True:
            self._loss_value = self._acc_loss / self._step
            self._prob_value = self._acc_prob / self._step
            self._real_value = self._acc_real / self._step
        else:
            self._loss_value = run_values.results[1]
            self._prob_value = run_values.results[2]
            self._real_value = run_values.results[3]

        self._auc_value = run_values.results[0]

        self._last_result['auc'] = self._auc_value
        self._last_result['loss'] = self._loss_value

        if self._step % self._print_interval == 0:
            try:
                self._perf_auc.logstash(micros=int(self._auc_value * 1000000))
                self._perf_loss.logstash(micros=int(self._loss_value *
                                                    1000000))
                self._perf_prob.logstash(micros=int(self._prob_value *
                                                    1000000))
                self._perf_real.logstash(micros=int(self._real_value *
                                                    1000000))
            except Exception as e:
                logger.info(e)

        if self._step % self._print_interval == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            examples_per_sec = self._print_interval * self._batch_size * 1.0 / duration
            iteration_per_sec = self._print_interval * 1.0 / duration

            logger.info(
                'step %d, auc = %.4f, loss = %.10f, prob_mean: %.6f, real_mean: %.6f, (%.1f it/sec; %.1f examples/sec)',
                self._step, self._auc_value, self._loss_value,
                self._prob_value, self._real_value, iteration_per_sec,
                examples_per_sec)


class SaveModelHook(tf.train.SessionRunHook):
    def __init__(self,
                 model,
                 auc_tensor,
                 loss_tensor,
                 prob_mean_tensor,
                 real_mean_tensor,
                 ckp_nfs_path='',
                 ckp_var_2_btq={},
                 wait_ps=True):
        self._model = model
        self._auc_tensor = auc_tensor
        self._loss_tensor = loss_tensor
        self._prob_mean_tensor = prob_mean_tensor
        self._real_mean_tensor = real_mean_tensor
        self._wait_ps = wait_ps

        self._ckp_nfs_path = ckp_nfs_path
        self._ckp_var_2_btq = ckp_var_2_btq
        self._use_btq = self._model._config.use_btq
        self._ckp_save_btq_incr_step = self._model._config.ckp_save_btq_incr_step
        self._ckp_save_btq_full_interval = self._model._config.ckp_save_btq_full_interval
        self._ckp_save_nfs_full_interval = self._model._config.ckp_save_nfs_full_interval
        self._push_dense_ops = None
        logger.info(
            "ckp config: ckp_nfs_path(%s), use_btq(%d), "
            "ckp_save_btq_incr_step(%d), ckp_save_btq_full_interval(%d), "
            "ckp_save_nfs_full_interval(%d), wait_ps(%d)", self._ckp_nfs_path,
            self._use_btq, self._ckp_save_btq_incr_step,
            self._ckp_save_btq_full_interval, self._ckp_save_nfs_full_interval, self._wait_ps)

    def set_push_dense_ops(self, push_dense_ops):
        self._push_dense_ops = push_dense_ops

    def begin(self):
        self._step = 0
        self._auc = 0.0
        self._train_loss = 0.0
        self._prob_mean = 0.0
        self._real_mean = 0.0
        self._last_full_nfs_save_ts = int(time.time())
        self._last_full_btq_save_ts = int(time.time())
        self._last_incr_btq_save_ts = int(time.time())
        self._need_save_nfs_full = False
        self._need_save_btq_full = False
        self._need_save_btq_incr = False
        self._run_args = None
        self._now = None
        self._version_ts = None

    def before_run(self, run_context):
        self._step += 1
        self._now = datetime.now()
        self._version_ts = int(self._now.timestamp())

        self._need_save_nfs_full = (self._ckp_save_nfs_full_interval != 0) and (
            (self._version_ts - self._last_full_nfs_save_ts) >
            self._ckp_save_nfs_full_interval) and (
                self._model.is_meet_save_condition(
                    self._auc, self._train_loss, self._prob_mean,
                    self._real_mean, self._step))

        self._need_save_btq_full = self._use_btq \
            and self._model.is_meet_save_condition(self._auc, self._train_loss, self._prob_mean, self._real_mean, self._step) \
            and ((self._ckp_save_btq_full_interval != 0 \
                     and (self._version_ts - self._last_full_btq_save_ts) > self._ckp_save_btq_full_interval) \
                 or (self._ckp_save_btq_incr_step > 0 \
                     and self._step == self._ckp_save_btq_incr_step + 1))  # 训练开始一段时间后默认发送 btq全量数据

        self._need_save_btq_incr = self._use_btq and (self._ckp_save_btq_incr_step != 0) and (
            self._step % self._ckp_save_btq_incr_step
            == 0) and (self._model.is_meet_save_condition(
                self._auc, self._train_loss, self._prob_mean, self._real_mean,
                self._step))

        # 发起save流程之前，请将最新的dense var push到ps上
        self._run_args = [
            self._auc_tensor, self._loss_tensor, self._prob_mean_tensor,
            self._real_mean_tensor]

        if self._need_save_nfs_full or self._need_save_btq_full or self._need_save_btq_incr:
            if self._push_dense_ops != None:
                self._run_args.append(self._push_dense_ops)

        return tf.train.SessionRunArgs(self._run_args)

    def after_run(self, run_context, run_values):
        # full nfs save
        if self._need_save_nfs_full:
            self._model.save(self._now, 2, 1, self._wait_ps)
            self._last_full_nfs_save_ts = self._version_ts
            logger.info("finish full nfs save with version = %d, wait_ps: %s", self._version_ts, self._wait_ps)

        # full btq save
        if self._need_save_btq_full:
            self._model.save(self._now, 2, 2, False)
            self._last_full_btq_save_ts = self._version_ts
            logger.info("finish full btq save with version[%d]" % self._version_ts)

        # incr btq save
        if self._need_save_btq_incr:
            self._model.save(self._now, 1, 2, False)
            self._last_incr_btq_save_ts = self._version_ts
            logger.info("finish incr btq save with version[%d]" % self._version_ts)

        if len(self._run_args) == 4:
            self._auc, self._train_loss, self._prob_mean, self._real_mean = run_values.results
        elif len(self._run_args) == 5:
            push_res, self._auc, self._train_loss, self._prob_mean, self._real_mean = run_values.results

class DebugInfoHook(tf.train.SessionRunHook):
    """"print variables"""
    def __init__(self, debug_info_tensor, var_tensor, print_interval=10, limit=100):
        self._debug_info_tensor = debug_info_tensor
        self._var_tensor = var_tensor
        self._print_interval = print_interval
        self._limit = limit

    def begin(self):
        self._step = 0
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._step % self._print_interval == 0:
            return tf.train.SessionRunArgs([self._debug_info_tensor, self._var_tensor])

    def after_run(self, run_context, run_values):
        if self._step % self._print_interval == 0:
            debug_info, value = run_values.results
            batch_size = len(debug_info)
            for i in range(batch_size):
                if type(value[i]) == np.ndarray:
                    v = value[i].tolist()
                    logger.info('i: %d, key: %s, value: %s, last: %s, sum: %f',
                                i, str(debug_info[i]), v[:self._limit],
                                v[-self._limit:], sum(v))
                else:
                    logger.info('i: %d, key: %s, value: %s', i, str(debug_info[i]), value[i])


class TensorDebugHook(tf.train.SessionRunHook):
    """"print variables"""
    def __init__(self, config, dnn_input, debug_info, result_tensor):
        self.result = []

        self.var_tensor = []
        self._config = config
        if self._config.split_input:
            self.var_tensor.extend(dnn_input)
        else:
            self.var_tensor.append(dnn_input)

        self.var_tensor.append(debug_info)
        self.var_tensor.append(result_tensor)

        self.sparse_names = ["embedding_%d" % i for i in range(len(self._config.sparse_input))]
        self.dense_names = ["dense_field_%s" % i for i in range(len(self._config.dense_input))]

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.var_tensor)

    def after_run(self, run_context, run_values):
        res = run_values.results
        batch_size = len(res[0])

        sparse_inputs = []
        dense_inputs = []
        if self._config.split_input:
            user_embedding = res[0][i]
            item_embedding = res[1][i]
            user_dense = res[2][i]
            item_dense = res[3][i]
            res = res[4:]
            embedding_sizes = self._config.input_sparse_emb_size[: self._config.input_sparse_user_count]
            sparse_inputs.extend(np.split(user_embedding, embedding_sizes, axis=1))

            embedding_sizes = self._config.input_sparse_emb_size[self._config.input_sparse_user_count:]
            sparse_inputs.extend(np.split(item_embedding, embedding_sizes, axis=1))

            dense_sizes = self._config.dense_fields[: self._config.input_dense_user_count]
            dense_inputs.extend(np.split(user_dense, dense_sizes, axis=1))

            dense_sizes = self._config.dense_fields[self._config.input_dense_user_count:]
            dense_inputs.extend(np.split(item_dense, dense_sizes, axis=1))
        else:
            indices = [0]
            self._config.input_sparse_emb_size + self._config.dense_fields
            for size in self._config.input_sparse_emb_size + self._config.dense_fields:
                indices.append(indices[-1] + size)

            inputs = np.split(res[0], indices[1:], axis=1)
            res = res[1:]
            sparse_inputs = inputs[: len(self.sparse_names)]
            dense_inputs = inputs[len(self.sparse_names):]

        debug_info = res[0]
        res = res[1:]
        for i in range(batch_size):
            item = {}
            item["debug_info"] = debug_info[i].decode('UTF-8')
            for k, name in enumerate(self.sparse_names):
                v = sparse_inputs[k][i]
                if type(v) == np.ndarray:
                    v = v.tolist()
                else:
                    v = v.item()
                item[name] = v

            for k, name in enumerate(self.dense_names):
                v = dense_inputs[k][i]
                if type(v) == np.ndarray:
                    v = v.tolist()
                else:
                    v = v.item()
                item[name] = v

            v = res[-1]
            if type(v) == np.ndarray:
                v = v.tolist()
            else:
                v = v.item()
            item["predict_result"] = v

            self.result.append(item)

class DebugBsHook(tf.train.SessionRunHook):
    """"print variables"""
    def __init__(self, config, dnn_input, result_tensor):
        self.result = []

        self.var_tensor = []
        self._config = config
        self.var_tensor.append(dnn_input)
        self.var_tensor.append(result_tensor)

        self.sparse_names = ["embedding_%d" % i for i in range(len(self._config.sparse_input))]
        self.dense_names = ["dense_field_%s" % i for i in range(len(self._config.dense_input))]
        self.step = 0

    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.var_tensor)

    def after_run(self, run_context, run_values):
        res = run_values.results
        batch_size = len(res[0])

        logger.info('step: %d, batch_size: %d, res: %s', self.step, batch_size, str(res))
        logger.info('sparse_names: %s', ','.join(self.sparse_names))

        sparse_inputs = []
        dense_inputs = []
        indices = [0]
        total_size = self._config.input_sparse_emb_size + self._config.dense_fields
        logger.info('total_size: %s', str(total_size))
        for size in total_size:
            indices.append(indices[-1] + size)

        logger.info('indices: %s', json.dumps(indices))

        inputs = np.split(res[0], indices[1:], axis=1)
        sparse_inputs = inputs[: len(self.sparse_names)]
        dense_inputs = inputs[len(self.sparse_names):]

        for i in range(batch_size):
            item = {}
            for k, name in enumerate(self.sparse_names):
                v = sparse_inputs[k][i]
                if type(v) == np.ndarray:
                    v = v.tolist()
                else:
                    v = v.item()
                item[name] = v

            for k, name in enumerate(self.dense_names):
                v = dense_inputs[k][i]
                if type(v) == np.ndarray:
                    v = v.tolist()
                else:
                    v = v.item()
                item[name] = v

            item["predict_result"] = float(res[1][i])

            logger.info('i: %s, value: %s', i, json.dumps(item))

class LoggerSklearnAUC(tf.train.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self,labels, probs,weight):
        self.labels = labels
        self.probs = probs
        self.weight = weight
        self._step = 0
        self.probs_acc = []
        self.labels_acc = []
        self.weight_acc = []

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs([self.labels, self.probs, self.weight])

    def after_run(self, run_context, run_values):
        self.weight_acc.extend(run_values.results[2])
        self.probs_acc.extend(run_values.results[1])
        self.labels_acc.extend(run_values.results[0])

    def end(self,sess):
        from sklearn.metrics import roc_curve, auc
        logger.info("session end")
        fpr,tpr,threshold = roc_curve(self.labels_acc, self.probs_acc, pos_label=1.0, sample_weight=self.weight_acc)
        roc_auc = auc(fpr,tpr)
        logger.info("step: %s, sklearn auc: %s",self._step,roc_auc)

