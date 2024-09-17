from __future__ import print_function
import os
import logging
import time
import codecs
import psutil
import math
import json
import types
import pyarrow
import tensorflow as tf
import horovod.tensorflow as hvd
from io import StringIO
from datetime import datetime, timedelta
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder
from horovod.tensorflow.mpi_ops import Average

from .trainer_ops_wrapper import TrainerOps
from .sniper_config import SniperConf

from .dist import InitHooks
from .dist import rank
from .dist import PeriodAvgNetworkVaiablesHook

from .util import logger, run_command, get_root_dir
from .util import get_tf_session_config

from .hooks import SaveModelHook, ClickhouseHook, DatasetInitializerHook


class Trainer(object):

    def __init__(self, sniper_conf, trainer_ops, model):
        self._config = sniper_conf
        self._trainer_ops = trainer_ops
        self._model = model

        self._sess_config = get_tf_session_config(
            self._config.per_process_gpu_memory_fraction,
            tf_intra_op_parallelism_threads = 64,
            tf_inter_op_parallelism_threads = 64,
            use_xla = self._config.use_xla
        )

        if self._config['is_local']:
            self.start_local_ps_hub()

        self._src_to_number = {
            'SRC_INVALID': 0,
            'SRC_HDFS': 1,
            'SRC_KAFKA': 2,
            'SRC_AIPLATFORM_COMMON_SAMPLE': 3,
            'SRC_BTQ': 4,
            'SRC_DRAGON': 5
        }

        self._work_mode_to_number = {
            'MODE_INVALID': 0,
            'MODE_TRAIN': 1,
            'MODE_VALID': 2,
            'MODE_COUNT': 3
        }

        self._model.build()

        logger.info('trainer build done')

    def start_local_ps_hub(self):
        """Start local ps and hub.

        If ps and hub are already running, kill first, then start.
        """
        for p in psutil.process_iter():
            if p.name().find('ps_') >= 0 or p.name().find('hub_') >= 0:
                p.kill()

        pwd = os.getcwd()
        os.chdir(get_root_dir())
        logger.info("pwd: %s, root: %s", pwd, get_root_dir())

        run_command(["./ps_server", "> ps.log 2>&1", "&"])
        run_command(["./hub_server", "> hub.log 2>&1", "&"])

        os.chdir(pwd)

    def start_sample(
        self,
        parallel: int,
        src: int,
        file_list: [str],
        work_mode: int = 1
    ):
        """Start sample.

        Args:
            src: 1 for hdfs, 2 for kafka.
            file_list: Filenames for offline training.
            work_mode: 1 for train

        Returns:
            None.
        """
        if self._config.use_auto_shard and work_mode != 1:
            raise Exception("auto_shard can by used only when training!")

        if rank() == 0:
            logger.info('start sample, work_mode: %d', work_mode)
            start_sample_ret = self._trainer_ops.start_sample(
                parallel, src, file_list, work_mode)
            with tf.Session(config=self._sess_config) as sess:
                sess.run(start_sample_ret)

    def prepare_input(self, work_mode: int):
        """Prepare dnn_input before gpu computing, avoiding split, concat on gpu.
        """
        def compose_data(*argv):
            batch_id = argv[0]
            labels = argv[1]
            dense = argv[2]
            embedding_list = argv[3:3 + len(self._config.sparse_input)]
            embeddings = tf.concat(embedding_list, 1) if len(embedding_list) > 0 else []

            debug_info = argv[-1] if self._config.debug_info != '' else []

            return batch_id, labels, dense, embeddings, debug_info

        ds = self._trainer_ops.prefetch_dataset(work_mode)

        num_parallel_calls = 5 if not self._config.debug_offline else 1
        ds = ds.map(compose_data, num_parallel_calls=num_parallel_calls)
        num_prefetch = 5 if not self._config.debug_offline else 1
        ds = ds.prefetch(num_prefetch)

        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
        iterator = ds.make_initializable_iterator()
        return iterator

    def feature_columns_input(self, input_batch):
        batch_id = input_batch[0]
        labels = input_batch[1]
        dense = input_batch[2]
        embeddings = input_batch[3]
        debug_info = input_batch[4]

        labels = tf.slice(labels, [0, 0], [1, self._config.batch_size])[0]

        if self._config.split_input:
            dense_user_size = sum(
                self._config.dense_fields[:self._config.input_dense_user_count])
            dense_item_size = sum(
                self._config.dense_fields[self._config.input_dense_user_count:])
            user_dense, item_dense = tf.split(
                dense, [dense_user_size, dense_item_size], axis=1)

            user_embedding_size = sum(
                self._config.input_sparse_emb_size[:self._config.
                                                   input_sparse_user_count])
            item_embedding_size = sum(
                self._config.input_sparse_emb_size[self._config.
                                                   input_sparse_user_count:])
            user_embedding, item_embedding = tf.split(
                embeddings, [user_embedding_size, item_embedding_size], axis=1)
            if len(self._config.input_sparse_emb_size) > 0:
                dnn_input = [
                    user_embedding, item_embedding, user_dense, item_dense
                ]
            else:
                dnn_input = [user_dense, item_dense]
        else:
            if len(self._config.input_sparse_emb_size) > 0:
                dnn_input = tf.concat([embeddings, dense], 1)
            else:
                dnn_input = dense
        return batch_id, labels, dnn_input, embeddings, debug_info

    def train_model_fn(self, iterator, hooks, learning_rate):
        global_step = tf.train.get_or_create_global_step()
        batch_id, labels, dnn_input, embeddings, debug_info = self.feature_columns_input(
            iterator.get_next())

        optimizer = 0
        logger.info(self._config)
        if self._config["optimizer"] == "adagrad":
            logger.info("train adagrad")
            optimizer = tf.train.AdagradOptimizer(
                learning_rate * self._config.lr_scale,
                initial_accumulator_value=self._config.eps)
        else:
            assert 0, "use an unknown optimizer %s" % self._config["optimizer"]

        if self._config.mixed_precision:
            optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                optimizer)

        train_loss = self._model.inference(dnn_input, labels, False, True,
                                           hooks, debug_info)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(train_loss, global_step=global_step)

        train_op = tf.group([train_op, update_ops])
        self._update_op = [train_op]
        if len(self._config.input_sparse_emb_size) > 0:
            emb_grad = tf.gradients(train_loss, embeddings)[0]
            push_grad_op = self._trainer_ops.push_grad(batch_id,
                                                      self._config.sparse_input,
                                                      emb_grad, learning_rate)
            self._update_op.append(push_grad_op)

        if rank() == 0:
            push_dense_ops = self._model.push_dense_vars()
            for x in hooks:
                if type(x) == SaveModelHook:
                    x.set_push_dense_ops(push_dense_ops)
            final_push_dense_op = tf.group(push_dense_ops)
            hooks.append(tf.train.FinalOpsHook([final_push_dense_op]))

    def eval_model_fn(self, iterator, hooks):
        global_step = tf.train.get_or_create_global_step()
        batch_id, labels, dnn_input, embeddings, debug_info = self.feature_columns_input(
            iterator.get_next())
        train_loss = self._model.inference(dnn_input, labels, False, False,
                                           hooks, debug_info)
        self._update_op = [train_loss]

    def run_session(self, sess, local_step, feed_dict):
        sess.run(self._update_op, feed_dict=feed_dict)

    def run_kafka_session(self, sess, local_step, feed_dict):
        if local_step % 1000 == 0 and self._config['profile']:
            my_profiler = model_analyzer.Profiler(graph=tf.get_default_graph())
            run_meta = tf.RunMetadata()
            up_op = self._update_op
            sess.run(
                up_op,
                feed_dict=feed_dict,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_meta)
            my_profiler.add_step(step=local_step, run_meta=run_meta)

            profile_op_builder = option_builder.ProfileOptionBuilder()
            profile_op_builder.select(['micros', 'occurrence'])
            profile_op_builder.order_by('micros')
            profile_op_builder.with_max_depth(4)
            my_profiler.profile_graph(profile_op_builder.build())
        else:
            sess.run(self._update_op, feed_dict=feed_dict)

    def train(self, train_path):
        work_mode = self._work_mode_to_number['MODE_TRAIN']
        self.start_sample(self._config['hub_worker'],
                          self._src_to_number['SRC_HDFS'],
                          train_path,
                          work_mode)

        # Must call dist.InitHooks() before training.
        hooks = InitHooks()

        with tf.Graph().as_default():
            learning_rate = tf.placeholder(dtype=tf.float32)

            iter = self.prepare_input(1)
            hooks.append(DatasetInitializerHook(iter))

            self.train_model_fn(iter, hooks, learning_rate)
            hooks.append(
                PeriodAvgNetworkVaiablesHook(
                    self._config.period_avg_network_vaiables_hook))

            t1 = time.time()

            logger.info(
                '---------------------- start training -----------------------'
            )

            ckpt_dir = None
            if rank() == 0:
                ckpt_dir = self._config.local_ckpt_dir
            with tf.train.MonitoredTrainingSession(
                    hooks=hooks,
                    config=self._sess_config,
                    checkpoint_dir=ckpt_dir,
            ) as mon_sess:

                local_step = 0
                logger.info("training default graph %s" %
                            str(tf.get_default_graph()))
                while True:
                    try:
                        lr = self._model.get_learning_rate(
                            self._config.total_train_step, local_step,
                            self._config.base_lr)
                        if local_step % 100 == 0:
                            logger.info("current lr: %f", lr)
                        self.run_session(mon_sess, local_step,
                                         {learning_rate: lr})
                        local_step += 1
                        if self._config.train_steps != 0 and local_step >= self._config.train_steps:
                            break
                    except tf.errors.OutOfRangeError:
                        logger.info("OutOfRangeError, train: %d" % local_step)
                        break
                    except tf.errors.UnknownError as e:
                        logger.info("UnknownError, train: %d" % local_step)
                        break

            t2 = time.time()
            logger.info(
                '---------------------- training completed (total time: %ds)-----------------------'
                % (t2 - t1))

    def evaluate(self, valid_path):
        work_mode = self._work_mode_to_number['MODE_VALID']
        if len(valid_path) > 0:
            logger.info('len(valid_path) > 0, use offline data for predict: %s',
                        str(valid_path))
            self.start_sample(self._config['hub_worker'], 1, valid_path,
                              work_mode)
        elif self._config.kafka_train:
            logger.info('kafka_train: true, use kafka data for predict')
            self.start_sample(self._config['hub_worker'], 2, [], work_mode)
        else:
            logger.info('no valid_path, use dragon input')
            self.start_sample(self._config['hub_worker'],
                              self._src_to_number['SRC_DRAGON'], [], work_mode)

        hooks = []

        with tf.Graph().as_default():
            iter = self.prepare_input(2)
            hooks.append(DatasetInitializerHook(iter))
            self.eval_model_fn(iter, hooks)

            ckpt_dir = None
            if rank() == 0:
                ckpt_dir = self._config.local_ckpt_dir
            with tf.train.MonitoredTrainingSession(
                    hooks=hooks,
                    config=self._sess_config,
                    checkpoint_dir=ckpt_dir,
            ) as mon_sess:
                logger.info("eval default graph %s" %
                            str(tf.get_default_graph()))
                logger.info(
                    '---------------------- start validation -----------------------'
                )

                total_in_validation = 0
                while True:
                    try:
                        mon_sess.run(self._update_op)
                        total_in_validation += 1
                        if self._config.evaluate_steps != 0 and total_in_validation >= self._config.evaluate_steps:
                            break
                    except tf.errors.OutOfRangeError:
                        logger.info("OutOfRangeError, validate: %d" %
                                    total_in_validation)
                        break
                logger.info(
                    '---------------------- validation completed -----------------------'
                )

    def kafka_train(self):
        hooks = InitHooks()

        if self._model._config.is_online_warmup and self._model.restored_model_path == '':
            raise Exception('must provied warmup path for kafka train!')
        if (self._config['btq_topic_num']) == 0:
            self.start_sample(self._config['hub_worker'], 2, [])
        else:
            self.start_sample(self._config['hub_worker'], 4, [])

        t1 = time.time()
        logger.info(
            '---------------------- start training -----------------------')

        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.placeholder(dtype=tf.float32)
            input_iter = self.prepare_input(1)

            hooks.append(DatasetInitializerHook(input_iter))
            self.train_model_fn(input_iter, hooks, learning_rate)
            hooks.append(
                PeriodAvgNetworkVaiablesHook(
                    self._config.period_avg_network_vaiables_hook))

            logger.info(hooks)

            if len(list(filter(lambda x: type(x) == ClickhouseHook, hooks))) == 0 or \
               len(list(filter(lambda x: type(x) == SaveModelHook, hooks))) == 0:
                raise Exception(
                    'online train must add ClickhouseHook and SaveModelHook!')

            ckpt_dir = None
            if rank() == 0:
                ckpt_dir = self._config.local_ckpt_dir
            with tf.train.MonitoredTrainingSession(
                    hooks=hooks,
                    config=self._sess_config,
                    checkpoint_dir=ckpt_dir,
            ) as mon_sess:
                logger.info("training default graph %s" %
                            str(tf.get_default_graph()))
                training_round = 0
                while True:
                    training_round += 1
                    local_step = 0

                    logger.info(
                        "--------------- start round %d ----------------",
                        training_round
                    )

                    while local_step < self._config.save_ckpt_step:

                        try:
                            lr = self._config['base_lr']
                            if local_step % 100 == 0:
                                logger.info('current lr: %s', lr)
                            self.run_kafka_session(mon_sess, local_step,
                                                   {learning_rate: lr})
                            local_step += 1
                        except tf.errors.OutOfRangeError:
                            logger.info("OutOfRangeError, train: %d", local_step)
                            break

                    logger.info(
                        "--------------- end round %d ----------------",
                        training_round
                    )

    def count_feature(self, train_path: list):
        assert len(train_path) > 0, 'must provide train_path!'

        self.start_sample(self._config['hub_worker'],
                          self._src_to_number['SRC_HDFS'],
                          train_path,
                          self._work_mode_to_number['MODE_COUNT'])

        with tf.Graph().as_default():
            res = self._trainer_ops.count_feature()
            with tf.train.MonitoredTrainingSession(
                    config=self._sess_config,) as mon_sess:
                s = mon_sess.run(res)[0]

                d = {}
                feature_count = json.loads(s)
                if type(feature_count) == dict:
                    for name in feature_count:
                        if name in self._config.embedding_table:
                            field = int(name.split('_')[1])
                            table = self._config.embedding_table[name]

                            d[name] = {
                                'feature_count': feature_count[name],
                                'class_name': self._config.sparse_input[field],
                                'capacity': table['capacity'],
                                'hash_bucket_size': table['hash_bucket_size']
                            }

                logger.info('count_feature done, len: %d', len(d))
                return d

    def save_feature_count(self):
        path = os.path.join(
            self._config.ckp_save_nfs_path,
            '%s_feature_count' % (self._config.model_name),
            'feature_count.%s' % (datetime.now().strftime('%Y%m%d%H%M'))
        )

        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(path):
            fs.mkdir(path)

        self._model.save_feature_count(path)
