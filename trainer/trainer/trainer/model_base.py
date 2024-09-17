import tensorflow as tf
import math
import os
import re
import json
import pyarrow
from tensorflow.python.ops import variables
from abc import ABC, abstractmethod
from datetime import datetime
import codecs
from functools import wraps
import time
from collections import defaultdict

from .trainer_ops_wrapper import TrainerOps
from .sniper_config import SniperConf, parse_feature_config

from .util import logger, get_tf_session_config
from .util import get_dense_vars
from .util import get_variable
from .util import get_restore_vars

from .dist import rank

from .AUC import auc as auc_eval


class ModelBase(ABC):
    """Base class for all model definition.
    """
   
    def __init__(
        self,
        sniper_conf: SniperConf,
        trainer_ops: TrainerOps,
    ):
        self._config = sniper_conf
        self._trainer_ops = trainer_ops

        self._ckp_sess_config = get_tf_session_config(
            tf_intra_op_parallelism_threads = 64,
            tf_inter_op_parallelism_threads = 64,
            use_xla = self._config.use_xla
        )

        self._sess_config = get_tf_session_config(
            tf_intra_op_parallelism_threads = 64,
            tf_inter_op_parallelism_threads = 64,
            use_xla = self._config.use_xla
        )

        self._last_export_day = datetime.now().day
        self._last_export_time = datetime.now()
        self._full_export_flags = [False, False]

        self._last_result = {}
        self.restored_model_path = ''

    def create_dense_vars(self):
        """Create dense var on ps.
        """
        dense_vars = get_dense_vars()
        create_ops = []

        for var in dense_vars:
            op = self._trainer_ops.create_dense_var(var, var.name)
            create_ops.append(op)

        return create_ops

    def create_sparse_vars(self):
        """Create sparse vars on ps.
        """
        return [self._trainer_ops.create_embedding_table()]

    def pull_dense_vars(self):
        """Ops that pull variable parameters from ps.
        """
        pull_ops = []
        dense_vars = get_dense_vars()

        for var in dense_vars:
            op = self._trainer_ops.pull_variable(var.name, 1, var, var)
            pull_ops.append(op)

        return pull_ops

    def pull_sparse_vars(self):
        """Pull embedding parameters from ps.
        """
        pull_ops = []
        for i in range(len(self._config.input_sparse_emb_table_name)):
            var_name = self._config.input_sparse_emb_table_name[i]
            meta_var_name = self._config.meta_emb_table_name[i]

            var = None
            opt_var = None

            for v in tf.global_variables():
                if v.name == "%s:0" % meta_var_name:
                    var = v
                    continue

                if v.name == "%s/Adagrad:0" % (meta_var_name):
                    opt_var = v
                    continue

            if var and opt_var:
                with tf.device('/cpu:0'):
                    op = self._trainer_ops.pull_variable(var_name, 2, var, opt_var)
                pull_ops.append(op)
            else:
                logger.critical("could not find var: %s" % var_name)

        return pull_ops

    def push_dense_vars(self):
        """Ops that pull dense variable parameters from ps.
        """
        dense_vars = get_dense_vars()
        push_ops = []

        for var in dense_vars:
            op = self._trainer_ops.push_variable(var.name, 1, var, var)
            push_ops.append(op)

        return push_ops

    def push_sparse_vars(self):
        """Push sparse embedding to ps.
        """
        push_ops = []
        pushed_map = {}

        logger.info(
            "input_sparse_emb_table_name: %s",
            self._config.input_sparse_emb_table_name
        )

        for i in range(len(self._config.input_sparse_emb_table_name)):
            var_name = self._config.input_sparse_emb_table_name[i]
            var = None
            opt_var = None

            if var_name in pushed_map:
                continue

            pushed_map[var_name] = True

            for v in tf.global_variables():
                if v.name == "%s:0" % var_name:
                    var = v
                    continue
                if v.name == "%s/Adagrad:0" % (var_name):
                    opt_var = v
                    continue

            if var and opt_var:
                logger.info("add push sparse variable: %s, shape:%s" %
                            (var_name, var.shape))
                op = self._trainer_ops.push_variable(var_name, 2, var, opt_var)
                push_ops.append(op)
            else:
                logger.critical("could not find var: %s" % var_name)

        return push_ops

    def get_saved_dense_var_names(self):
        """Get saved dense variable names.
        """
        dense_vars = get_dense_vars()
        return [
            v.name for v in dense_vars if not v.name.endswith('/Adagrad:0') or
            v.name.endswith("/Adam:0") or v.name.endswith("/Adam_1:0")
        ]

    def save_dense_vars(
        self,
        version: int,
        ckp_type: int,
        ckp_target: int,
        nfs_path: str,
        queues: dict,
        var_2_emb_shard: dict = {}
    ):
        save_dense_ops = []

        if len(var_2_emb_shard) > 0:
            nfs_path = os.path.join(nfs_path, 'local')

        for var_name in self.get_saved_dense_var_names():
            logger.info('save_dense_vars: %s', var_name)
            queue = ""

            if ckp_target == 2:
                if var_name in queues:
                    queue = queues[var_name]
                else:
                    queue = queues['DEFAULT_BTQ_TOPIC']

            op = self._trainer_ops.save(
                version,
                var_name,
                ckp_type,
                ckp_target,
                nfs_path,
                queue
            )

            save_dense_ops.append(op)

        return save_dense_ops

    def get_saved_sparse_var_names(self):
        names = []
        saved_map = {}

        for i in range(len(self._config.input_sparse_emb_table_name)):
            var_name = self._config.input_sparse_emb_table_name[i]
            var = None
            opt_var = None

            if var_name in saved_map:
                continue

            saved_map[var_name] = True

            for v in tf.global_variables():
                if v.name == "%s:0" % var_name:
                    var = v
                    continue

            if var:
                names.append(var_name)
            else:
                logger.info('get save sparse cannot find var: %s', var_name)

        return names

    def save_sparse_vars(
        self,
        version: int,
        ckp_type: int,
        ckp_target: int,
        nfs_path: str,
        queues: dict,
        var_2_emb_shard: dict = {}
    ):
        """Save sparse embedding parameters to file or other storage.
        """
        save_ops = []

        for var_name in self.get_saved_sparse_var_names():
            queue = ""

            if ckp_target == 2:
                if var_name in queues:
                    queue = queues[var_name]
                else:
                    queue = queues['DEFAULT_BTQ_TOPIC']

            final_path = nfs_path
            if var_name in var_2_emb_shard:
                final_path = os.path.join(nfs_path, var_2_emb_shard[var_name])

            op = self._trainer_ops.save(
                version,
                var_name,
                ckp_type,
                ckp_target,
                final_path,
                queue
            )

            save_ops.append(op)

        return save_ops

    def check_ckp(
        self,
        version: int,
        ckp_type: int,
        ckp_target: int
    ):
        """Check whether checkpoint saving has finished.
        """
        check_ckp_ops = []
        op = self._trainer_ops.check_ckp(version, ckp_type, ckp_target)
        check_ckp_ops.append(op)
        return check_ckp_ops

    def restore_sparse_vars(self, model_path: str):
        """Restore variables from model_path.

        It will take some time. It must be waited for the restore operation to finish.
        """
        restore_ops = []

        if not tf.io.gfile.exists(model_path):
            logger.critical("model path not exist : %s" % model_path)

        files = tf.io.gfile.listdir(model_path)

        model_files = []
        model_file_names = tf.io.gfile.listdir(model_path)

        for f_n in model_file_names:
            model_files.append(model_path + "/" + f_n)

        # restore mbedding to ps
        restored_map = {}

        for i in range(len(self._config.input_sparse_emb_table_name)):
            var_name = self._config.input_sparse_emb_table_name[i]
            var = None
            opt_var = None

            if var_name in restored_map:
                continue

            restored_map[var_name] = True
            for v in tf.global_variables():
                if v.name == "%s:0" % var_name:
                    var = v
                    continue

            if var:
                shard_idxs = set()

                for f in model_files:
                    if f.startswith(model_path + "/" + var_name + "."):

                        vs = f.split(".")
                        if len(vs) >= 3:
                            tmp = vs[-2].split("_")
                            shard_idxs.add(int(tmp[0]))
                        else:
                            logger.critical(
                                "model file path is not valid: %s" % f)
                            return

                tmp = list(shard_idxs)
                tmp.sort()

                shard_num = len(shard_idxs)

                if shard_num == 0:
                    logger.info("var[%s] model file not found" % var_name)
                    continue

                # partial is not allowed
                config_shard_num = len(
                    self._config.embedding_table[var_name]['shard'])

                assert shard_num == config_shard_num, \
                    "var[%s] model_shard[%d] != config_shard[%d]" % (
                        var_name, shard_num, config_shard_num)

                if len(tmp) != (tmp[-1] + 1):
                    logger.critical("var[%s] model file is not invalid" %
                                    var_name)
                    return

                for shard_idx in shard_idxs:
                    shard_nfs_weight_paths = []
                    shard_nfs_adagrad_paths = []

                    for f in model_files:
                        if f.startswith(
                                model_path + "/" + var_name + "." + str(shard_idx)
                        ):
                            if f.endswith(".weight"):
                                shard_nfs_weight_paths.append(f)
                            elif f.endswith(".adagrad"):
                                shard_nfs_adagrad_paths.append(f)
                            else:
                                logger.critical(
                                    "model file path is not valid: %s" % f)
                                return

                    op = self._trainer_ops.restore(
                        var_name,
                        shard_idx,
                        shard_num,
                        shard_nfs_weight_paths,
                        shard_nfs_adagrad_paths
                    )

                    restore_ops.append(op)

        return restore_ops

    def freeze(self):
        dense_varnames = []
        dense_var_queues = []
        sparse_varnames = []
        sparse_var_queues = []
        dense_vars = get_dense_vars()
        sparse_vars = self._config.input_sparse_emb_table_name

        for var in dense_vars:
            # Parameter of optimizer do not need to be exported.
            if (var.name.endswith("/Adagrad:0")
                or var.name.endswith("/Adam:0")
                or var.name.endswith("/Adam_1:0")):
                continue

            dense_varnames.append(var.name)
            if self._config.use_btq:
                if var.name in self._config.var_2_btq:
                    dense_var_queues.append(self._config.var_2_btq[var.name])
                else:
                    dense_var_queues.append(
                        self._config.var_2_btq["DEFAULT_BTQ_TOPIC"])

                    logger.warn(
                        "could not find var: %s in var_2_btq, we use default topic: %s",
                        var.name,
                        self._config.var_2_btq["DEFAULT_BTQ_TOPIC"]
                    )

        freezed_map = {}

        for var_name in sparse_vars:
            if var_name in freezed_map:
                continue

            sparse_varnames.append(var_name)
            freezed_map[var_name] = True

            if self._config.use_btq and len(self._config.var_2_btq) > 0:
                if var_name in self._config.var_2_btq:
                    sparse_var_queues.append(self._config.var_2_btq[var_name])
                else:
                    sparse_var_queues.append(
                        self._config.var_2_btq["DEFAULT_BTQ_TOPIC"])

                    logger.warn(
                        "could not find var: %s in var_2_btq, we use default topic: %s",
                        var_name,
                        self._config.var_2_btq["DEFAULT_BTQ_TOPIC"]
                    )

        return self._trainer_ops.freeze(
            self._config.model_name,
            dense_varnames,
            dense_var_queues,
            sparse_varnames,
            sparse_var_queues
        )

    def get_learning_rate(
        self,
        total_train_step: int,
        cur_step: int,
        base_lr: float = 0.007
    ) -> float:
        if cur_step < total_train_step:
            lr = base_lr * math.exp((float(cur_step) / float(total_train_step) - 1) * 7)

            return lr

        return base_lr

    def _update_model_version(
        self,
        fs,
        model_export_path: str,
        cur_model_path: str
    ):
        model_version_path = os.path.join(model_export_path, 'model_version')
        model_versions = []

        if not fs.exists('%s/model_tf.data-00000-of-00001' % (cur_model_path)):
            logger.info('model_path: %s not success, skip', cur_model_path)
            return

        if fs.exists(model_version_path):
            with fs.open(model_version_path, 'rb') as f:
                content = str(f.readall(), encoding='utf8')

                for line in content.split('\n'):
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    if len(tokens) != 3:
                        continue

                    model_versions.append(line)

        model_versions.append("%s\tmd5\t1" % (cur_model_path))

        with fs.open(model_version_path, 'wb') as f:
            content = bytes('\n'.join(model_versions[-5:]) + '\n', encoding='utf8')
            f.write(content)

    @abstractmethod
    def inference(
        self,
        dnn_input,
        labels,
        is_cpu_ps: bool,
        is_train: bool,
        hooks,
        debug_info
    ):
        """Unified interface for model definition.
        """
        pass

    def exclude_dense_input(self, exclude_dense_fields):
        if len(exclude_dense_fields) > 0:
            input_list = list()

            for i in range(len(exclude_dense_fields)):
                input_list.append(
                    tf.placeholder(
                        tf.float32,
                        name="exclude_dense_field_" + str(i),
                        shape=[None, exclude_dense_fields[i]]
                    ))

            return tf.concat(input_list, 1)

        return None

    def sparse_input(self, export_mode: int = 0):
        """Sparse input tensor.

        Args:
            export_mode: 0 means all parameters will be saved using `tf.save`,
                including sparse embedding parameters. 1 means sparse embedding
                parameters will be saved by ps, not `tf.save`.
        """
        export_mode = 0

        if "export_mode" in self._config.conf["trainer"]:
            export_mode = int(self._config.conf["trainer"]["export_mode"])

        restored_model_path = self._get_warmup_model()

        input_list = list()
        for i in range(len(self._config.input_sparse_emb_table_name)):
            # Placeholder for sparse feature i.
            field_ph = tf.sparse_placeholder(tf.int64, name="field_" + str(i))

            # Embedding table variable name.
            emb_var_name = self._config.meta_emb_table_name[i]

            # Bucket size for sparse feature.
            bucket_size = self._config.input_sparse_emb_table_capacity[i]

            # Embedding size for sparse features. The default is 16.
            embedding_size = self._config.input_sparse_emb_size[i]

            if export_mode == 1:
                if (len(restored_model_path) == 0
                    or self._config.restore_mode == 1):
                    # No hash model need a small faked bucket size to create the variable.
                    bucket_size = 10001

            emb_var = get_variable(
                name = emb_var_name,
                shape = [bucket_size, embedding_size],
                initializer = tf.random_uniform_initializer(-0.01, 0.01),
                trainable = True,
                is_cpu_ps = True)

            embed = tf.nn.embedding_lookup_sparse(
                emb_var,
                field_ph,
                None,
                combiner="sum"
            )

            input_list.append(embed)

        return input_list

    def dense_input(self):
        """Concat all sparse feature embedding sum to get the first layer of network.
        """
        input_list = list()
        for i in range(len(self._config.input_dense_size)):
            dim = self._config.input_dense_size[i]
            input_list.append(
                tf.placeholder(tf.float32,
                               name="dense_field_" + str(i),
                               shape=[None, dim]))
        return input_list

    def _build_internal(self, is_train: bool):
        """Build the graph by feature list and model definition.
        """
        global_step = tf.train.get_or_create_global_step()

        input_list = []
        dnn_input = None

        sparse_input_list = self.sparse_input(self._config.export_mode)
        dense_input_list = self.dense_input()

        if self._config.split_input:
            sparse_user = sparse_input_list[:self._config.
                                            input_sparse_user_count]
            sparse_item = sparse_input_list[self._config.
                                            input_sparse_user_count:]

            dense_user = dense_input_list[:self._config.input_dense_user_count]
            dense_item = dense_input_list[self._config.input_dense_user_count:]

            split_inputs = [sparse_user, sparse_item, dense_user, dense_item]
            dnn_input = []

            for input in split_inputs:
                if len(input) > 0:
                    dnn_input.append(tf.concat(input, 1))
                else:
                    dnn_input.append(tf.placeholder(tf.float32, shape=[None,
                                                                       0]))
        else:
            input_list.extend(sparse_input_list)
            input_list.extend(dense_input_list)
            dnn_input = tf.concat(input_list, 1)

        labels = tf.placeholder(tf.int64, name='label')

        train_loss = self.inference(dnn_input, labels, True, is_train, None, [])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            if self._config["optimizer"] == "adam":
                logger.info("build adam")
                optimizer = tf.train.AdamOptimizer()
            elif self._config["optimizer"] == "adagrad":
                logger.info("build adagrad")
                optimizer = tf.train.AdagradOptimizer(0.1)
            else:
                assert 0, "use an unknown optimizer %s" % self._config[
                    "optimizer"]
            if self._config.mixed_precision:
                optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                    optimizer)

            train_op = optimizer.minimize(train_loss, global_step=global_step)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(),
                           name="init_op")

        return init_op

    def _get_last_model_version(self, model_warmup_path: str):
        fs = pyarrow.HadoopFileSystem()

        model_version_path = os.path.join(model_warmup_path, 'model_version')
        model_versions = []

        if fs.exists(model_version_path):
            with fs.open(model_version_path, 'rb') as f:
                content = str(f.readall(), encoding='utf8')

                for line in content.split('\n'):
                    if len(line) == 0:
                        continue
                    tokens = line.split()
                    if len(tokens) != 3:
                        continue
                    model_versions.append(tokens[0])

            if len(model_versions) > 0:
                return model_versions[-1]

        return ""

    def build(self, is_train: bool = True):
        with tf.Graph().as_default():
            # build graph
            init_op = self._build_internal(is_train)

            with tf.Session(config=self._ckp_sess_config) as sess:
                sess.run(init_op)

                return self.build_with_session(sess, is_train)

    def build_with_session(self, sess, is_train: bool = True):
        if rank() == 0:
            if tf.gfile.Exists(self._config.local_ckpt_dir):
                tf.gfile.DeleteRecursively(self._config.local_ckpt_dir)

            tf.gfile.MakeDirs(self._config.local_ckpt_dir)

        model_path = self._get_warmup_model()

        # Restore trainer from checkpoint.
        restore_var_list = get_restore_vars(self._config.restore_mode)

        saver = tf.train.Saver(restore_var_list)

        localckpt_var_list = [
            v for v in tf.global_variables() if v.name.find('embedding') < 0
        ]

        localckpt_saver = tf.train.Saver(localckpt_var_list)

        # Only rank 0 create var and restore, before training
        # root will broadcast global vars to others trainers
        if rank() == 0:
            # Create all need variables on parameter server
            create_ops = self.create_dense_vars() + self.create_sparse_vars()
            sess.run(create_ops)

            # Restore feature_count if use feature_inclusion_freq and restore_mode == 1.
            # Must after create var
            if self._config.feature_inclusion_freq > 0 and self._config.restore_mode == 1:
                self.restore_feature_count(sess)

            # Restore from model_tf.
            if len(model_path) > 0:
                logger.info("restore from model:" + str(model_path))
                saver.restore(sess, '%s/model_tf' % (model_path))

                # Save to local.
                localckpt_saver.save(sess, self._config.local_ckpt_dir + "/model")

                logger.info(
                    'save nn to localckpt[%s] complete',
                    self._config.local_ckpt_dir
                )

            # Restore ps from checkpoint.
            if len(model_path) > 0:
                if self._config.restore_mode == 0:
                    push_op = self.push_dense_vars() + self.push_sparse_vars()
                    logger.info(push_op)
                    sess.run(push_op)
                else:
                    sess.run(self.push_dense_vars())
                    logger.info(self._ckp_sess_config)

                    restore_ops = self.restore_sparse_vars(model_path)
                    logger.info(restore_ops)

                    sess.run(restore_ops)

            if self._config.use_btq and self._config.ckp_save_btq_incr_step != 0:
                # Incremental save on ps.
                now = datetime.now()
                version = int(now.timestamp())
                logger.info("save btq incr config: version(%d)", version)
                sess.run(
                    self.save_sparse_vars(version * 1000, 1, 2,
                                          self._config.ckp_save_nfs_path,
                                          self._config.var_2_btq, {}))

            # Freeze model
            sess.run(self.freeze())

    def build_meta_graph(self, is_train: bool = False):
        """Build meta for inference.
        """
        if rank() == 0:
            fs = pyarrow.HadoopFileSystem()
            path = os.path.join(self._config.model_export_root_path,
                                self._config.model_name)
            fs.mkdir(path)

            with fs.open(os.path.join(path, 'model_feature'),
                         'wb') as f_feature_file:
                with codecs.open(self._config.feature_file, 'r',
                                 'utf-8') as local_feature_file:
                    f_feature_file.write(
                        bytes(local_feature_file.read(), encoding='utf8'))
                    logger.info('put feature file to hdfs path: %s',
                                os.path.join(path, 'model_feature'))

            with tf.Graph().as_default():
                init_op = self._build_internal(is_train)
                saver = tf.train.Saver()
                with tf.Session(config=self._sess_config) as sess:
                    sess.run(init_op)
                    if is_train:
                        meta_filename = os.path.join(
                            self._config.dirname,
                            '%s_train.meta' % (self._config.model_name))
                    else:
                        meta_filename = os.path.join(
                            self._config.dirname,
                            '%s_predict.meta' % (self._config.model_name))

                    tf.train.export_meta_graph(meta_filename)
                    logger.info('store meta to %s' % (meta_filename))

                    with fs.open(os.path.join(path, 'model.meta'),
                                 'wb') as f_meta:
                        with open(meta_filename, 'rb') as local_meta:
                            f_meta.write(local_meta.read())
                            logger.info('put meta to hdfs path: %s',
                                        os.path.join(path, 'model.meta'))

    def put_info_json(
        self,
        cur_model_path: str,
        shards: list
    ):
        info = {'sniper_version': '3.0.0'}

        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(cur_model_path):
            fs.mkdir(cur_model_path)

        for x in shards + ['']:
            with fs.open(os.path.join(cur_model_path, x, 'info.json'),
                         'wb') as f:
                content = bytes(json.dumps(info, ensure_ascii=False),
                                encoding='utf8')
                f.write(content)

    def save(
        self,
        now: datetime,
        ckp_type: int,
        ckp_target: int,
        wait_ps: bool = True
    ):
        version = int(now.timestamp())
        if rank() == 0:
            logger.info('start save, version(%d) ckp_type(%d) ckp_target(%d)',
                        version, ckp_type, ckp_target)

            if ckp_target == 1:
                # hdfs.
                save_rollback = False

                # big ps rollback
                if self._config.use_btq:
                    rollback_root = '/home/ad/big_model_rollback'

                    rollback_full_path = os.path.join(
                        rollback_root,
                        self._config.model_name
                    )

                    last_rollback = self._get_last_model_version(
                        rollback_full_path
                    )

                    if (last_rollback == ''
                        or last_rollback.split('.')[-1][:8] <
                        datetime.now().strftime('%Y%m%d')):
                        logger.info(
                            'last_rollback: %s, save rollback model: %s',
                            last_rollback,
                            os.path.join(rollback_root,
                                         self._config.model_name))
                        cur_model_path = '%s/model_tf.%s' % (
                            rollback_full_path, now.strftime("%Y%m%d%H%M"))

                        rollback_version = int(datetime.now().timestamp()) + 10

                        shards = [
                            x for x in self._config.var_2_emb_shard.values()
                            if x == 'local' or str(x).isdigit()
                        ]

                        self.put_info_json(cur_model_path, list(set(shards)))
                        logger.info(
                            'rollback_version: %d, rollback_path: %s',
                            rollback_version,
                            cur_model_path
                        )

                        self.save_to_hdfs(
                            cur_model_path,
                            self._config.export_mode,
                            rollback_version,
                            self._config.var_2_emb_shard,
                            wait_ps
                        )

                        save_rollback = True

                if not save_rollback:
                    if len(self._config.ckp_save_nfs_path) == 0:
                        logger.critical("save nfs path is empty")
                        return

                    model_save_path = os.path.join(
                        self._config.ckp_save_nfs_path,
                        self._config.model_name
                    )

                    timestamp = now.strftime("%Y%m%d%H%M")
                    cur_model_path = os.path.join(
                        model_save_path,
                        "model_tf.%s" % timestamp
                    )

                    self.save_to_hdfs(
                        cur_model_path,
                        self._config.export_mode,
                        version,
                        {},
                        wait_ps
                    )

                    if self._config.feature_count_save_path != '':
                        self.save_feature_count(
                            '%s/feature_count.%s',
                            self._config.feature_count_save_path,
                            timestamp
                        )
            elif ckp_target == 2:
                # btq
                with tf.Graph().as_default():
                    # build graph
                    init_op = self._build_internal(True)
                    save_ops = []

                    if ckp_type == 1:
                        # incr
                        save_ops = self.save_dense_vars(
                            version * 1000, 1, 2,
                            self._config.ckp_save_nfs_path,
                            self._config.var_2_btq)
                    elif ckp_type == 2:
                        # full
                        save_ops = self.save_dense_vars(
                            version * 1000, 2, 2,
                            self._config.ckp_save_nfs_path,
                            self._config.var_2_btq) + self.save_sparse_vars(
                                version * 1000, 2, 2,
                                self._config.ckp_save_nfs_path,
                                self._config.var_2_btq, {})

                    check_ckp_op = self.check_ckp(
                        version * 1000,
                        ckp_type,
                        ckp_target
                    )

                    with tf.Session(config=self._sess_config) as sess:
                        sess.run(init_op)
                        sess.run(save_ops)
                        if wait_ps:
                            sess.run(check_ckp_op)

    def save_to_hdfs(
        self,
        cur_model_path: str,
        export_mode: int,
        version: int,
        var_2_emb_shard: dict = {},
        wait_ps: bool = True
    ):
        logger.info(
            'cur_model_path: %s, export_mode: %d, wait_ps: %s',
            cur_model_path,
            export_mode,
            wait_ps
        )

        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(cur_model_path):
            fs.mkdir(cur_model_path)

        with tf.Graph().as_default():
            # build graph
            init_op = self._build_internal(True)

            # pull lastest from ps
            pull_dense_ops = self.pull_dense_vars()
            pull_sparse_ops = self.pull_sparse_vars()

            check_ckp_op = self.check_ckp(version * 1000, 2, 1)
            saver = tf.train.Saver()

            with tf.Session(config=self._ckp_sess_config) as sess:
                sess.run(init_op)
                if export_mode == 0:
                    sess.run(pull_dense_ops + pull_sparse_ops)
                else:
                    # worker只导出 dense部分
                    sess.run(pull_dense_ops)

                var_names = [v.name for v in tf.global_variables()]
                var_values = sess.run(var_names)

                if export_mode == 1:
                    # 发起 ps nfs全量导出
                    save_path = "hdfs://default%s" % cur_model_path
                    save_ps_vars = self.save_sparse_vars(
                        version * 1000, 2, 1, save_path, {}, var_2_emb_shard) + \
                        self.save_dense_vars(
                        version * 1000, 2, 1, save_path, {}, var_2_emb_shard)
                    sess.run(save_ps_vars)

                try:
                    saver.save(sess,
                               'hdfs://default%s/model_tf' % cur_model_path,
                               write_meta_graph=False,
                               write_state=False)
                except Exception as e:
                    logger.info(
                        'save model_tf fail, cur_model_path: %s/model_tf',
                        cur_model_path)
                    logger.info(e)
                    return

                if wait_ps and export_mode == 1:
                    sess.run(check_ckp_op)

                self._update_model_version(fs, os.path.dirname(cur_model_path),
                                           cur_model_path)

    def export_to_hdfs(self):
        self.save(datetime.now(), 2, 1, True)

    def is_meet_save_condition(
        self,
        auc: float,
        train_loss: float,
        prob_mean: float,
        real_mean: float,
        step: int,
    ) -> bool:
        if auc > self._config.auc_lower_bound:
            return True
        else:
            return False

    def to_compatible_checkpoint(
        self,
        model_path: str,
        feature_file: str,
        checkpoint_version: str = '3',
    ) -> str:
        if rank() > 0:
            return ''

        assert len(model_path) > 0, 'must provide model_path, but is empty!'

        if not model_path.startswith('viewfs'):
            model_path = "viewfs://hadoop-lt-cluster%s" % model_path
        if not model_path.endswith('model_tf'):
            model_path = os.path.join(model_path, 'model_tf')

        with tf.Graph().as_default():
            # build graph
            init_op = self._build_internal(True)

            # restore trainer from checkpoint
            restore_var_list = get_restore_vars(
                self._config.restore_mode)
            saver = tf.train.Saver(restore_var_list)
            with tf.Session(config=self._ckp_sess_config) as sess:
                sess.run(init_op)

                # only root create var and restore, before training
                # root will broadcast global vars to others vars
                # create all need variables on parameter server
                create_ops = self.create_dense_vars() + self.create_sparse_vars(
                )
                sess.run(create_ops)
                logger.info("restore from model:" + str(model_path))
                saver.restore(sess, model_path)

                if self._config.restore_mode == 0:
                    push_ops = self.push_dense_vars() + self.push_sparse_vars()
                    sess.run(push_ops)
                else:
                    raise Exception('unsupported restore_mode: %d' %
                                    (self._config.restore_mode))

        with tf.Graph().as_default():
            # build graph
            self._build_internal(True)
            # pull lastest from ps
            pull_ops = self.pull_sparse_vars() + self.pull_dense_vars()
            arr = []
            v = []
            old_names = []
            new_names = []
            assignops = []

            with tf.device('CPU:0'):
                for i, class_name in enumerate(self._config.sparse_input):
                    t = tf.get_default_graph().get_tensor_by_name(
                        'emb_%s:0' % (class_name))
                    t1 = tf.Variable(t, name='embedding_%d' % (i))

                    t_ada = tf.get_default_graph().get_tensor_by_name(
                        'emb_%s/Adagrad:0' % (class_name)
                    )

                    t1_ada = tf.Variable(t_ada, name='embedding_%d/Adagrad' % (i))

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer(),
                               name="init_op")

            new_saver = tf.train.Saver()
            with tf.Session(config=self._sess_config) as sess:
                sess.run(init_op)
                sess.run(pull_ops)
                sess.run(arr)
                values = sess.run(old_names)
                for i in range(len(new_names)):
                    assignops.append(tf.assign(new_names[i], values[i]))

                sess.run(assignops)

                res = sess.run(arr[:2])
                logger.info(res)
                logger.info('compare')
                logger.info(sess.run(v))

                logger.info('all variables: ')
                for v in tf.global_variables():
                    logger.info(v.name)
                p = re.compile('(/model_tf.[\d]+)')
                new_path = p.sub(r'_compatible\1', model_path)
                new_saver.save(sess,
                               new_path,
                               write_meta_graph=False,
                               write_state=False)
                logger.info('save to new_path: %s', new_path)

                return new_path

    def _get_warmup_model(self):
        model_path = self._config.warmup_tf_model
        if len(model_path) == 0 and len(self._config.warmup_path) > 0:
            model_path = self._get_last_model_version(self._config.warmup_path)

        if len(model_path) > 0 and not model_path.startswith('viewfs'):
            model_path = 'viewfs://hadoop-lt-cluster%s' % (model_path)

        self.restored_model_path = model_path
        return model_path

    def save_feature_count(self, path):
        logger.info('start save_feature_count, path: %s', path)
        with tf.Graph().as_default():
            nfs_path = 'hdfs://default%s' % (path)

            ops = []
            for i in range(len(self._config.input_sparse_emb_table_name)):
                var_name = self._config.input_sparse_emb_table_name[i]
                op = self._trainer_ops.save_feature_count(var_name, nfs_path)
                ops.append(op)

            with tf.train.MonitoredTrainingSession(
                    config=self._ckp_sess_config,) as mon_sess:
                mon_sess.run(ops)

            self._update_feature_count_version(path)
            logger.info('save_feature_count done, path: %s', path)

    def _update_feature_count_version(self, path):
        version_path = os.path.join(self._config.feature_count_save_path,
                                    'feature_count_version')
        versions = []

        fs = pyarrow.HadoopFileSystem()
        if fs.exists(version_path):
            with fs.open(version_path, 'rb') as f:
                versions = str(f.readall(), encoding='utf8').strip().split()

        versions.append(path)
        with fs.open(version_path, 'wb') as f:
            content = bytes('\n'.join(versions), encoding='utf8')
            f.write(content)

    def get_feature_count_path(self):
        if self._config.feature_count_version != '':
            return self._config.feature_count_version

        if len(self._config.feature_count_path) > 0:
            fs = pyarrow.HadoopFileSystem()
            version_path = os.path.join(self._config.feature_count_path,
                                        'feature_count_version')
            versions = []
            if fs.exists(version_path):
                with fs.open(version_path, 'rb') as f:
                    content = str(f.readall(), encoding='utf8').strip().split()
                    if len(content) > 0:
                        return context[0]

        return ''

    def restore_feature_count(self, sess = None):
        path = self.get_feature_count_path()
        if path == '':
            return

        fs = pyarrow.HadoopFileSystem()
        if not fs.exists(path):
            logger.info('feature_count_path: %s not exists', path)
            return

        logger.info('start restore_feature_count, feature_count_path: %s', path)

        d = defaultdict(dict)
        for x in fs.ls(path):
            suffix = x.split('/')[-1]
            if not suffix.startswith('embedding_'):
                continue

            arr = suffix.split('.')
            field = int(arr[0].split('_')[-1])
            shard_idx = int(arr[1])
            if shard_idx not in d[field]:
                d[field][shard_idx] = []

            d[field][shard_idx].append((x))

        ops = []
        for i in range(len(self._config.input_sparse_emb_table_name)):
            var_name = self._config.input_sparse_emb_table_name[i]
            shard_paths = []
            for shard_idx in range(len(d[i])):
                shard_paths.append(','.join(d[i][shard_idx]))

            op = self._trainer_ops.restore_feature_count(var_name, shard_paths)
            ops.append(op)

        sess.run(ops)
        logger.info('restore_feature_count done, path: %s', path)

    def migrate_checkpoint(
        self,
        feature_config: str,
        checkpoint: str,
        ignore_size_mismatch=False,
    ):
        if rank() != 0:
            return

        with tf.device('/cpu:0'):
            with tf.Graph().as_default():
                with tf.Session(config=self._sess_config) as sess:
                    self._migrate_checkpoint(sess, feature_config, checkpoint,
                                             ignore_size_mismatch)

    def _migrate_checkpoint(
        self,
        sess,
        feature_config: str,
        checkpoint: str,
        ignore_size_mismatch: bool = False,
    ):
        if not checkpoint.endswith("model_tf"):
            checkpoint = self._get_last_model_version(checkpoint)

        checkpoint_version = self._config.checkpoint_version

        checkpoint_version = self._config.checkpoint_version
        if len(checkpoint) > 0 and not checkpoint.startswith('viewfs'):
            checkpoint = 'viewfs://hadoop-lt-cluster%s' % checkpoint

        assert len(checkpoint) > 0, "checkpoint invalid"

        p = re.compile('(/model_tf.[\d]+)')
        new_path = p.sub(r'_migrate\1', checkpoint)
        logger.info("checkpoint will migrate from %s into %s", checkpoint,
                    new_path)

        model_path = checkpoint[:len(checkpoint) - len("/model_tf")]
        new_model_path = new_path[:len(new_path) - len("/model_tf")]

        export_mode = 0
        if "export_mode" in self._config.conf["trainer"]:
            export_mode = int(self._config.conf["trainer"]["export_mode"])

        embedding_files = []
        embedding_opt_files = []
        embedding2shard = {}
        new_opt_path = ""
        if export_mode == 1:
            model_file_names = tf.io.gfile.listdir(model_path)
            logger.info('model_file_names is %s' % model_file_names)
            for f_n in model_file_names:
                if "embedding_" not in f_n:
                    continue
                modelfile = model_path + "/" + f_n
                if checkpoint_version == '3' and f_n.endswith('.adagrad'):
                    embedding_opt_files.append(modelfile)
                else:
                    embedding_files.append(model_path + "/" + f_n)
                # shard
                emb_idx = -1
                shard_idx = -1

                for s in f_n.split("."):
                    if "embedding_" in s:
                        emb_idx = int(s.split("_")[1])
                    if "shard" in s:
                        shard_idx = int(s[len("shard"):])
                    if "_" in s and "embedding" not in s and "sparse" not in s:
                        shard_idx = int(s.split('_')[0])

                logger.info('emb_idx is %s' % emb_idx)
                logger.info('shard_idx is %s' % shard_idx)

                if emb_idx < 0 or shard_idx < 0:
                    logger.info("model file %s invaild" % f_n)
                    return

                if emb_idx not in embedding2shard:
                    embedding2shard[emb_idx] = [shard_idx]
                else:
                    if shard_idx not in embedding2shard[emb_idx]:
                        embedding2shard[emb_idx].append(shard_idx)

            if checkpoint_version == '2':
                opt_path = "%s.opt" % model_path
                model_file_names = tf.io.gfile.listdir(opt_path)
                for f_n in model_file_names:
                    if "embedding_" not in f_n:
                        continue
                    embedding_opt_files.append(opt_path + "/" + f_n)

        data = parse_feature_config(feature_config, [0])
        logger.info("data is %s" % data)
        src_feats = data["feature_column"]
        logger.info('src_feats is %s' % src_feats)
        dst_feats = self._config.feature_column

        src_emb_table = data["embedding_table"]
        dst_emb_table = self._config.embedding_table

        # dst 2 src feat
        has_size_mismatch = False
        related = {}
        slot2src_idx = {}
        for i, dst_feat in enumerate(dst_feats):
            target = None
            slot, prefix = -1, -1
            class_type = dst_feat["class"]
            class_name = dst_feat['class_name']
            if class_type == "numeric_column":
                size = dst_feat[class_type]["dim"]
            elif class_type == 'seq_column':
                embedding_name = dst_feat[class_type]["emb_table"]
                size = dst_emb_table[embedding_name]["hash_bucket_size"]
                slot = dst_feat["attrs"]["share_slot"]
                if slot in slot2src_idx:
                    related[i] = slot2src_idx[slot]
                    continue
            else:
                embedding_name = dst_feat[class_type]["emb_table"]
                size = dst_emb_table[embedding_name]["hash_bucket_size"]
                prefix = dst_feat["attrs"]["prefix"]

            for j, src_feat in enumerate(src_feats):
                src_class_type = src_feat["class"]
                src_class_name = src_feat["class_name"]
                if class_type != src_class_type:
                    continue
                if src_class_type == "numeric_column":
                    src_size = src_feat[class_type]["dim"]
                    if class_name != src_class_name:
                        continue
                elif src_class_type == "seq_column":
                    src_slot = src_feat["attrs"]["share_slot"]
                    src_embedding_name = src_feat[src_class_type]["emb_table"]
                    src_size = src_emb_table[src_embedding_name][
                        "hash_bucket_size"]
                    if slot != src_slot:
                        continue
                else:
                    src_prefix = src_feat["attrs"]["prefix"]
                    src_embedding_name = src_feat[src_class_type]["emb_table"]
                    src_size = src_emb_table[src_embedding_name][
                        "hash_bucket_size"]
                    if class_name != src_class_name or prefix != src_prefix:
                        continue

                if src_size != size:
                    logger.info("feature %s size mismatch, src %d, dst %d",
                                class_name, src_size, size)
                    has_size_mismatch = True
                    break

                target = j
                break
            related[i] = target
            # Save seq_column for later smae slot.
            if class_type == 'seq_column' and slot != -1:
                slot2src_idx[slot] = target

        if has_size_mismatch and not ignore_size_mismatch:
            logger.info("size mismatch and ignore_size_mismatch is False")
            return

        # Change train_config ps_shard.
        if export_mode == 1:
            # Get all shard.
            all_shards = {}
            for x in self._config.embedding_table:
                for addr in self._config.embedding_table[x]['shard']:
                    if addr not in all_shards:
                        all_shards[addr] = 0
                    all_shards[addr] += 1

            addr2idx = {
                addr: idx for idx, addr in enumerate(
                    self._config.conf['clusterspec']['ps'])
            }
            changed_set = set()
            # Change shard according to feat relation.
            for dst_feat_idx, src_feat_idx in related.items():
                dst_feat = dst_feats[dst_feat_idx]
                dst_class_type = dst_feat["class"]

                if src_feat_idx is None or dst_class_type == "numeric_column":
                    continue
                dst_emb_name = dst_feat[dst_class_type]["emb_table"]

                if dst_emb_name in changed_set:
                    continue

                shard_num = len(embedding2shard[src_feat_idx])
                dst_shard_num = len(self._config.embedding_table[dst_emb_name]["shard"])

                if dst_shard_num == shard_num:
                    logger.info("ps shard of %s eq with checkpoint" % dst_emb_name)
                    continue
                elif dst_shard_num > shard_num:
                    logger.info("ps shard of %s gt checkpoint, remove some" % dst_emb_name)

                    ps_shard = self._config.embedding_table[dst_emb_name]['shard'][:shard_num]

                    for addr in self._config.embedding_table[dst_emb_name]['shard'][shard_num:]:
                        all_shards[addr] -= 1
                else:
                    if len(all_shards) < shard_num:
                        logger.info(
                            "ps not enough, need %s for %s, actual %s",
                            shard_num,
                            dst_emb_name,
                            len(all_shards)
                        )

                        return

                    logger.info("ps shard of %s lt checkpoint, add some", dst_emb_name)

                    ps_shard = self._config.embedding_table[dst_emb_name]["shard"]

                    while True:
                        # Find min count.
                        min_count = 100000
                        min_addr = None
                        for addr, count in all_shards.items():
                            logger.info("==== dst_feat:{}, ".format(dst_feat))
                            if addr in ps_shard:
                                continue
                            if count < min_count:
                                min_count = count
                                min_addr = addr
                        ps_shard.append(min_addr)
                        all_shards[min_addr] += 1
                        if len(ps_shard) == shard_num:
                            break

                # Write back.
                self._config.embedding_table[dst_emb_name]['shard'] = ps_shard
                self._config.conf["final_ps_shard"][str(dst_feat_idx)] = {
                    'load': 0.0,
                    'shard': [addr2idx[x] for x in ps_shard]
                }

                changed_set.add(dst_emb_name)

            # Out for loop.
            self._config.ps_shard = {
                x: self._config.embedding_table[x]['shard']
                for x in self._config.embedding_table
            }

            self._config.total_sparse_shard = sum([
                len(self._config.embedding_table[x]['shard'])
                for x in self._config.embedding_table
            ])

            self._config.conf["embedding_table"] = self._config.embedding_table

            # Write file back.
            conf_file = self._trainer_ops._conf_file
            json.dump(
                self._config.conf,
                codecs.open(conf_file, 'w', 'utf-8'),
                ensure_ascii=False,
                indent=4
            )

            ps_shard = self._config.conf['final_ps_shard']
            logger.info("final_ps_shard:{}".format(ps_shard))
            for idx, conf in (dict.items(ps_shard)):
                logger.info(idx, conf)
                for id_shard in range(len(conf['shard'])):
                    logger.info('===idx:{}, conf:{}, cur:{}'.format(
                        idx, conf, conf['shard'][id_shard]))
                    if isinstance(conf['shard'][id_shard], str):
                        conf['shard'][id_shard] = int(
                            conf['shard'][id_shard].split('.')[0].split(
                                '-')[-1])
                    logger.info(conf)
            logger.info(ps_shard)

            shard_filename = os.path.join(
                self._config.conf['trainer']['dirname'],
                'ps_shard.json'
            )

            json.dump(
                ps_shard,
                codecs.open(shard_filename, 'w', 'utf-8'),
                ensure_ascii=False,
                indent=4
            )

            logger.info(
                "write conf file %s and shard file %s back",
                conf_file,
                shard_filename
            )

        # Build graph.
        init_op = self._build_internal(True)
        sess.run(init_op)

        name2var = {
            var.name.split(":")[0]: var for var in tf.global_variables()
        }

        sess.run(self.create_sparse_vars())
        reader = tf.train.load_checkpoint(checkpoint)

        # 1. Change order of embedding.
        logger.info("related is %s" % related)
        ops = []
        transfered_emb_set = set()
        for dst_feat_idx, src_feat_idx in related.items():
            dst_feat = dst_feats[dst_feat_idx]
            dst_class_type = dst_feat['class']
            dst_class_name = dst_feat['class_name']
            if src_feat_idx is None or dst_class_type == "numeric_column":
                continue

            src_feat = src_feats[src_feat_idx]
            src_emb_name = src_feat[dst_class_type]["emb_table"]
            src_emb_ada_name = "%s/Adagrad" % src_emb_name
            dst_emb_name = dst_feat[dst_class_type]["emb_table"]
            dst_emb_ada_name = "%s/Adagrad" % dst_emb_name

            if dst_class_type == 'seq_column' and dst_emb_name in transfered_emb_set:
                logger.info(
                    "feature %s shared embedding %s has already been transfered, skip",
                    dst_class_name,
                    dst_emb_name
                )
                continue

            if dst_class_type == 'seq_column' and dst_emb_ada_name in transfered_emb_set:
                logger.info(
                    "feature %s shared embedding %s has already been transfered, skip",
                    dst_class_name,
                    dst_emb_ada_name
                )
                continue

            logger.info(
                "feature %s mv embedding %s to %s",
                dst_class_name,
                src_emb_name,
                dst_emb_name
            )

            logger.info(
                "feature %s mv embedding %s to %s",
                dst_class_name,
                src_emb_ada_name,
                dst_emb_ada_name
            )

            if export_mode == 1:
                if checkpoint_version == '3':
                    target = src_emb_name + "."  # embedding_7.
                    shard_idxs = set()
                    for f in embedding_files:
                        if target not in f:
                            continue
                        vs = f.split(".")
                        logger.info("vs is %s", vs)
                        if len(vs) >= 3:
                            shard_idxs.add(int(vs[-2].split("_")[0]))
                            logger.info("shard_idxs is %s", embedding_files)
                        else:
                            logger.critical("model file path is not valid: %s", f)
                            return
                    shard_num = len(shard_idxs)
                    logger.info("shard_idxs is %s", shard_idxs)
                    for shard_idx in shard_idxs:
                        shard_nfs_weight_paths = []
                        shard_nfs_adagrad_paths = []

                        for f in embedding_files + embedding_opt_files:
                            if target + str(shard_idx) + '_' not in f:
                                logger.info("target is %s", target)
                                logger.info("shard_idx is %s", shard_idx)
                                continue
                            if f.endswith(".adagrad"):
                                shard_nfs_adagrad_paths.append(f)

                                logger.info("shard_nfs_adagrad_paths is %s",
                                            shard_nfs_adagrad_paths)
                            else:
                                shard_nfs_weight_paths.append(f)
                                logger.info("shard_nfs_weight_paths is %s",
                                            shard_nfs_weight_paths)
                        logger.info(
                            "dst_emb_name %s, shard_idx %s, weight %s, adagrad %s",
                            dst_emb_name, shard_idx, shard_nfs_weight_paths,
                            shard_nfs_adagrad_paths)

                        op = self._trainer_ops.restore(dst_emb_name, shard_idx,
                                                      shard_num,
                                                      shard_nfs_weight_paths,
                                                      shard_nfs_adagrad_paths)
                        ops.append(op)
            else:
                src_emb = reader.get_tensor(src_emb_name)
                src_emb_ada = reader.get_tensor(src_emb_ada_name)
                dst_emb = name2var[dst_emb_name]
                dst_emb_ada = name2var[dst_emb_ada_name]
                ops.append(tf.assign(dst_emb, src_emb))
                ops.append(tf.assign(dst_emb_ada, src_emb_ada))

            transfered_emb_set.add(dst_emb_name)
            transfered_emb_set.add(dst_emb_ada_name)

        if len(ops) > 0:
            sess.run(ops)

        # 2. Change shape of first layer of dnn.
        ops = []
        src_input_size_list = []
        for feat in src_feats:
            cls = feat["class"]
            if cls == "embedding_column" or cls == "seq_column":
                emb_table = feat[cls]["emb_table"]
                size = data["embedding_table"][emb_table]["dim"]
                src_input_size_list.append(size)
            elif cls == "numeric_column":
                src_input_size_list.append(feat["numeric_column"]["dim"])

        dst_input_size_list = []
        for feat in dst_feats:
            cls = feat["class"]
            if cls == "embedding_column" or cls == "seq_column":
                emb_table = feat[cls]["emb_table"]
                size = 16
                dst_input_size_list.append(size)
            elif cls == "numeric_column":
                dst_input_size_list.append(feat["numeric_column"]["dim"])

        src_input_size = sum(src_input_size_list)
        dst_input_size = self._config.input_sparse_total_size + \
            self._config.input_dense_total_size
        logger.info("dnn_input, size %d -> %d", src_input_size, dst_input_size)

        # Default.
        for var_name in name2var:

            if ("embedding_" in var_name
                or 'global_step' in var_name
                or 'out_of_range_ind' in var_name
                or "BigSumEmbedding" in var_name):
                continue

            if not reader.has_tensor(var_name):
                continue

            var = name2var[var_name]

            src_var = reader.get_tensor(var_name)

            if src_var.shape == var.shape:
                if src_var.shape == ():
                    logger.info("mv %s %s" % (var_name, var_name))
                    a = sess.run(tf.assign(var, src_var))
                    continue
                elif src_var.shape[0] != src_input_size or var.shape[
                        0] != dst_input_size:
                    logger.info("mv %s %s" % (var_name, var_name))
                    a = sess.run(tf.assign(var, src_var))
                    continue

            # shape not equal and is not first layer
            if src_var.shape[0] != src_input_size:
                logger.info(
                    "var:{} shape is change and is not first layer, so reinit!".
                    format(var_name))
                continue

            # Reorder the field.
            var_parts = []
            dsts = tf.split(var, var.shape[0], axis=0)

            logger.info(' ** Current var: %s' % var_name)

            for i, feat in enumerate(dst_feats):
                # Split and concat.
                shape = var.shape.as_list()
                if feat["class"] == "numeric_column":
                    shape[0] = self._config.input_dense_size[i - len(self._config.sparse_input)]
                else:
                    shape[0] = self._config.input_sparse_emb_size[i]

                src_feat_idx = related[i]

                if src_feat_idx is None:
                    import numpy as np
                    if feat["class"] != "numeric_column":
                        cstart = i * self._config.input_sparse_emb_size[i]

                        logger.info(
                            "Copy sparse feature: %d, cstart: %d, size: %d",
                            i,
                            cstart,
                            shape[0]
                        )

                        for ii in range(shape[0]):
                            var_parts.append(dsts[cstart + ii])
                    else:
                        dense_start = 16 * len(self._config.sparse_input)
                        tcsum = np.cumsum(self._config.input_dense_size)
                        cstart = dense_start + tcsum[i-len(self._config.sparse_input)] - shape[0]

                        logger.info(
                            "Copy dense feature: len of dsts: %d, csize: %d, ci: %d",
                            len(dsts),
                            shape[0],
                            i
                        )

                        for ii in range(shape[0]):
                            var_parts.append(dsts[cstart + ii])

                    continue

                sizes = [
                    sum(src_input_size_list[:src_feat_idx]),
                    src_input_size_list[src_feat_idx],
                    sum(src_input_size_list[src_feat_idx + 1:])
                ]

                _, this_t, _ = tf.split(src_var, sizes, axis=0)
                var_parts.append(this_t)

            logger.info("mv %s %s into %s %s" %
                        (var_name, src_var.shape, var_name, var.shape))
            tensor = sess.run(tf.concat(var_parts, axis=0))
            sess.run(tf.assign(var, tensor))

        migrate_vars = locals()
        migrate_vars.pop('self')
        self.migrate_complex_network(**migrate_vars)

        # Save.
        logger.info("start to save model!")
        if export_mode == 1:
            p = re.compile('/model_tf.([\d]+)')
            version = int(p.search(model_path).group(1))
            save_path = "hdfs://default%s" % new_model_path[
                len("viewfs://hadoop-lt-cluster"):]
            logger.info("save sparse embedding into %s" % save_path)
            save_ps_vars = self.save_sparse_vars(version, 2, 1, save_path, {}, {})

            sess.run(save_ps_vars)
            check_ckp_op = self.check_ckp(version * 1000, 2, 1)
            sess.run(check_ckp_op)
        saver = tf.train.Saver()
        saver.save(sess, new_path, write_meta_graph=False, write_state=False)
        logger.info("save into new path %s", new_path)

        return new_path

    def migrate_complex_network(self, **kwargs):
        """Extensions for complex network process.
        """
        pass
