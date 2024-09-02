import tensorflow as tf
"""
Borrowed from tensorflow to patch reset_op for the metrics AUC
"""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


def _safe_div(numerator, denominator, name):
    """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
    t = math_ops.truediv(numerator, denominator)
    zero = array_ops.zeros_like(t, dtype=denominator.dtype)
    condition = math_ops.greater(denominator, zero)
    zero = math_ops.cast(zero, t.dtype)
    return array_ops.where(condition, t, zero, name=name)


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

  If running in a `DistributionStrategy` context, the variable will be
  "tower local". This means:

  *   The returned object will be a container with separate variables
      per replica/tower of the model.

  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica/tower.

  *   To get a metric's result value, we need to sum the variable values
      across the replicas/towers before computing the final answer.
      Furthermore, the final answer should be computed once instead of
      in every replica/tower. Both of these are accomplished by
      running the computation of the final result value inside
      `tf.contrib.distribution_strategy_context.get_tower_context(
      ).merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a tower-local variable in a computation returns
      the sum across all replicas/towers.

  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.

  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a tower-local variable container.
  """
    # Note that synchronization "ON_READ" implies trainable=False.
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        synchronization=variable_scope.VariableSynchronization.ON_READ,
        aggregation=variable_scope.VariableAggregation.SUM,
        name=name)


def _remove_squeezable_dimensions(predictions, labels, weights):
    """Squeeze or expand last dim if needed.

  Squeezes last dim of `predictions` or `labels` if their rank differs by 1
  (using confusion_matrix.remove_squeezable_dimensions).
  Squeezes or expands last dim of `weights` if its rank differs by 1 from the
  new rank of `predictions`.

  If `weights` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Optional label `Tensor` whose dimensions match `predictions`.
    weights: Optional weight scalar or `Tensor` whose dimensions match
      `predictions`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
    the last dimension squeezed, `weights` could be extended by one dimension.
  """
    predictions = ops.convert_to_tensor(predictions)
    if labels is not None:
        labels, predictions = confusion_matrix.remove_squeezable_dimensions(
            labels, predictions)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    if weights is None:
        return predictions, labels, None

    weights = ops.convert_to_tensor(weights)
    weights_shape = weights.get_shape()
    weights_rank = weights_shape.ndims
    if weights_rank == 0:
        return predictions, labels, weights

    predictions_shape = predictions.get_shape()
    predictions_rank = predictions_shape.ndims
    if (predictions_rank is not None) and (weights_rank is not None):
        # Use static rank.
        if weights_rank - predictions_rank == 1:
            weights = array_ops.squeeze(weights, [-1])
        elif predictions_rank - weights_rank == 1:
            weights = array_ops.expand_dims(weights, [-1])
    else:
        # Use dynamic rank.
        weights_rank_tensor = array_ops.rank(weights)
        rank_diff = weights_rank_tensor - array_ops.rank(predictions)

        def _maybe_expand_weights():
            return control_flow_ops.cond(
                math_ops.equal(rank_diff, -1),
                lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

        # Don't attempt squeeze if it will fail based on static check.
        if ((weights_rank is not None) and
            (not weights_shape.dims[-1].is_compatible_with(1))):
            maybe_squeeze_weights = lambda: weights
        else:
            maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

        def _maybe_adjust_weights():
            return control_flow_ops.cond(math_ops.equal(rank_diff, 1),
                                         maybe_squeeze_weights,
                                         _maybe_expand_weights)

        # If weights are scalar, do nothing. Otherwise, try to add or remove a
        # dimension to match predictions.
        weights = control_flow_ops.cond(math_ops.equal(weights_rank_tensor, 0),
                                        lambda: weights, _maybe_adjust_weights)
    return predictions, labels, weights


def _aggregate_across_towers1(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across towers."""

    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution, '_outer_control_flow_context'):
            # If there was an outer context captured before this method was called,
            # then we enter that context to create the metric value op. If the
            # caputred context is `None`, ops.control_dependencies(None) gives the
            # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
            # captured context.
            # This special handling is needed because sometimes the metric is created
            # inside a while_loop (and perhaps a TPU rewrite context). But we don't
            # want the value op to be evaluated every step or on the TPU. So we
            # create it outside so that it can be evaluated at the end on the host,
            # once the update ops have been evaluted.

            # pylint: disable=protected-access
            if distribution._outer_control_flow_context is None:
                with ops.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            ops.add_to_collections(metrics_collections, metric_value)
        return metric_value

    #return distribution_strategy_context.get_tower_context().merge_call(fn, *args)
    return distribution_strategy_context.get_replica_context().merge_call(
        fn, *args)


def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
    """Aggregate metric value across replicas."""

    def fn(distribution, *a):
        """Call `metric_value_fn` in the correct control flow context."""
        if hasattr(distribution.extended, '_outer_control_flow_context'):
            # If there was an outer context captured before this method was called,
            # then we enter that context to create the metric value op. If the
            # caputred context is `None`, ops.control_dependencies(None) gives the
            # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
            # captured context.
            # This special handling is needed because sometimes the metric is created
            # inside a while_loop (and perhaps a TPU rewrite context). But we don't
            # want the value op to be evaluated every step or on the TPU. So we
            # create it outside so that it can be evaluated at the end on the host,
            # once the update ops have been evaluted.

            # pylint: disable=protected-access
            if distribution.extended._outer_control_flow_context is None:
                with ops.control_dependencies(None):
                    metric_value = metric_value_fn(distribution, *a)
            else:
                distribution.extended._outer_control_flow_context.Enter()
                metric_value = metric_value_fn(distribution, *a)
                distribution.extended._outer_control_flow_context.Exit()
                # pylint: enable=protected-access
        else:
            metric_value = metric_value_fn(distribution, *a)
        if metrics_collections:
            ops.add_to_collections(metrics_collections, metric_value)
        return metric_value

    return distribution_strategy_context.get_replica_context().merge_call(
        fn, args=args)


def _confusion_matrix_at_thresholds(labels,
                                    predictions,
                                    thresholds,
                                    weights=None,
                                    includes=None):
    """Computes true_positives, false_negatives, true_negatives, false_positives.

  This function creates up to four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives`.
  `true_positive[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `false_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `True`.
  `true_negatives[i]` is defined as the total weight of values in `predictions`
  at most `thresholds[i]` whose corresponding entry in `labels` is `False`.
  `false_positives[i]` is defined as the total weight of values in `predictions`
  above `thresholds[i]` whose corresponding entry in `labels` is `False`.

  For estimation of these metrics over a stream of data, for each metric the
  function respectively creates an `update_op` operation that updates the
  variable and returns its value.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    thresholds: A python list or tuple of float thresholds in `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    includes: Tuple of keys to return, from 'tp', 'fn', 'tn', fp'. If `None`,
        default to all four.

  Returns:
    values: Dict of variables of shape `[len(thresholds)]`. Keys are from
        `includes`.
    update_ops: Dict of operations that increments the `values`. Keys are from
        `includes`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      `includes` contains invalid keys.
  """
    all_includes = ('tp', 'fn', 'tn', 'fp')
    if includes is None:
        includes = all_includes
    else:
        for include in includes:
            if include not in all_includes:
                raise ValueError('Invalid key: %s.' % include)

    with ops.control_dependencies([
            check_ops.assert_greater_equal(
                predictions,
                math_ops.cast(0.0, dtype=predictions.dtype),
                message='predictions must be in [0, 1]'),
            check_ops.assert_less_equal(predictions,
                                        math_ops.cast(1.0,
                                                      dtype=predictions.dtype),
                                        message='predictions must be in [0, 1]')
    ]):
        predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=math_ops.to_float(predictions),
            labels=math_ops.cast(labels, dtype=dtypes.bool),
            weights=weights)

    num_thresholds = len(thresholds)

    # Reshape predictions and labels.
    predictions_2d = array_ops.reshape(predictions, [-1, 1])
    labels_2d = array_ops.reshape(math_ops.cast(labels, dtype=dtypes.bool),
                                  [1, -1])

    # Use static shape if known.
    num_predictions = predictions_2d.get_shape().as_list()[0]

    # Otherwise use dynamic shape.
    if num_predictions is None:
        num_predictions = array_ops.shape(predictions_2d)[0]
    thresh_tiled = array_ops.tile(
        array_ops.expand_dims(array_ops.constant(thresholds), [1]),
        array_ops.stack([1, num_predictions]))

    # Tile the predictions after thresholding them across different thresholds.
    pred_is_pos = math_ops.greater(
        array_ops.tile(array_ops.transpose(predictions_2d),
                       [num_thresholds, 1]), thresh_tiled)
    if ('fn' in includes) or ('tn' in includes):
        pred_is_neg = math_ops.logical_not(pred_is_pos)

    # Tile labels by number of thresholds
    label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
    if ('fp' in includes) or ('tn' in includes):
        label_is_neg = math_ops.logical_not(label_is_pos)

    if weights is not None:
        weights = weights_broadcast_ops.broadcast_weights(
            math_ops.to_float(weights), predictions)
        weights_tiled = array_ops.tile(array_ops.reshape(weights, [1, -1]),
                                       [num_thresholds, 1])
        thresh_tiled.get_shape().assert_is_compatible_with(
            weights_tiled.get_shape())
    else:
        weights_tiled = None

    values = {}
    update_ops = {}
    reset_ops = {}

    if 'tp' in includes:
        true_p = metric_variable([num_thresholds],
                                 dtypes.float32,
                                 name='true_positives')
        is_true_positive = math_ops.to_float(
            math_ops.logical_and(label_is_pos, pred_is_pos))
        if weights_tiled is not None:
            is_true_positive *= weights_tiled
        update_ops['tp'] = state_ops.assign_add(
            true_p, math_ops.reduce_sum(is_true_positive, 1))
        reset_ops['tp'] = state_ops.assign(
            true_p, tf.constant([0] * num_thresholds, dtype=dtypes.float32))
        values['tp'] = true_p

    if 'fn' in includes:
        false_n = metric_variable([num_thresholds],
                                  dtypes.float32,
                                  name='false_negatives')
        is_false_negative = math_ops.to_float(
            math_ops.logical_and(label_is_pos, pred_is_neg))
        if weights_tiled is not None:
            is_false_negative *= weights_tiled
        update_ops['fn'] = state_ops.assign_add(
            false_n, math_ops.reduce_sum(is_false_negative, 1))
        reset_ops['fn'] = state_ops.assign(
            false_n, tf.constant([0] * num_thresholds, dtype=dtypes.float32))
        values['fn'] = false_n

    if 'tn' in includes:
        true_n = metric_variable([num_thresholds],
                                 dtypes.float32,
                                 name='true_negatives')
        is_true_negative = math_ops.to_float(
            math_ops.logical_and(label_is_neg, pred_is_neg))
        if weights_tiled is not None:
            is_true_negative *= weights_tiled
        update_ops['tn'] = state_ops.assign_add(
            true_n, math_ops.reduce_sum(is_true_negative, 1))
        reset_ops['tn'] = state_ops.assign(
            true_n, tf.constant([0] * num_thresholds, dtype=dtypes.float32))
        values['tn'] = true_n

    if 'fp' in includes:
        false_p = metric_variable([num_thresholds],
                                  dtypes.float32,
                                  name='false_positives')
        is_false_positive = math_ops.to_float(
            math_ops.logical_and(label_is_neg, pred_is_pos))
        if weights_tiled is not None:
            is_false_positive *= weights_tiled
        update_ops['fp'] = state_ops.assign_add(
            false_p, math_ops.reduce_sum(is_false_positive, 1))
        reset_ops['fp'] = state_ops.assign(
            false_p, tf.constant([0] * num_thresholds, dtype=dtypes.float32))
        values['fp'] = false_p

    return values, update_ops, reset_ops


def _aggregate_variable(v, collections):
    f = lambda distribution, value: distribution.read_var(value)
    return _aggregate_across_towers(collections, f, v)


def auc(labels,
        predictions,
        weights=None,
        num_thresholds=10000,
        metrics_collections=None,
        updates_collections=None,
        curve='ROC',
        name=None,
        summation_method='trapezoidal'):
    """Computes the approximate AUC via a Riemann sum.

  The `auc` function creates four local variables, `true_positives`,
  `true_negatives`, `false_positives` and `false_negatives` that are used to
  compute the AUC. To discretize the AUC curve, a linearly spaced set of
  thresholds is used to compute pairs of recall and precision values. The area
  under the ROC-curve is therefore computed using the height of the recall
  values by the false positive rate, while the area under the PR-curve is the
  computed using the height of the precision values by the recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`.

  For best results, `predictions` should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case. Setting `summation_method`
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `auc`.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` whose shape matches `predictions`. Will be cast to
      `bool`.
    predictions: A floating point `Tensor` of arbitrary shape and whose values
      are in the range `[0, 1]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    num_thresholds: The number of thresholds to use when discretizing the roc
      curve.
    metrics_collections: An optional list of collections that `auc` should be
      added to.
    updates_collections: An optional list of collections that `update_op` should
      be added to.
    curve: Specifies the name of the curve to be computed, 'ROC' [default] or
      'PR' for the Precision-Recall-curve.
    name: An optional variable_scope name.
    summation_method: Specifies the Riemann summation method used
      (https://en.wikipedia.org/wiki/Riemann_sum): 'trapezoidal' [default] that
      applies the trapezoidal rule; 'careful_interpolation', a variant of it
      differing only by a more correct interpolation scheme for PR-AUC -
      interpolating (true/false) positives but not the ratio that is precision;
      'minoring' that applies left summation for increasing intervals and right
      summation for decreasing intervals; 'majoring' that does the opposite.
      Note that 'careful_interpolation' is strictly preferred to 'trapezoidal'
      (to be deprecated soon) as it applies the same method for ROC, and a
      better one (see Davis & Goadrich 2006 for details) for the PR curve.

  Returns:
    auc: A scalar `Tensor` representing the current area-under-curve.
    update_op: An operation that increments the `true_positives`,
      `true_negatives`, `false_positives` and `false_negatives` variables
      appropriately and whose value matches `auc`.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
    if context.executing_eagerly():
        raise RuntimeError(
            'tf.metrics.auc is not supported when eager execution '
            'is enabled.')

    with variable_scope.variable_scope(name, 'auc',
                                       (labels, predictions, weights)):
        if curve != 'ROC' and curve != 'PR':
            raise ValueError('curve must be either ROC or PR, %s unknown' %
                             (curve))
        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        values, update_ops, reset_ops = _confusion_matrix_at_thresholds(
            labels, predictions, thresholds, weights)

        # Add epsilons to avoid dividing by 0.
        epsilon = 1.0e-6

        def interpolate_pr_auc(tp, fp, fn):
            """Interpolation formula inspired by section 4 of Davis & Goadrich 2006.

      Note here we derive & use a closed formula not present in the paper
      - as follows:
      Modeling all of TP (true positive weight),
      FP (false positive weight) and their sum P = TP + FP (positive weight)
      as varying linearly within each interval [A, B] between successive
      thresholds, we get
        Precision = (TP_A + slope * (P - P_A)) / P
      with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).
      The area within the interval is thus (slope / total_pos_weight) times
        int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
        int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}
      where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in
        int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)
      Bringing back the factor (slope / total_pos_weight) we'd put aside, we get
         slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight
      where dTP == TP_B - TP_A.
      Note that when P_A == 0 the above calculation simplifies into
        int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)
      which is really equivalent to imputing constant precision throughout the
      first bucket having >0 true positives.

      Args:
        tp: true positive counts
        fp: false positive counts
        fn: false negative counts
      Returns:
        pr_auc: an approximation of the area under the P-R curve.
      """
            dtp = tp[:num_thresholds - 1] - tp[1:]
            p = tp + fp
            prec_slope = _safe_div(dtp, p[:num_thresholds - 1] - p[1:],
                                   'prec_slope')
            intercept = tp[1:] - math_ops.multiply(prec_slope, p[1:])
            safe_p_ratio = array_ops.where(
                math_ops.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0),
                _safe_div(p[:num_thresholds - 1], p[1:],
                          'recall_relative_ratio'), array_ops.ones_like(p[1:]))
            return math_ops.reduce_sum(_safe_div(
                prec_slope * (dtp + intercept * math_ops.log(safe_p_ratio)),
                tp[1:] + fn[1:],
                name='pr_auc_increment'),
                                       name='interpolate_pr_auc')

        def compute_auc(tp, fn, tn, fp, name):
            """Computes the roc-auc or pr-auc based on confusion counts."""
            if curve == 'PR':
                if summation_method == 'trapezoidal':
                    logging.warning(
                        'Trapezoidal rule is known to produce incorrect PR-AUCs; '
                        'please switch to "careful_interpolation" instead.')
                elif summation_method == 'careful_interpolation':
                    # This one is a bit tricky and is handled separately.
                    return interpolate_pr_auc(tp, fp, fn)
            rec = math_ops.div(tp + epsilon, tp + fn + epsilon)
            if curve == 'ROC':
                fp_rate = math_ops.div(fp, fp + tn + epsilon)
                x = fp_rate
                y = rec
            else:  # curve == 'PR'.
                prec = math_ops.div(tp + epsilon, tp + fp + epsilon)
                x = rec
                y = prec
            if summation_method in ('trapezoidal', 'careful_interpolation'):
                # Note that the case ('PR', 'careful_interpolation') has been handled
                # above.
                return math_ops.reduce_sum(math_ops.multiply(
                    x[:num_thresholds - 1] - x[1:],
                    (y[:num_thresholds - 1] + y[1:]) / 2.),
                                           name=name)
            elif summation_method == 'minoring':
                return math_ops.reduce_sum(math_ops.multiply(
                    x[:num_thresholds - 1] - x[1:],
                    math_ops.minimum(y[:num_thresholds - 1], y[1:])),
                                           name=name)
            elif summation_method == 'majoring':
                return math_ops.reduce_sum(math_ops.multiply(
                    x[:num_thresholds - 1] - x[1:],
                    math_ops.maximum(y[:num_thresholds - 1], y[1:])),
                                           name=name)
            else:
                raise ValueError('Invalid summation_method: %s' %
                                 summation_method)

        # sum up the areas of all the trapeziums
        def compute_auc_value(_, values):
            return compute_auc(values['tp'], values['fn'], values['tn'],
                               values['fp'], 'value')

        auc_value = _aggregate_across_towers(metrics_collections,
                                             compute_auc_value, values)
        update_op = compute_auc(update_ops['tp'], update_ops['fn'],
                                update_ops['tn'], update_ops['fp'], 'update_op')

        reset_op = compute_auc(reset_ops['tp'], reset_ops['fn'],
                               reset_ops['tn'], reset_ops['fp'], 'reset_op')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return auc_value, update_op, reset_op


def RiemannRunningAUCMeter(labels, predictions, name="", num_thresholds=1e6):
    with tf.variable_scope(name, 'auc', (labels, predictions)):
        p_cnt = tf.get_variable("p_cnt", [
            1000000,
        ],
                                dtype=tf.int32,
                                initializer=tf.constant_initializer(0),
                                collections=[
                                    tf.GraphKeys.LOCAL_VARIABLES,
                                    tf.GraphKeys.METRIC_VARIABLES
                                ],
                                trainable=False)

        n_cnt = tf.get_variable("n_cnt", [
            1000000,
        ],
                                dtype=tf.int32,
                                initializer=tf.constant_initializer(0),
                                collections=[
                                    tf.GraphKeys.LOCAL_VARIABLES,
                                    tf.GraphKeys.METRIC_VARIABLES
                                ],
                                trainable=False)
