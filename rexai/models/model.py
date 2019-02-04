import tensorflow as tf
from tefla.core.layers import conv2d, fully_connected as fc, max_pool, prelu, register_to_collections
from tefla.core import initializers as initz
from tefla.core.layer_arg_ops import common_layer_args, make_args, end_points
from tefla.core.layers import dropout


def model(height, width, num_actions, is_training=False, reuse=None, name=None):
  common_args = common_layer_args(is_training, reuse)
  conv_args = make_args(
      batch_norm=True,
      activation=prelu,
      w_init=initz.he_normal(scale=1),
      untie_biases=False,
      **common_args)
  logits_args = make_args(activation=None, w_init=initz.he_normal(scale=1), **common_args)
  fc_args = make_args(activation=prelu, w_init=initz.he_normal(scale=1), **common_args)
  pool_args = make_args(padding='SAME', **common_args)
  with tf.variable_scope(name):
    state = register_to_collections(
        tf.placeholder(shape=[None, 4, height, width], dtype=tf.float32, name='state'),
        name='state',
        **common_args)
    state_perm = tf.transpose(state, perm=[0, 2, 3, 1])
    summary_ops = [
        tf.summary.image(
            "states", state[:, 0, :, :][..., tf.newaxis], max_outputs=10, collections='train')
    ]
    conv1_0 = conv2d(state_perm, 32, filter_size=8, stride=(1, 1), name="conv1_0", **conv_args)
    conv1_1 = conv2d(conv1_0, 64, filter_size=8, stride=(2, 2), name="conv1_1", **conv_args)
    pool = max_pool(conv1_1, filter_size=2, name="maxpool", **pool_args)
    conv2_0 = conv2d(pool, 128, filter_size=4, stride=2, name="conv2_0", **conv_args)
    conv2_1 = conv2d(conv2_0, 256, filter_size=3, stride=(2, 2), name="conv2_1", **conv_args)
    conv3_0 = conv2d(conv2_1, 256, filter_size=4, stride=1, name="conv3_0", **conv_args)
    conv3_1 = conv2d(conv3_0, 512, filter_size=4, stride=2, name="conv3_1", **conv_args)
    # Dueling
    value_hid = fc(conv3_1, 512, name="value_hid", **fc_args)
    adv_hid = fc(conv3_1, 512, name="adv_hid", **fc_args)

    value = fc(value_hid, 1, name="value", **logits_args)
    advantage = fc(adv_hid, num_actions, name="advantage", **logits_args)

    # Average Dueling
    Qs = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))

    # action with highest Q values
    a = register_to_collections(tf.argmax(Qs, 1), name='a', **common_args)
    # Q value belonging to selected action
    Q = register_to_collections(tf.reduce_max(Qs, 1), name='Q', **common_args)
    summary_ops.append(tf.summary.histogram("Q", Q, collections='train'))
    # For training
    Q_target = register_to_collections(
        tf.placeholder(shape=[None], dtype=tf.float32), name='Q_target', **common_args)
    actions = register_to_collections(
        tf.placeholder(shape=[None], dtype=tf.int32), name='actions', **common_args)
    actions_onehot = tf.one_hot(
        actions, num_actions, on_value=1., off_value=0., axis=1, dtype=tf.float32)

    Q_tmp = tf.reduce_sum(tf.multiply(Qs, actions_onehot), axis=1)
    loss = register_to_collections(
        tf.reduce_mean(tf.square(Q_target - Q_tmp)), name='loss', **common_args)
    summary_ops.append(tf.summary.scalar("loss", loss, collections='train'))
    register_to_collections(summary_ops, name='summary_ops', **common_args)
    return end_points(is_training)
