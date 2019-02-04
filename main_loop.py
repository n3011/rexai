import os
from functools import partial
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from tefla.da.preprocessor import RexPreprocessor
from tefla.da.standardizer import SamplewiseStandardizer

from rexai.models.model import model
from rexai.core.agent import DDQNAgent
from rexai.env.environment import Environment
from rexai.utils.common_utils import summarize

# Application flags
tf.app.flags.DEFINE_string("logdir", "./logs/",
                           "Path to store the model and tensorboard logs or restore the model")
tf.app.flags.DEFINE_integer("refresh_rex", 100, "Reloading the browser every x epochs")
tf.app.flags.DEFINE_integer("update_target_network_rex", 20, "Reloading the browser every x epochs")
tf.app.flags.DEFINE_boolean("training", True, "Train a new model")
tf.app.flags.DEFINE_boolean("visualize", True, "Visualize")
FLAGS = tf.app.flags.FLAGS


def main(_):

  if FLAGS.training:
    try:
      os.makedirs(FLAGS.logdir)
    except Exception:
      pass

  with tf.Session() as sess:
    writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph) if FLAGS.visualize else None
    summary_scalars = ["exploration", "ep_steps", "ep_reward"]
    env = Environment("127.0.0.1", 9090)
    standardizer = SamplewiseStandardizer(6)
    preprocessor = RexPreprocessor(env.screen_width, env.screen_height, standardizer)
    agent = DDQNAgent(
        env, model, sess, FLAGS.logdir, is_training=FLAGS.training, reuse=None, writer=writer)

    if FLAGS.training:
      summarize_func = partial(summarize, summary_scalars, sess, writer)
      agent.train(preprocessor, summarize_func, FLAGS.refresh_rex, FLAGS.update_target_network_rex)
    else:
      agent.play(preprocessor, FLAGS.logdir)


if __name__ == '__main__':
  tf.app.run()
