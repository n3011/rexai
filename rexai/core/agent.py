""" Implements the deep q-learning agent"""
import random
from functools import reduce
import numpy as np
import numpy.random as rnd
import tensorflow as tf

from .memory import Memory

log=tf.logging


class DDQNAgent(object):

  def __init__(self, env, model, session, logdir, is_training=False, reuse=None, writer=None, optimizer_name='adam', learning_rate=0.001):
    self.path_checkpoints = logdir
    self.session = session
    self.env = env
    self.num_actions = len(env.actions)
    self.memory_size = 10000
    self.explore_prob = 1.
    self.explore_min = 0.01
    self.explore_decay = 0.997
    self.batch_size = 32
    self.discount = .95
    self.optimizer_name = optimizer_name
    self.learning_rate = learning_rate
    self.height = env.screen_height
    self.width = env.screen_width
    self.writer = writer
    self.memory = Memory(self.memory_size)
    self.main_dqn_ep = model(
        self.height, self.width, self.num_actions, name="main", is_training=is_training, reuse=reuse)
    self.source_vars = [var for var in tf.trainable_variables() if var.name.startswith('main')]
    self.target_dqn_ep = model(
        self.height,
        self.width,
        self.num_actions,
        name="target",
        is_training=is_training,
        reuse=reuse)
    self.target_vars = [var for var in tf.trainable_variables() if var.name.startswith('target')]
    self.session.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

    self.update_target_network()
    self.saver = tf.train.Saver()

  def get_action_and_q(self, ep, states):
    """
        Args:
            ep: a `dict`, with keys as tensor name and values as tf placeholder.
            states: a `np.ndarray`, states.

        Retuns:
            array[0]: actions: is a array of length len(state) with the action with the highest score
            array[1]: q value: is a array of length len(state) with the Q-value belonging to the action
        """
    states = states.reshape(-1, 4, self.height, self.width)
    return self.session.run([ep['a'], ep['Q']], {ep['state']: states})

  def get_action(self, ep, states):
    """
        Args:
            ep: a `dict`, with keys as tensor name and values as tf placeholder.
            states: a `np.ndarray`, states.

        Retuns:
            - if states contains only a single state then we return the optimal action as an integer,
            - if states contains an array of states then we return the optimal action for each state of the array
        """
    states = states.reshape(-1, 4, self.height, self.width)
    num_states = states.shape[0]
    actions = self.session.run(ep['a'], {ep['state']: states})
    return actions[0] if num_states == 1 else actions

  def _optimizer(self,
                 lr,
                 optname='momentum',
                 decay=0.9,
                 momentum=0.9,
                 epsilon=1e-08,
                 beta1=0.5,
                 beta2=0.999,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 accum_val=0.1,
                 lr_power=-0.5):
    """definew the optimizer to use.
    Args:
        lr: learning rate, a scalar or a policy
        optname: optimizer name
        decay: variable decay value, scalar
        momentum: momentum value, scalar
    Returns:
        optimizer to use
    """
    if optname == 'adadelta':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.AdadeltaOptimizer(
          learning_rate=lr, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
    if optname == 'adagrad':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.AdagradOptimizer(
          lr, initial_accumulator_value=0.1, use_locking=False, name='Adadelta')
    if optname == 'rmsprop':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.0, epsilon=epsilon)
    if optname == 'momentum':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.MomentumOptimizer(
          lr, momentum, use_locking=False, name='momentum', use_nesterov=True)
    if optname == 'adam':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.AdamOptimizer(
          learning_rate=lr,
          beta1=beta1,
          beta2=beta2,
          epsilon=epsilon,
          use_locking=False,
          name='Adam')
    if optname == 'proximalgd':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.ProximalGradientDescentOptimizer(
          lr,
          l1_regularization_strength=l1_reg,
          l2_regularization_strength=l2_reg,
          use_locking=False,
          name='ProximalGradientDescent')
    if optname == 'proximaladagrad':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.ProximalAdagradOptimizer(
          lr,
          initial_accumulator_value=accum_val,
          l1_regularization_strength=l1_reg,
          l2_regularization_strength=l2_reg,
          use_locking=False,
          name='ProximalGradientDescent')
    if optname == 'ftrl':
      log.info('Using {} optimizer'.format(optname))
      opt = tf.train.FtrlOptimizer(
          lr,
          learning_rate_power=lr_power,
          initial_accumulator_value=accum_val,
          l1_regularization_strength=l1_reg,
          l2_regularization_strength=l2_reg,
          use_locking=False,
          name='Ftrl')
    return opt

  def create_train_op(self):
    optimizer = self._optimizer(self.learning_rate, optname=self.optimizer_name)
    grads_and_vars = optimizer.compute_gradients(self.main_dqn_ep['loss'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='main')
    grads_and_vars = self.__clip_grad_norms(grads_and_vars)
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.apply_gradients(grads_and_vars)

  def __clip_grad_norms(self, gradients_to_variables, max_norm=10):
    """Clips the gradients by the given value.

        Args:
            gradients_to_variables: A list of gradient to variable pairs (tuples).
            max_norm: the maximum norm value.

        Returns:
            A list of clipped gradient to variable pairs.
        """
    grads_and_vars = []
    for grad, var in gradients_to_variables:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          tmp = tf.clip_by_norm(grad.values, max_norm)
          grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
        else:
          grad = tf.clip_by_norm(grad, max_norm)
      grads_and_vars.append((grad, var))
    return grads_and_vars

  def __train(self, ep, states, actions, targets, cnt):
    states = states.reshape(-1, 4, self.height, self.width)
    feed_dict = {ep['state']: states, ep['actions']: actions, ep['Q_target']: targets}
    summary, _ = self.session.run([tf.summary.merge(ep['summary_ops']), self.train_op], feed_dict)
    if self.writer:
      self.writer.add_summary(summary, global_step=cnt)

  def tranfer_variables_from(self, source, target):
    """
            Builds the operations required to transfer the values of the variables
            from other to self
        """
    ops = []
    for var_source, var_target in zip(source, target):
      ops.append(var_source.assign(var_target.value()))

    self.session.run(ops)

  def act(self, state):
    """
        Args:
            states: a `np.ndarray`, states.

        Retuns: an action and a boolean.
        The returned boolean: - False: action generated by the DQN
                              - True: random action (exploration)
        """
    if self.explore_prob > 0 and rnd.rand() <= self.explore_prob:
      # explore
      return rnd.randint(self.num_actions), True

    return self.get_action(self.main_dqn_ep, state), False

  def remember(self, state, action, reward, state_next, crashed):
    self.memory.remember(state, action, reward, state_next, crashed)

  def replay(self, cnt):
    if self.memory.current_size < self.batch_size:
      return

    print("...Training...")
    states, actions, rewards, states_next, crashes = self.memory.sample(self.batch_size)
    target = rewards
    # add Q value of next state to not terminal states (i.e. not crashed)
    target[~crashes] += self.discount * \
        self.get_action_and_q(self.target_dqn_ep, states_next[~crashes])[1]
    self.__train(self.main_dqn_ep, states, actions, target, cnt)

  def explore_less(self):
    self.explore_prob = max(self.explore_min, self.explore_prob * self.explore_decay)

  def update_target_network(self):
    self.tranfer_variables_from(self.source_vars, self.target_vars)

  def save(self, cnt):
    save_path = self.saver.save(self.session, self.path_checkpoints + "rex.ckpt", global_step=cnt)
    print("Model saved in file: %s" % save_path)

  def load(self, checkpoint_name):
    self.saver.restore(self.session, checkpoint_name)
    print("Model restored:", checkpoint_name)

  def play(self, preprocessor, logdir):
    name = logdir + "rex.ckpt"
    self.load(name)
    self.explore_prob = 0.0

    while True:
      frame, _, crashed = self.env.start_game()
      frame = preprocessor.preprocess_for_eval(frame)
      state = preprocessor.get_initial_state(frame)

      while not crashed:
        action, _ = self.act(state)
        next_frame, reward, crashed = self.env.do_action(action)
        print("action: {}".format(self.env.actions[action]))
        next_frame = preprocessor.preprocess_for_eval(next_frame)
        next_state = preprocessor.get_updated_state(next_frame)

        state = next_state

      print("Crash")

  def train(self, preprocessor, summarize_function, refresh_rex, update_target_network_rex):
    self.create_train_op()
    self.session.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
    latest_checkpoint = tf.train.latest_checkpoint(self.path_checkpoints)
    if latest_checkpoint is not None:
      self.load(latest_checkpoint)
    self.update_target_network()

    epoch = 0
    while True:
      epoch += 1
      print("\nEpoch: ", epoch)
      print(self.env)
      frame, _, crashed = self.env.start_game()
      frame = preprocessor.preprocess_for_train(frame)
      state = preprocessor.get_initial_state(frame)
      ep_steps, ep_reward = 0, 0

      while not crashed:

        action, explored = self.act(state)
        next_frame, reward, crashed = self.env.do_action(action)
        # A '*' is appended to the action if it was randomly chosen (i.e. not produced by the network)
        action_str = self.env.actions[action] + ["", "*"][explored]
        print("action: {}\t crashed: {}".format(action_str, crashed))
        next_frame = preprocessor.preprocess_for_train(next_frame)
        next_state = preprocessor.get_updated_state(next_frame)
        self.remember(state, action, reward, next_state, crashed)

        ep_steps += 1
        ep_reward += reward

        state = next_state

      self.replay(epoch)
      self.explore_less()

      if epoch % update_target_network_rex == 0:
        self.update_target_network()

      stats = {"exploration": self.explore_prob, "ep_steps": ep_steps, "ep_reward": ep_reward}
      summarize_function(epoch, stats)

      if epoch % 5 == 0:
        self.save(epoch)

      if epoch % refresh_rex == 0:
        self.env.refresh_game()
