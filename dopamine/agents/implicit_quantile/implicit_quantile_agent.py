# 1 coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The implicit quantile networks (IQN) agent.

The agent follows the description given in "Implicit Quantile Networks for
Distributional RL" (Dabney et. al, 2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import atari_lib
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class ImplicitQuantileAgent(rainbow_agent.RainbowAgent):
    """An extension of Rainbow to perform implicit quantile regression."""

    def __init__(self,
                 sess,
                 num_actions,
                 network=atari_lib.implicit_quantile_network,
                 kappa=1.0,
                 dueltype=None,
                 num_tau_samples=32,
                 num_tau_prime_samples=32,
                 num_quantile_samples=32,
                 quantile_embedding_dim=64,
                 double_dqn=False,
                 summary_writer=None,
                 summary_writing_frequency=500):
        """Initializes the agent and constructs the Graph.

        Most of this constructor's parameters are IQN-specific hyperparameters whose
        values are taken from Dabney et al. (2018).

        Args:
          sess: `tf.Session` object for running associated ops.
          num_actions: int, number of actions the agent can take at any state.
          network: function expecting three parameters:
            (num_actions, network_type, state). This function will return the
            network_type object containing the tensors output by the network.
            See dopamine.discrete_domains.atari_lib.nature_dqn_network as
            an example.
          kappa: float, Huber loss cutoff.
          num_tau_samples: int, number of online quantile samples for loss
            estimation.
          num_tau_prime_samples: int, number of target quantile samples for loss
            estimation.
          num_quantile_samples: int, number of quantile samples for computing
            Q-values.
          quantile_embedding_dim: int, embedding dimension for the quantile input.
          double_dqn: boolean, whether to perform double DQN style learning
            as described in Van Hasselt et al.: https://arxiv.org/abs/1509.06461.
          summary_writer: SummaryWriter object for outputting training statistics.
            Summary writing disabled if set to None.
          summary_writing_frequency: int, frequency with which summaries will be
            written. Lower values will result in slower training.
        """
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        self._dueltype = dueltype
        print(self._dueltype)
        if len(dueltype.split('-')) > 1:
            self.optimal_factor = float(dueltype.split('-')[1]) * 0.01
        else:
            self.optimal_factor = 0
        self.kappa = kappa
        print('optimal factor:', self.optimal_factor)
        # num_tau_samples = N below equation (3) in the paper.
        self.num_tau_samples = num_tau_samples
        # num_tau_prime_samples = N' below equation (3) in the paper.
        self.num_tau_prime_samples = num_tau_prime_samples
        # num_quantile_samples = k below equation (3) in the paper.
        self.num_quantile_samples = num_quantile_samples
        if 'optimal' not in dueltype:
            self.num_tau_samples = 64
            self.num_tau_prime_samples = 64
            self.num_quantile_samples = 32
        else:
            if len(dueltype.split('-')) > 2:
                tau = int(dueltype.split('-')[2])
                self.num_tau_samples = tau
                self.num_tau_prime_samples = tau
                self.num_quantile_samples = tau
                print('tau:', tau)
        # quantile_embedding_dim = n above equation (4) in the paper.
        self.quantile_embedding_dim = quantile_embedding_dim
        # option to perform double dqn.
        self.double_dqn = double_dqn
        self.optimizer1 = tf.train.RMSPropOptimizer(
            learning_rate=0.00025 * self.optimal_factor,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)

        super(ImplicitQuantileAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            network=network,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)

    def _get_network_type(self):
        """Returns the type of the outputs of the implicit quantile network.

        Returns:
          _network_type object defining the outputs of the network.
        """
        return collections.namedtuple(
            'iqn_network',
            ['quantile_values', 'quantiles', 'quantile_values_origin', 'quantiles_origin', 'Fv_diff', 'v_diff',
             'quantile_values_mid', 'quantiles_mid', 'L_tau'])

    def _network_template(self, state, num_quantiles):
        r"""Builds an Implicit Quantile ConvNet.

        Takes state and quantile as inputs and outputs state-action quantile values.

        Args:
          state: A `tf.placeholder` for the RL state.
          num_quantiles: int, number of quantile inputs.

        Returns:
          _network_type object containing quantile value outputs of the network.
        """
        return self.network(self.num_actions, self.quantile_embedding_dim,
                            self._get_network_type(), state, num_quantiles, self._dueltype)

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                if 'optimal' in self._dueltype:
                    _, _, loss, loss1, quan_value, quan, vdiff = self._sess.run(self._train_op)
                elif 'iqn' in self._dueltype:
                    _, loss, quan_value, quan, vdiff = self._sess.run(self._train_op)
                    loss1 = None
                if self.training_steps % 50000 == 0:
                    print(">>> loss:", loss)
                    print(">>> loss1:", loss1)
                    print(">>> factor:", self.optimal_factor)
                    batchsize = 32
                    quan_value = np.reshape(quan_value, [batchsize, self.num_tau_samples])
                    quan = np.reshape(quan, [batchsize, self.num_tau_samples])
                    quan_value = quan_value[0].tolist()
                    quan = quan[0].tolist()
                    vdiff = vdiff[:, 0].tolist()
                    print(">>> value:", quan_value)
                    print(">>> quans:", quan)
                    print(">>> vdiff:", vdiff)
                    print(">>> vdiff_sum:", np.sum(vdiff))
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps)

            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.training_steps += 1

    def _build_networks(self):
        """Builds the IQN computations needed for acting and training.

        These are:
          self.online_convnet: For computing the current state's quantile values.
          self.target_convnet: For computing the next state's target quantile
            values.
          self._net_outputs: The actual quantile values.
          self._q_argmax: The action maximizing the current state's Q-values.
          self._replay_net_outputs: The replayed states' quantile values.
          self._replay_next_target_net_outputs: The replayed next states' target
            quantile values.
        """
        # Calling online_convnet will generate a new graph as defined in
        # self._get_network_template using whatever input is passed, but will always
        # share the same weights.
        self.online_convnet = tf.make_template('Online', self._network_template)
        self.target_convnet = tf.make_template('Target', self._network_template)

        # Compute the Q-values which are used for action selection in the current
        # state.
        self._net_outputs = self.online_convnet(self.state_ph,
                                                self.num_quantile_samples)
        # Shape of self._net_outputs.quantile_values:
        # num_quantile_samples x num_actions.
        # e.g. if num_actions is 2, it might look something like this:
        # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
        #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
        # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
        if 'ws' in self._dueltype:
            self._q_values = tf.reduce_sum(self._net_outputs.quantile_values * self._net_outputs.v_diff,
                                           axis=0)  # NOTE: quantile_values = quantile_values_mid
        else:
            self._q_values = tf.reduce_mean(self._net_outputs.quantile_values, axis=0)
        self._q_argmax = tf.argmax(self._q_values, axis=0)

        self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                       self.num_tau_samples)
        # Shape: (num_tau_samples x batch_size) x num_actions.
        self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
        self._replay_net_quantiles = self._replay_net_outputs.quantiles

        # Do the same for next states in the replay buffer.
        self._replay_net_target_outputs = self.target_convnet(
            self._replay.next_states, self.num_tau_prime_samples)
        # Shape: (num_tau_prime_samples x batch_size) x num_actions.
        vals = self._replay_net_target_outputs.quantile_values
        self._replay_net_target_quantile_values = vals

        # Compute Q-values which are used for action selection for the next states
        # in the replay buffer. Compute the argmax over the Q-values.
        if self.double_dqn:
            outputs_action = self.online_convnet(self._replay.next_states,
                                                 self.num_quantile_samples)
        else:
            outputs_action = self.target_convnet(self._replay.next_states,
                                                 self.num_quantile_samples)

        # Shape: (num_quantile_samples x batch_size) x num_actions.
        target_quantile_values_action = outputs_action.quantile_values  # NOTE: quantile_values = quantile_values_mid
        # Shape: num_quantile_samples x batch_size x num_actions.
        target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                                   [self.num_quantile_samples,
                                                    self._replay.batch_size,
                                                    self.num_actions])
        # Shape: batch_size x num_actions.
        if 'ws' in self._dueltype:
            v_diff = tf.reshape(outputs_action.v_diff, [self.num_quantile_samples, self._replay.batch_size, 1])
            self._replay_net_target_q_values = tf.squeeze(tf.reduce_sum(
                target_quantile_values_action * v_diff, axis=0))
        else:
            self._replay_net_target_q_values = tf.squeeze(tf.reduce_mean(
                target_quantile_values_action, axis=0))
        self._replay_next_qt_argmax = tf.argmax(
            self._replay_net_target_q_values, axis=1)

    def _build_target_quantile_values_op(self):
        """Build an op used as a target for return values at given quantiles.

        Returns:
          An op calculating the target quantile return.
        """
        batch_size = tf.shape(self._replay.rewards)[0]
        # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
        rewards = self._replay.rewards[:, None]
        rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

        is_terminal_multiplier = 1. - tf.to_float(self._replay.terminals)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                      [self.num_tau_prime_samples, 1])

        # Get the indices of the maximium Q-value across the action dimension.
        # Shape of replay_next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.

        replay_next_qt_argmax = tf.tile(
            self._replay_next_qt_argmax[:, None], [self.num_tau_prime_samples, 1])

        # Shape of batch_indices: (num_tau_prime_samples x batch_size) x 1.
        batch_indices = tf.cast(tf.range(
            self.num_tau_prime_samples * batch_size)[:, None], tf.int64)

        # Shape of batch_indexed_target_values:
        # (num_tau_prime_samples x batch_size) x 2.
        batch_indexed_target_values = tf.concat(
            [batch_indices, replay_next_qt_argmax], axis=1)

        # Shape of next_target_values: (num_tau_prime_samples x batch_size) x 1.
        target_quantile_values = tf.gather_nd(
            self._replay_net_target_quantile_values,
            batch_indexed_target_values)[:, None]

        return rewards + gamma_with_terminal * target_quantile_values

    def _build_train_op(self):
        """Builds a training op.

        Returns:
          train_op: An op performing one step of training from replay data.
        """
        batch_size = tf.shape(self._replay.rewards)[0]

        target_quantile_values = tf.stop_gradient(
            self._build_target_quantile_values_op())
        # Reshape to self.num_tau_prime_samples x batch_size x 1 since this is
        # the manner in which the target_quantile_values are tiled.
        target_quantile_values = tf.reshape(target_quantile_values,
                                            [self.num_tau_prime_samples,
                                             batch_size, 1])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_prime_samples x 1 to prepare for computation of
        # Bellman errors.
        # Final shape of target_quantile_values:
        # batch_size x num_tau_prime_samples x 1.
        target_quantile_values = tf.transpose(target_quantile_values, [1, 0, 2])

        # Shape of indices: (num_tau_samples x batch_size) x 1.
        # Expand dimension by one so that it can be used to index into all the
        # quantiles when using the tf.gather_nd function (see below).
        indices = tf.range(self.num_tau_samples * batch_size)[:, None]

        # Expand the dimension by one so that it can be used to index into all the
        # quantiles when using the tf.gather_nd function (see below).
        reshaped_actions = self._replay.actions[:, None]
        reshaped_actions = tf.tile(reshaped_actions, [self.num_tau_samples, 1])
        # Shape of reshaped_actions: (num_tau_samples x batch_size) x 2.
        reshaped_actions = tf.concat([indices, reshaped_actions], axis=1)

        chosen_action_quantile_values = tf.gather_nd(
            self._replay_net_quantile_values, reshaped_actions)
        # Reshape to self.num_tau_samples x batch_size x 1 since this is the manner
        # in which the quantile values are tiled.
        chosen_action_quantile_values = tf.reshape(chosen_action_quantile_values,
                                                   [self.num_tau_samples,
                                                    batch_size, 1])
        # Transpose dimensions so that the dimensionality is batch_size x
        # self.num_tau_samples x 1 to prepare for computation of
        # Bellman errors.
        # Final shape of chosen_action_quantile_values:
        # batch_size x num_tau_samples x 1.
        chosen_action_quantile_values = tf.transpose(
            chosen_action_quantile_values, [1, 0, 2])  # batchsize x quan x 1

        # Shape of bellman_erors and huber_loss:
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        bellman_errors = target_quantile_values[:, :, None, :] - chosen_action_quantile_values[:, None, :, :]
        if 'optimal12' in self._dueltype and 'fixbugtarg' in self._dueltype:
            print("============================================================= fixbug")
            print(bellman_errors.shape, self._replay_net_outputs.v_diff.shape, self.num_tau_samples)
            bellman_errors = bellman_errors * self._replay_net_outputs.v_diff[:, :, None, None] * self.num_tau_samples
        # The huber loss (see Section 2.3 of the paper) is defined via two cases:
        # case_one: |bellman_errors| <= kappa
        # case_two: |bellman_errors| > kappa
        huber_loss_case_one = tf.to_float(
            tf.abs(bellman_errors) <= self.kappa) * 0.5 * bellman_errors ** 2
        huber_loss_case_two = tf.to_float(
            tf.abs(bellman_errors) > self.kappa) * self.kappa * (
                                      tf.abs(bellman_errors) - 0.5 * self.kappa)
        huber_loss = huber_loss_case_one + huber_loss_case_two

        # Reshape replay_quantiles to batch_size x num_tau_samples x 1
        replay_quantiles = tf.reshape(
            self._replay_net_quantiles, [self.num_tau_samples, batch_size, 1])
        replay_quantiles = tf.transpose(replay_quantiles, [1, 0, 2])  # batchsize x quan x 1

        # Tile by num_tau_prime_samples along a new dimension. Shape is now
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        # These quantiles will be used for computation of the quantile huber loss
        # below (see section 2.3 of the paper).
        replay_quantiles = tf.to_float(tf.tile(
            replay_quantiles[:, None, :, :], [1, self.num_tau_prime_samples, 1, 1]))
        # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
        quantile_huber_loss = (tf.abs(tf.stop_gradient(replay_quantiles) - tf.stop_gradient(
            tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa
        # Sum over current quantile value (num_tau_samples) dimension,
        # average over target quantile value (num_tau_prime_samples) dimension.
        # Shape: batch_size x num_tau_prime_samples x 1.
        loss = tf.reduce_sum(quantile_huber_loss, axis=2)
        # Shape: batch_size x 1.
        loss = tf.reduce_mean(loss, axis=1)

        if 'optimal1' in self._dueltype:
            chosen_action_L_tau = tf.gather_nd(self._replay_net_outputs.L_tau, reshaped_actions)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", chosen_action_L_tau.shape)
            loss1 = tf.reduce_mean(chosen_action_L_tau, axis=0)
            print(loss1.shape)
        elif 'optimal' in self._dueltype:
            print(reshaped_actions.shape)
            chosen_action_quantile_values_origin = tf.gather_nd(
                self._replay_net_outputs.quantile_values_origin, reshaped_actions)
            print(chosen_action_quantile_values_origin.shape)
            grad = tf.gradients([chosen_action_quantile_values_origin], [self._replay_net_outputs.quantiles_origin])[0]
            grad = tf.reshape(grad, [self.num_quantile_samples + 1, batch_size])[:-1, :]
            grad = tf.reshape(grad, [self.num_quantile_samples, batch_size])
            div = self._replay_net_outputs.Fv_diff / tf.expand_dims(self._replay_net_outputs.v_diff, 2)
            div = tf.reshape(div, [batch_size * self.num_quantile_samples, self.num_actions])
            div = tf.gather_nd(div, reshaped_actions)
            div = tf.reshape(div, [self.num_quantile_samples, batch_size])
            print("grad ==>> ", grad)
            print(div)
            # Shape: batch_size x 1.
            loss1 = tf.expand_dims(tf.reduce_mean(grad - div, axis=0), 1)
            print("loss1:", loss1.shape)
            print("loss:", loss.shape)

        # TODO(kumasaurabh): Add prioritized replay functionality here.
        update_priorities_op = tf.no_op()
        with tf.control_dependencies([update_priorities_op]):
            if self.summary_writer is not None:
                with tf.variable_scope('Losses'):
                    tf.summary.scalar('QuantileLoss', tf.reduce_mean(loss))
            if 'optimal' in self._dueltype:
                iqn_params, optimal_params = [], []
                params = tf.trainable_variables()
                for p in params:
                    if 'optimal' in p.name:
                        optimal_params.append(p)
                    else:
                        iqn_params.append(p)
                print("optimal_params:>>>>>>", optimal_params)
                print("iqn_params:>>>>>>", iqn_params)
                # batchsize x quan
                # batchsize x quan
                # quan x batchsize
                return self.optimizer.minimize(tf.reduce_mean(loss), var_list=iqn_params), self.optimizer1.minimize(
                    tf.reduce_mean(loss1), var_list=optimal_params), \
                       tf.reduce_mean(loss), tf.reduce_mean(loss1), \
                       tf.squeeze(chosen_action_quantile_values), \
                       tf.squeeze(replay_quantiles[:, 0, :, :]), \
                       self._replay_net_outputs.v_diff
            else:
                print("---->>>", replay_quantiles.shape)
                return self.optimizer.minimize(tf.reduce_mean(loss)), tf.reduce_mean(loss), \
                       tf.squeeze(chosen_action_quantile_values), tf.squeeze(
                    replay_quantiles[:, 0, :, :]), self._replay_net_outputs.v_diff
