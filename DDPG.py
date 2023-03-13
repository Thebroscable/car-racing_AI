from keras.models import load_model, save_model
import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.activations import relu, tanh
from keras.callbacks import Callback
from keras import backend as k
import gc


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def build_actor_models(input_dim, n_actions):
    observation_input = Input(shape=input_dim)

    x = Conv2D(filters=6, kernel_size=7, strides=3, input_shape=input_dim)(observation_input)
    x = Activation(relu)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=12, kernel_size=4)(x)
    x = Activation(relu)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)

    y = Dense(216)(x)
    y = Activation(relu)(y)
    y = Dense(n_actions)(y)
    y = Activation(tanh)(y)

    actor = Model(inputs=observation_input, outputs=y)

    return actor


def build_critic_models(input_dim, n_actions):
    action_input = Input(shape=(n_actions,))
    observation_input = Input(shape=input_dim)

    x = Conv2D(filters=6, kernel_size=7, strides=3, input_shape=input_dim)(observation_input)
    x = Activation(relu)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=12, kernel_size=4)(x)
    x = Activation(relu)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)

    z = Concatenate()([action_input, x])
    z = Dense(312)(z)
    z = Activation(relu)(z)
    z = Dense(1)(z)

    critic = Model(inputs=[observation_input, action_input], outputs=z)

    return critic


class Agent:
    def __init__(self, input_dim, n_actions,
                 lr1, lr2, gamma, tau=0.005):
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma

        self.actor = \
            build_actor_models(input_dim, n_actions)
        self.critic = \
            build_critic_models(input_dim, n_actions)
        self.actor_target = \
            build_actor_models(input_dim, n_actions)
        self.critic_target = \
            build_critic_models(input_dim, n_actions)

        self.actor_opt = Adam(learning_rate=lr1)
        self.critic_opt = Adam(learning_rate=lr2)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.memory = \
            ReplayBuffer(100000, input_dim, (self.n_actions,))

    def store_data(self, state, action, reward, next_state, done):
        self.memory.store_data(state, action, reward, next_state, done)

    def make_action(self, state, evaluate=False):
        state = state[np.newaxis, :]
        actions = tf.squeeze(self.actor(state))
        actions = actions.numpy()
        if not evaluate:
            actions += np.random.normal(scale=0.4,
                                        size=self.n_actions)
        actions = [np.clip(actions[0], -1, 1),
                   np.clip(actions[1], 0, 1),
                   np.clip(actions[2], 0, 1)]
        return np.squeeze(actions)

    def update_targets(self):
        update_target(self.actor_target.get_weights(),
                      self.actor.get_weights(), self.tau)
        update_target(self.critic_target.get_weights(),
                      self.critic.get_weights(), self.tau)

    @tf.function
    def train(self, batch_size):
        if batch_size > self.memory.mem_cntr:
            batch_size = self.memory.mem_cntr

        states, actions, rewards, next_states, done = \
            self.memory.sample_data(batch_size)

        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)

        with tf.GradientTape() as tape:
            target_actions = \
                self.actor_target(next_states, training=True)
            target_next_state_values = \
                self.critic_target([next_states,
                                    target_actions],
                                   training=True)
            y = rewards + self.gamma * \
                target_next_state_values * done
            critic_value = \
                self.critic([states, actions], training=True)
            critic_loss = tf.keras.losses.MSE(y, critic_value)

        critic_grads = \
            tape.gradient(critic_loss,
                          self.critic.trainable_variables)
        self.critic_opt.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_value = \
                self.critic([states, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grads = \
            tape.gradient(actor_loss,
                          self.actor.trainable_variables)
        self.actor_opt.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))

    def save_model(self, actor_file='ddpg_actor.h5'):
        save_model(self.actor, actor_file)

    def load_model(self, actor_file='ddpg_actor.h5'):
        self.actor = load_model(actor_file)


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
