from keras.models import load_model, save_model
import numpy as np
from ReplayBuffer import ReplayBuffer
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, TimeDistributed
from keras.optimizers import Adam
from keras.models import Sequential
from keras import activations
from keras.callbacks import Callback
from keras import backend as k
import gc


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def build_model(input_dim, n_actions, lr):
    model = Sequential([
        Conv2D(filters=6, kernel_size=7, strides=3, input_shape=input_dim),
        Activation(activations.relu),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=12, kernel_size=4),
        Activation(activations.relu),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(216),
        Activation(activations.relu),
        Dense(n_actions)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return model


class Agent:
    def __init__(self, input_dim, n_actions, lr, gamma, replace_target=100, epsilon=1.0,
                 epsilon_dec=0.999, epsilon_end=0.1):
        self.n_actions = n_actions
        self.replace_target = replace_target

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end

        self.primary = build_model(input_dim, n_actions, lr)
        self.target = build_model(input_dim, n_actions, lr)
        self.memory = ReplayBuffer(100000, input_dim)

    def store_data(self, state, action, reward, next_state, done):
        self.memory.store_data(state, action, reward, next_state, done)

    def make_action(self, state):
        state = state[np.newaxis, :]

        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            actions = self.primary.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def train(self, batch_size):
        if batch_size > self.memory.mem_cntr:
            return

        states, actions, rewards, next_states, done = \
            self.memory.sample_data(batch_size)

        primaty_next_state_values = self.primary.predict(next_states, verbose=0)
        target_next_state_values = self.target.predict(next_states, verbose=0)

        targets = self.primary.predict(states, verbose=0)

        max_actions = np.argmax(primaty_next_state_values, axis=1)
        batch_index = np.arange(batch_size, dtype=np.int32)

        targets[batch_index, actions] = rewards + self.gamma * \
            target_next_state_values[batch_index, max_actions.astype(int)]*done
        self.primary.fit(states, targets, verbose=0, callbacks=ClearMemory())

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
            self.epsilon_end else self.epsilon_end

        if self.memory.mem_cntr % self.replace_target == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target.set_weights(self.primary.get_weights())

    def save_model(self, model_file='model/ddqn_model.h5'):
        save_model(self.primary, model_file)

    def load_model(self, model_file='model/ddqn_model.h5'):
        self.primary = load_model(model_file)
        self.primary.summary()
