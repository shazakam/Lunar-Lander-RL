import keras
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from replay_buffer import ReplayBuffer
import keras_tuner as kt
import numpy as np


class DDQN(object):
    def __init__(
        self,
        alpha,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        epsilon_decrement=0.996,
        epsilon_end=0.01,
        memory_size=1000000,
        f_name="ddqn_model",
        replace_target=100,
        fc1_dims=256,
        fc2_dims=256,
        verbose=0,
    ):
        self.alpha = alpha
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decrement = epsilon_decrement
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.model = f_name
        self.replace_target = replace_target
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.memory = ReplayBuffer(memory_size, input_dims, n_actions, True)
        self.q_eval = self._get_compiled_model()
        self.q_target = self._get_compiled_model()
        self.verbose = verbose

    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

    def choose_action(self, state):
        # handle batch training and single memory feedforward
        state = state[np.newaxis, :]
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def learn(self):
        # avoid filling up agents memory with random actions
        if self.memory.memory_counter > self.batch_size:
            state, action, new_state, reward, done = self.memory.sample_buffer(
                self.batch_size
            )
            # go back from one hot encoding to regular encoding
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state, verbose=self.verbose)
            q_eval = self.q_eval.predict(new_state, verbose=self.verbose)

            q_pred = self.q_eval.predict(state, verbose=self.verbose)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = (
                reward
                + self.gamma * q_next[batch_index, max_actions.astype(int)] * done
            )

            performance_metrics_history = self.q_eval.fit(
                state, q_target, verbose=self.verbose
            )

            self.epsilon = (
                self.epsilon * self.epsilon_decrement
                if self.epsilon > self.epsilon_min
                else self.epsilon_min
            )

            if self.memory.memory_counter % self.replace_target == 0:
                self._update_network_params()

            return performance_metrics_history

    def _update_network_params(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model)

    def load_model(self):
        self.q_eval = load_model(self.model)

        if self.epsilon <= self.epsilon_min:
            self._update_network_params()

    def _get_compiled_model(
        self, units1=None, units2=None, activation="relu", loss="mse", lr=None
    ):
        model = Sequential()
        model.add(
            Dense(
                units=self.fc1_dims if units1 == None else units1,
                input_shape=(self.input_dims,),
                activation=activation,
            )
        )
        model.add(
            Dense(
                units=self.fc2_dims if units1 == None else units2, activation=activation
            )
        )
        model.add(Dense(self.n_actions))
        model.compile(
            loss=loss,
            optimizer=Adam(learning_rate=self.alpha if lr == None else lr),
        )
        return model

    def build_model_tuning(self, hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        activation = hp.Choice("activation", ["relu", "tanh"])
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        # call existing model-building code with the hyperparameter values.
        model = self._get_comiled_model(
            units=units, activation=activation, loss="mse", lr=lr
        )
        return model
