from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import json


class Agent:

    def __init__(self, input_shape=4, action_space=2):
        self.max_memory_len = 5000
        self.state_memory = deque([], maxlen=self.max_memory_len)
        self.action_memory = deque([], maxlen=self.max_memory_len)
        self.reward_memory = deque([], maxlen=self.max_memory_len)
        self.next_state_memory = deque([], maxlen=self.max_memory_len)
        self.terminal_memory = deque([], maxlen=self.max_memory_len)
        self.memory_len = 0
        self.epsilon = 1
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.gamma = 0.9
        self.alpha = 0.05
        self.batch_size = 100
        self.batch_index = np.arange(self.batch_size)
        self.memory_index = np.arange(self.max_memory_len)
        self.tau = 0.1

        self.model = self.create_model(input_shape, action_space)
        self.target = self.create_model(input_shape, action_space)

    def create_model(self, input_shape, actions):
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=[input_shape]))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(actions, activation='linear'))

        model.compile(optimizer=Adam(lr=self.alpha), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, terminal):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.next_state_memory.append(next_state)
        self.terminal_memory.append(terminal)
        self.memory_len += 1
        self.memory_len = min(self.memory_len, self.max_memory_len)

    def sample(self):
        return np.random.choice(self.memory_index[:self.memory_len], self.batch_size, replace=False)

    def deque2numpy(self):
        return np.array(self.state_memory), np.array(self.action_memory), np.array(self.reward_memory), \
            np.array(self.next_state_memory), np.array(self.terminal_memory)

    def train(self):
        if self.memory_len < self.batch_size:
            print('\tMemorising', end='')
            return

        samples = self.sample()
        state_mem, action_mem, reward_mem, next_state_mem, terminal_mem = self.deque2numpy()
        states = state_mem[samples]
        actions = action_mem[samples]
        rewards = reward_mem[samples]
        next_states = next_state_mem[samples]
        terminals = terminal_mem[samples]

        q_eval = self.model.predict(states)
        q_next = self.target.predict(next_states)
        q_tar = q_eval.copy()
        q_tar[self.batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * np.bitwise_not(terminals)
        self.model.fit(states, q_tar, verbose=0, epochs=1)
        # ct = 1
        # for sample in samples:
        #     state, action, reward, next_state, done = sample
        #     state, next_state = state.reshape(1, -1), next_state.reshape(1, -1)
        #
        #     q_eval = self.model.predict(state)
        #     q_next = self.target.predict(next_state)
        #     q_tar = q_eval.copy()
        #
        #     q_tar[0][action] = reward + self.gamma*np.max(q_next, axis=1) * done
        #     print('\tTraining {}'.format(ct), end='\t')
        #     self.model.fit(state, q_tar, verbose=0, epochs=1)
        #     ct += 1

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target.set_weights(target_weights)

    def action(self, state, observe=False):
        if observe:
            a = self.model.predict(state.reshape(1, -1))
            return np.argmax(a, axis=1)[0]

        if np.random.random() < self.epsilon:
            print('\tR', end='\t')
            return np.random.choice([0, 1])
        else:
            a = self.model.predict(state.reshape(1, -1))
            print('\tB', end='\t')
            return np.argmax(a, axis=1)[0]

    def save_model(self):
        self.model.save('best.h5', overwrite=True)
        d = dict()
        d['epsilon'] = self.epsilon
        with open('config.json', 'w') as f:
            json.dump(d, f, indent=4)

    def load_model(self):
        self.model = load_model('best.h5')
        self.target = load_model('best.h5')
        with open('config.json', 'r') as f:
            d = json.load(f)

        self.epsilon = d['epsilon']
