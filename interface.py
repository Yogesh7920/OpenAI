import gym
from agent import Agent
from collections import deque
import numpy as np


class Interact:

    def __init__(self, s='CartPole-v0'):
        self.env = gym.make(s)
        self.agent = Agent()
        self.scores = []
        self.avg_scores = deque([], maxlen=100)
        self.freq_target_train = 1

    def train(self, episodes=20):
        for epi in range(1, episodes+1):
            score = 0
            state = self.env.reset()
            done = False
            ct = 1
            rb = [0, 0]
            while not done:
                ct += 1
                action, ind = self.agent.action(state)
                rb[ind] += 1
                new_state, reward, done, _ = self.env.step(action)

                self.agent.remember(state, action, reward, new_state, done)

                state = new_state
                score += reward
            self.agent.train()
            if epi % self.freq_target_train:
                self.agent.target_train()
            self.scores.append(score)
            self.avg_scores.append(score)
            avg = 0
            if epi % 100 == 0:
                avg = np.mean(self.avg_scores)
                self.agent.save_model()
                print('EPISODE = {}, AVG = {}'.format(epi, avg))
            if avg >= 195:
                break

    def observe(self, episodes=5):
        total_score = []
        self.agent.load_model()
        for epi in range(1, episodes+1):
            score = 0
            state = self.env.reset()
            done = False
            ct = 1
            while not done:
                print('{}.{}'.format(epi, ct), end=' ')
                ct += 1
                self.env.render()
                action = self.agent.action(state)
                new_state, reward, done, _ = self.env.step(action)
                state = new_state
                score += reward
                print()

            total_score.append(score)
            self.env.close()

        return total_score
