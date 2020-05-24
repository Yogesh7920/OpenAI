import gym
from agent import Agent
from collections import deque
import numpy as np


class Interact:

    def __init__(self, s='CartPole-v0'):
        self.env = gym.make(s)
        self.agent = Agent()
        self.scores = []

    def train(self, episodes=20, patience=5):
        pat = deque([], maxlen=patience)
        for epi in range(1, episodes+1):
            score = 0
            state = self.env.reset()
            done = False
            ct = 1
            while not done:
                print('{}.{}'.format(epi, ct), end=' ')
                ct += 1
                action = self.agent.action(state)
                new_state, reward, done, _ = self.env.step(action)

                self.agent.remember(state, action, reward, new_state, done)

                self.agent.train()
                self.agent.target_train()
                state = new_state
                score += reward
                print()
            self.agent.epsilon = max(self.agent.epsilon * self.agent.epsilon_decay, self.agent.epsilon_min)
            self.scores.append(score)
            pat.append(score)
            if np.std(pat) < 1 and len(pat) == patience:
                return
            if epi % 2 == 0:
                self.agent.save_model()

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
