from collections import deque
import gym
import numpy as np
import sys
import random
from tensorflow import keras

"""
I really don't recommend trying to engage with this code before reading the 2 notebooks in this dir.
Feel free to run it, but there isn't any code explanation in here. 

This script will render the game on your screen.
It saves a 'cartpole-dqn2.h5' model in the same dir when it's done.
"""
class DQNAgent:

    def __init__(self):
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        

    def _build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(32, activation='relu', input_dim=4))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(8, activation='relu'))
        model.add(keras.layers.Dense(4, activation='relu'))
        model.add(keras.layers.Dense(2, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model
    
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(2)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])
    
    
    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        update_state = np.zeros((batch_size, 4))
        update_next_state = np.zeros((batch_size, 4))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_state[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_next_state[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_state)
        target_next = self.model.predict(update_next_state)
        target_val = self.target_model.predict(update_next_state)

        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.gamma * (
                    target_val[i][a])

        self.model.fit(update_state, target, batch_size=batch_size,
                       epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)
            

env = gym.envs.make('CartPole-v1')
env._max_episode_steps = 1000
agent = DQNAgent()

scores = []

for episode in range(10000):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, 4])
    
    score = 0

    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        reward = reward if not done or score == 999 else -100
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            agent.update_target_model()
            scores.append(score)
            print(f'episode:{episode}, score: {score}, epsilon: {round(agent.epsilon, 5)}')
            if np.mean(scores[-min(3, len(scores)):]) > 900:
                agent.save('cartpole-dqn2.h5')
                sys.exit()
        if len(agent.memory) > 512:
            agent.replay(512)

agent.save('cartpole-dqn2.h5')
