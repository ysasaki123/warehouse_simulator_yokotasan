from collections import defaultdict
from random import choice, random
import pickle

from sklearn.svm import SVR
import numpy as np
import pandas as pd

class SVRPlayer:
    def __init__(self):
        self.reg = SVR()
        self.reg.fit(np.random.random((10, 4)), np.random.random(10))
        self.experience = []
        self.epsilon = 0.5
        self.num_orders = 0

    def select_action(self, obs, train=True):
        possible_actions1 = []
        possible_actions2 = []
        features1 = []
        features2 = []

        # 注文できるのは最大で5個まで
        if self.num_orders > 10:
            self.num_orders = 0
            return (None, None, 0)

        for w in obs.warehouses:
            for product_id in w.product_repo:
                action = (w.id, product_id, 20)
                possible_actions1.append(action)
                features1.append(obs.get_feature(w.id, product_id) + [0])
                possible_actions2.append((w.id, product_id, 0))
                features2.append(obs.get_feature(w.id, product_id) + [1])

        values1 = self.reg.predict(features1)
        values2 = self.reg.predict(features2)

        remaining_orders = [
            (a, v1)
            for a, v1, v2 in zip(possible_actions1, values1, values2)
            if v1 > v2
        ]
        if not train:
            tmp = pd.DataFrame(features1, columns=['order', 'stock', 'waiting', 'finish']).drop('finish', axis=1)
            tmp['values1'] = values1
            tmp['values2'] = values2
            tmp['flag'] = [v1 < v2 for v1, v2 in zip(values1, values2)]
            tmp = pd.concat([
                tmp,
                pd.DataFrame(possible_actions1, columns=['wid', 'pid', 'amount']).drop('amount', axis=1)
            ], axis=1)
            tmp = tmp[['wid', 'pid', 'values1', 'values2', 'flag', 'order', 'stock', 'waiting']]
            print('data\n', tmp.sort_values(['values1']).to_csv(index=False, sep='\t'))

        # (None, None, 0) は注文終了の行動を意味する
        if train and random() < self.epsilon:
            if random() < 0.5 and self.num_orders != 0:
                return (None, None, 0)
            else:
                self.num_orders += 1
                return choice(possible_actions1)

        if not remaining_orders:
            self.num_orders = 0
            return (None, None, 0)

        result = sorted(remaining_orders, key=lambda x: x[1])[-1][0]
        self.num_orders += 1
        return result

    def create_data(self):
        experiences = self.experience
        prev_state = np.array([s1 + [int(a[2] == 0)] for s1,a,r,s2 in experiences])
        next_state = np.array([s2 + [0] for s1,a,r,s2 in experiences])
        next_state2 = np.array([s2 + [1] for s1,a,r,s2 in experiences])
        prev_q = self.reg.predict(prev_state)
        next_q = self.reg.predict(next_state)
        finish_q = self.reg.predict(next_state2)
        max_next_q = np.max([next_q, finish_q], axis=0)
        reward = np.array([r for s1,a,r,s2 in experiences])

        gamma = 0.7
        alpha = 0.9
        q = (1-gamma) * prev_q + gamma * (reward + alpha * max_next_q)
        # print(f'train {prev_state.shape}\n', np.concatenate((
        #     prev_state,
        #     next_state,
        #     # next_state2,
        #     np.expand_dims(reward, axis=1),
        #     np.expand_dims(next_q, axis=1),
        #     np.expand_dims(finish_q, axis=1),
        #     np.expand_dims(q, axis=1),
        #     ), axis=1).astype(int)[-20:,:])
        return prev_state, q

    def add_experience(self, state1, action, reward, state2):
        key = tuple(state1 + [int(action[2] == 0)])
        self.experience.append((state1, action, reward, state2))

    def train(self):
        x, y = self.create_data()
        if len(self.experience) > 500:
            self.reg.fit(x, y)
            self.experience = [self.experience[i] for i in self.reg.support_]

    def __str__(self):
        from itertools import product
        product_target = [
            range(0, 100, 20),
            range(0, 100, 20),
            range(0, 100, 20),
            [0,1],
        ]
        features = [feature for feature in product(*product_target)]
        df = pd.DataFrame(features, columns=['order', 'stock', 'waiting', 'finish'])
        values = df.values
        df['result'] = self.reg.predict(values)

        return (df
            .sort_values(['order', 'stock', 'waiting', 'finish'])
            .set_index(['order', 'stock', 'waiting', 'finish'])
            .unstack('finish')
            .reset_index()
            .to_csv(sep='\t', index=False)
        )
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.reg, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.reg = pickle.load(f)

