import pickle
from random import random, choice

import numpy as np
import pandas as pd
from termcolor import colored

class SimpleMemoryPlayer:
    def __init__(self):
        self.experience = []
        self.epsilon = 0.5
        self.num_orders = 0
        self.memory = {}

    def predict(self, state_actions):
        return np.array([
            self.memory.get(tuple(sa), 0)
            for sa in state_actions
        ])

    def select_action(self, obs, train=True):
        possible_actions1 = []
        possible_actions2 = []
        features1 = []
        features2 = []

        # 注文できるのは最大で10個まで
        if self.num_orders > 10:
            self.num_orders = 0
            return [(None, None, 0)]

        for w in obs.warehouses:
            for product_id in w.product_repo:
                action = (w.id, product_id, 20)
                possible_actions1.append(action)
                features1.append(obs.get_feature(w.id, product_id) + [0])
                possible_actions2.append((w.id, product_id, 0))
                features2.append(obs.get_feature(w.id, product_id) + [1])

        values1 = self.predict(features1)
        values2 = self.predict(features2)

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
        # tmp = pd.DataFrame(features1, columns=['order', 'stock', 'waiting', 'finish']).drop('finish', axis=1)
        # tmp['values1'] = values1
        # tmp['values2'] = values2
        # tmp['flag'] = [v1 < v2 for v1, v2 in zip(values1, values2)]
        # tmp = pd.concat([
        #     tmp,
        #     pd.DataFrame(possible_actions1, columns=['wid', 'pid', 'amount']).drop('amount', axis=1)
        # ], axis=1)
        # tmp = tmp[['wid', 'pid', 'values1', 'values2', 'flag', 'order', 'stock', 'waiting']]
        # # print('data\n', tmp.sort_values(['values1']).to_csv(index=False, sep='\t'))

        # (None, None, 0) は注文終了の行動を意味する
        if train and random() < self.epsilon:
            if random() < 0.5 and self.num_orders != 0:
                return [(None, None, 0)]
            else:
                self.num_orders += 1
                return [choice(possible_actions1) for _ in range(3)]

        if not remaining_orders:
            self.num_orders = 0
            return [(None, None, 0)]

        self.num_orders += 1
        return [k for k, v in remaining_orders]

    def add_experience(self, state1, action, reward, state2):
        self.experience.append((state1, action, reward, state2))
        if len(self.experience) > 1000:
            self.experience.pop(0)

    def create_data(self):
        prev_state = np.array([s1 + [int(a[2] == 0)] for s1,a,r,s2 in self.experience])
        next_state = np.array([s2 + [0] for s1,a,r,s2 in self.experience])
        next_state2 = np.array([s2 + [1] for s1,a,r,s2 in self.experience])
        prev_q = self.predict(prev_state)
        next_q = self.predict(next_state)
        finish_q = self.predict(next_state2)
        max_next_q = np.max([next_q, finish_q], axis=0)
        reward = np.array([r for s1,a,r,s2 in self.experience])

        gamma = 0.7
        alpha = 0.9
        q = (1-gamma) * prev_q + gamma * (reward + alpha * max_next_q)
        # print('train\n', np.concatenate((
        #     prev_state,
        #     next_state,
        #     next_state2,
        #     np.expand_dims(reward, axis=1),
        #     np.expand_dims(next_q, axis=1),
        #     np.expand_dims(finish_q, axis=1),
        #     np.expand_dims(q, axis=1),
        #     ), axis=1).astype(int)[-20:,:])
        return prev_state, q

    def train(self):
        X, y = self.create_data()
        for feature, value in zip(X, y):
            self.memory[tuple(feature)] = value

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    # def __str__(self):
    #     return (pd.DataFrame([
    #         [*k, int(v)]
    #         for k, v in self.memory.items()
    #     ], columns=['order', 'stock', 'waiting', 'finish', 'value'])
    #     .sort_values(['order', 'stock', 'waiting', 'finish'])
    #     # .set_index(['order', 'stock', 'waiting', 'finish'])
    #     # .unstack('finish')
    #     .to_csv(sep='\t', index=False)
    #     )

    def __str__(self):
        from itertools import product
        product_target = [
            range(0, 100, 20),
            range(0, 120, 30),
            range(0, 100, 20),
            [0,1],
        ]
        features = [feature for feature in product(*product_target)]
        df = pd.DataFrame(features, columns=['order', 'stock', 'waiting', 'finish'])
        df['result'] = df.apply(lambda x: self.memory.get(tuple(x), 0), axis=1)
        # df['exp'] = [[e[2] for e in self.experience.get(tuple(key), [])] for key in values]

        tmp = (df
            .sort_values(['order', 'stock', 'waiting', 'finish'])
            .set_index(['order', 'stock', 'waiting', 'finish'])
            .unstack('finish')
            .reset_index()
        )
        tmp['flag'] = (tmp[('result', 0)] > tmp[('result', 1)]).map({True: colored('注文', 'red'), False: colored('完了', 'green')})
        return tmp.to_csv(sep='\t', index=False)