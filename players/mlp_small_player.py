from collections import defaultdict
from random import choice, random, sample, randint
import pickle

from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from termcolor import colored

class SmallEnv:
    def __init__(self, inner_obs):
        self.features = inner_obs

    def step(self, actions):
        '''
        actionは注文量（0, 20, 100)
        obsは(在庫, 受注量, 発注量, 平均的な受注量, 平均的な納入時間)
        '''
        self.features = [
            [a,b,c + amount,d,e]
            for (a,b,c,d,e), amount in zip(self.features, actions)
        ]
        rewards = [-1*amount for amount in actions]
        total_amount = sum(actions)
        done = total_amount == 0
        info = {}
        return self.features, rewards, done, info
    

class MLPMultiPlayers:
    def __init__(self, hidden_layer_sizes=None):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (4,4,4,4,4,4,4)
        self.reg = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=10)
        self.reg.fit(np.random.random((10, 6)), np.random.random(10))
        self.experience = defaultdict(list)
        self.epsilon = 0.5
        self.num_orders = 0

        self.small_experience = defaultdict(list)
        self.last_state = {}

    def inner_select_actions(self, inner_obs, train=True):
        features = inner_obs
        order_values = self.reg.predict([feature + [0] for feature in features])
        finish_values = self.reg.predict([feature + [1] for feature in features])

        tmp = pd.DataFrame(features)
        tmp.columns = ['受注', '在庫', '発注', 'day', 'lead']
        tmp['expect'] = tmp.day * tmp.lead
        tmp.loc[:,'v1'] = order_values
        tmp.loc[:,'v2'] = finish_values
        tmp.loc[:,'result'] = (tmp['v1'] > tmp['v2']).map({True: colored('注文', 'red'), False: colored('完了', 'green')})
        tmp = tmp.sort_values(['day', 'lead'])
        print(tmp.to_csv(sep='\t'))

        if train and random() < self.epsilon:
            if random() < 0.5:
                return [0] * len(features)
            else:
                return [20] * len(features)

        # 20ずつ注文を増やしていく
        return [
            20 if order_value > finish_value else 0
            for order_value, finish_value in zip(order_values, finish_values)
        ]

    def select_action(self, obs, train=True):
        possible_actions = []
        features = []

        for w in obs.warehouses:
            for product_id in w.product_repo:
                possible_actions.append((w.id, product_id))
                features.append(obs.get_feature(w.id, product_id))

        # if train and random() < self.epsilon:
        #     if random() < 0.5:
        #         return [(w, p, 0) for w, p in possible_actions]
        #     else:
        #         return [(w, p, 20) for w, p in possible_actions]

        inner_obs = features

        env = SmallEnv(inner_obs)
        done = False
        total_amounts = [0] * len(possible_actions)
        for _ in range(100):
            actions = self.inner_select_actions(inner_obs, train)
            new_inner_obs, rewards, done, info = env.step(actions)

            self.small_experience = defaultdict(list)
            for action, prev_feature, amount, reward, feature in zip(possible_actions, inner_obs, actions, rewards, new_inner_obs):
                # 完了の経験は、最後にupdate_experienceで追加する
                if amount != 0:
                    self.small_experience[action].append((prev_feature, amount, reward, feature))
        
            inner_obs = new_inner_obs

            for i, amount in enumerate(actions):
                total_amounts[i] += amount
            if done:
                print(total_amounts)
                break
            else:
                print('done')
        # print('----inner loop finished----')

        self.last_state = {
            action: last_state
            for action, last_state in zip(possible_actions, inner_obs)
        } 

        return [(w, p, a) for (w, p), a in zip(possible_actions, total_amounts)]
    
    def update_experience(self, last_reward_dict, next_obs):
        for (wid, pid) in self.small_experience:
            self.experience[(wid, pid)] += self.small_experience[(wid, pid)]
            while len(self.experience[(wid, pid)]) > 5000:
                self.experience[(wid, pid)].pop(randint(0, len(self.experience[(wid, pid)]) - 1))

        for (wid, pid), last_obs in self.last_state.items():
            last_reward = last_reward_dict[(wid, pid)]
            last_exp = (last_obs, 0, last_reward, next_obs.get_feature(wid, pid))
            self.experience[(wid, pid)].append(last_exp)

            while len(self.experience[(wid, pid)]) > 5000:
                self.experience[(wid, pid)].pop(randint(0, len(self.experience[(wid, pid)]) - 1))

    def create_data(self):
        experiences = sum(self.experience.values(), [])
        num_sample = min(50000, len(experiences))
        print('num_exp', len(experiences))
        experiences = sample(experiences, num_sample)

        prev_state = np.array([s1 + [int(a == 0)] for s1,a,r,s2 in experiences])
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

        tmp = pd.DataFrame([
            s1 + [a, r]
            for s1, a, r, s2 in experiences
        ])
        tmp['q'] = q
        
        # print(tmp.to_csv(sep='\t'))
        return prev_state, q

    def add_experience(self, state1, action, reward, state2):
        key = tuple(state1[:3] + [int(action[2] == 0)])
        self.experience[key].append((state1, action, reward, state2))
        if len(self.experience[key]) > 4:
            self.experience[key].pop(0)

    def train(self):
        if len(self.experience) == 0:
            return
        x, y = self.create_data()
        if x.shape[0] > 5:
            self.reg.fit(x, y)

    def __str__(self):
        from itertools import product
        product_target = [
            range(0, 100, 20),
            range(0, 120, 30),
            range(0, 100, 30),
            [0,1],
            [1],
            [10],
        ]
        features = [feature for feature in product(*product_target)]
        df = pd.DataFrame(features, columns=['order', 'stock', 'waiting', 'finish', 'daily_order', 'lead_time'])
        values = df.values
        df['result'] = self.reg.predict(values)
        # df['exp'] = [[e[2] for e in self.experience.get(tuple(key), [])] for key in values]

        tmp = (df
            .sort_values(['order', 'stock', 'waiting', 'daily_order', 'lead_time', 'finish'])
            .set_index(['order', 'stock', 'waiting', 'daily_order', 'lead_time', 'finish'])
            .unstack('finish')
            .reset_index()
        )
        tmp['flag'] = (tmp[('result', 0)] > tmp[('result', 1)]).map({True: colored('注文', 'red'), False: colored('完了', 'green')})
        return tmp.head(100).to_csv(sep='\t', index=False)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.reg, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.reg = pickle.load(f)
            print(self.reg)
            input()

