from collections import defaultdict
from random import choice, random, sample, randint
import pickle

import numpy as np
import pandas as pd
from termcolor import colored

import keras
from keras import Sequential, regularizers, Input, Model
from keras.layers import Embedding, Dense, Flatten, Dropout, add, Concatenate, MaxPooling1D, Reshape, Conv1D, Lambda, Activation
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


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
        # rewards = [-1*amount for amount in actions]
        rewards = [0 for amount in actions]
        total_amount = sum(actions)
        done = total_amount == 0
        info = {}
        return self.features, rewards, done, info
   
def create_model():
    model = Sequential()

    model.add(Embedding(6, 1, input_length=6))
    model.add(Flatten())
    model.add(Dense(64, input_dim=6, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01)))
    model.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
    print(model.summary())
    input('after confirmation, press enter')
    return model

def create_model2():
    model = Model()

    inputs = Input(shape=(7,))
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01))(inputs)
    x = Reshape((32, 1))(x)
    x = Conv1D(32, (1,))(x)
    x = Conv1D(32, (1,))(x)
    x = MaxPooling1D()(x)
    x = Conv1D(16, (1,))(x)
    x = Conv1D(16, (1,))(x)
    x = MaxPooling1D()(x)
    x = Conv1D(16, (1,))(x)
    x = Conv1D(16, (1,))(x)
    x = Flatten()(x)

    # y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    # y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    # y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    # x = add([x, y])
    # y = Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    # y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    # y = Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    # x = add([x, y])

    y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    y = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    y = Dense(1, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)

    x = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    x = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    x = Dense(1, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)

    x = Concatenate()([x, y])
    x = Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01))(x)
    model = Model(inputs, x)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    print(model.summary())
    input('After confirm model, Press Enter')
    return model

def create_model2():
    model = Model()

    inputs = Input(shape=(6,))
    x = Embedding(10000, 2, input_length=6)(inputs)
    x = Flatten()(x)
    y = Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01))(inputs)
    x = Concatenate()([x, y])

    y = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    y = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)
    y = Dense(1, activation='relu', kernel_regularizer=regularizers.l1(0.01))(y)

    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)
    x = Dense(1, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x)

    x = Concatenate()([x, y])
    x = Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01))(x)
    model = Model(inputs, x)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    print(model.summary())
    input('After confirm model, Press Enter')
    return model

def residual_layer(x, size, prefix):
    y = Dense(size, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-1')(x)
    y = Dense(size, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-2')(y)
    y = Dense(size, kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-3')(y)
    x = add([x, y], name=f'{prefix}-add')
    return Activation('relu')(x)

def create_model2():
    model = Model()

    inputs = Input(shape=(7,))
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='1')(inputs)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='2')(x)
    residual_size = 4
    x = Dense(residual_size, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='3')(x)
    for i in range(4, 30):
        x = residual_layer(x, residual_size, i)
    x = Dense(8, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='31')(x)
    x = Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01), name='32')(x)
    model = Model(inputs, x)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    print(model.summary())
    input('after confirmation, press enter')
    return model


def divide(x, unit, prefix=1):
    return Concatenate(name=f'{prefix}-cc')([
        Dense(unit, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-1')(x),
        Dense(unit, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-2')(x),
        Dense(unit, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-3')(x),
        Dense(unit, activation='relu', kernel_regularizer=regularizers.l1(0.01), name=f'{prefix}-4')(x),
    ])
    

# def create_model2():
#     model = Model()
# 
#     inputs = Input(shape=(7,))
#     x = inputs
#     
#     x = divide(x, 4, 1)
#     x = divide(x, 4, 2)
#     x = divide(x, 4, 3)
#     x = Reshape((4, 4))(x)
#     x = MaxPooling1D()(x)
#     x = Flatten()(x)
#     x = divide(x, 4, 4)
#     x = divide(x, 4, 5)
#     x = divide(x, 4, 6)
#     x = Reshape((4, 4))(x)
#     x = MaxPooling1D()(x)
#     x = Flatten()(x)
#     x = divide(x, 4, 7)
#     x = divide(x, 4, 8)
#     x = divide(x, 4, 9)
#     x = Reshape((4, 4))(x)
#     x = MaxPooling1D()(x)
#     x = Flatten()(x)
#     x = Dense(4, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='10')(x)
#     x = Dense(4, activation='relu', kernel_regularizer=regularizers.l1(0.01), name='11')(x)
#     x = Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01), name='12')(x)
#     model = Model(inputs, x)
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#     # print(model.summary())
#     return model

# def create_model2():
#     model = Model()
# 
#     inputs = Input(shape=(7,))
#     x = inputs
#     
#     x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l1(0.01), name='1')(x)
#     x = Dense(64, activation='tanh', kernel_regularizer=regularizers.l1(0.01), name='2')(x)
#     x = Dense(1, activation='linear', kernel_regularizer=regularizers.l1(0.01), name='3')(x)
#     model = Model(inputs, x)
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#     # print(model.summary())
#     return model

class DLMultiPlayers:
    def __init__(self):
        with tf.device('/device:GPU:0'):
            self.target = create_model2()
        with tf.device('/device:GPU:1'):
            self.main = create_model2()
        self.target.fit(np.random.random((10, 7)), np.random.random(10))
        self.main.fit(np.random.random((10, 7)), np.random.random(10))
        self.experience = defaultdict(list)
        self.epsilon = 0.3
        self.num_orders = 0
        self.num_train = 0

        self.small_experience = defaultdict(list)
        self.last_state = {}

    def with_epsilon(self, epsilon):
        self.epsilon = epsilon

    def inner_select_actions(self, inner_obs, train, finish_flag):
        features = inner_obs
        order_values = self.main.predict(np.array([feature + [0, 1] for feature in features]))
        finish_values = self.main.predict(np.array([feature + [1, 0] for feature in features]))

        tmp = pd.DataFrame(features)
        tmp.columns = ['受注', '在庫', '発注', 'day', 'lead']
        tmp['expect'] = tmp.day * tmp.lead
        tmp.loc[:,'v1'] = order_values
        tmp.loc[:,'v2'] = finish_values
        tmp.loc[:,'result'] = (tmp['v1'] > tmp['v2']).map({True: colored('注文', 'red'), False: colored('完了', 'green')})
        tmp.loc[:,'fihish_flag'] = pd.Series(finish_flag).map({True: colored('完了', 'green'), False: colored('未完了', 'red')})
        # tmp = tmp.sort_values(['day', 'lead'])
        tmp = tmp.sort_index()
        print(tmp.to_csv(sep='\t'))

        if train and random() < self.epsilon:
            if random() < 0.2:
                return [0] * len(features)
            else:
                return [20 if not flag else 0 for flag in finish_flag]

        # 20ずつ注文を増やしていく
        return [
            20 if not flag and order_value > finish_value else 0
            for order_value, finish_value, flag in zip(order_values, finish_values, finish_flag)
        ]

    def select_action(self, obs, train=True):
        items = []
        features = []

        for w in obs.warehouses:
            for product_id in w.product_repo:
                items.append((w.id, product_id))
                features.append(obs.get_feature(w.id, product_id))

        # if train and random() < self.epsilon:
        #     if random() < 0.5:
        #         return [(w, p, 0) for w, p in items]
        #     else:
        #         return [(w, p, 20) for w, p in items]

        inner_obs = features

        env = SmallEnv(inner_obs)
        done = False
        total_amounts = [0] * len(items)
        finish_flag = [False] * len(items)
        self.small_experience = defaultdict(list)
        for i in range(100):
            actions = self.inner_select_actions(inner_obs, train, finish_flag)
            new_inner_obs, rewards, done, info = env.step(actions)

            
            for action, is_finished, prev_feature, amount, reward, feature in zip(items, finish_flag, inner_obs, actions, rewards, new_inner_obs):
                # 完了の経験は、最後にupdate_experienceで追加する
                if amount != 0 and not is_finished:
                    self.small_experience[action].append((prev_feature, amount, reward, feature))
        
            inner_obs = new_inner_obs

            for i, amount in enumerate(actions):
                if amount == 0:
                    finish_flag[i] = True
            for i, amount in enumerate(actions):
                if finish_flag[i]:
                    continue
                total_amounts[i] += amount
            if done:
                print(total_amounts)
                break
            else:
                print('done')
        # print('----inner loop finished----')

        self.last_state = {
            action: last_state
            for action, last_state in zip(items, inner_obs)
        } 

        return [(w, p, a) for (w, p), a in zip(items, total_amounts)]
    
    def update_experience(self, last_reward_dict, next_obs):
        for (wid, pid) in self.small_experience:
            self.experience[(wid, pid)] += self.small_experience[(wid, pid)]
            while len(self.experience[(wid, pid)]) > 50000:
                print('------------')
                self.experience[(wid, pid)].pop(randint(0, len(self.experience[(wid, pid)]) - 1))

        for (wid, pid), last_obs in self.last_state.items():
            last_reward = last_reward_dict[(wid, pid)]
            last_exp = (last_obs, 0, last_reward, next_obs.get_feature(wid, pid))
            self.experience[(wid, pid)].append(last_exp)

            while len(self.experience[(wid, pid)]) > 50000:
                print('------------')
                self.experience[(wid, pid)].pop(randint(0, len(self.experience[(wid, pid)]) - 1))

    def create_data(self):
        experiences = sum(self.experience.values(), [])
        num_sample = min(100000, len(experiences))
        print('num_exp', len(experiences))
        experiences = sample(experiences, num_sample)

        prev_state = np.array([s1 + [int(a == 0), int(a > 0)] for s1,a,r,s2 in experiences])
        next_state = np.array([s2 + [0, 1] for s1,a,r,s2 in experiences])
        next_state2 = np.array([s2 + [1, 0] for s1,a,r,s2 in experiences])
        prev_q = self.target.predict(prev_state)[:, 0]
        next_q = self.target.predict(next_state)[:, 0]
        finish_q = self.target.predict(next_state2)[:, 0]
        max_next_q = np.mean([next_q, finish_q], axis=0)
        reward = np.array([r for s1,a,r,s2 in experiences])

        self.num_train += 1
        gamma = 0.99
        alpha = 1 / self.num_train ** 0.2
        q = (1-alpha) * prev_q + alpha * (reward + gamma * max_next_q)

        tmp = pd.DataFrame([
            s1[:3] + s2[:3] + [a, r]
            for s1, a, r, s2 in experiences
        ])
        tmp['not_finish'] = next_q
        tmp['finish'] = finish_q
        tmp['q'] = q
        tmp = tmp.sort_values('q')
        # tmp = tmp[tmp[7] < 100]
        print('order by 0 and q desc head')
        print(tmp.sort_values([0, 'q'], ascending=False).head(20).to_csv(sep='\t'))
        print('order by q head')
        print(tmp.head(20).to_csv(sep='\t'))
        print('order by q tail')
        print(tmp.tail(20).to_csv(sep='\t'))
        print('order by 7 and q asc tail')
        print(tmp.sort_values([7, 'q'], ascending=True).tail(20).to_csv(sep='\t'))
        print('order by finish_q head')
        print(tmp.sort_values(['finish'], ascending=True).head(20).to_csv(sep='\t'))
        print('order by finish_q tail')
        print(tmp.sort_values(['finish'], ascending=True).tail(20).to_csv(sep='\t'))

        # if tmp[7].max() > 0:
        #     import pdb; pdb.set_trace()
        
        return prev_state, q

    def add_experience(self, state1, action, reward, state2):
        key = tuple(state1[:3] + [int(action[2] == 0)])
        self.experience[key].append((state1, action, reward, state2))
        if len(self.experience[key]) > 4:
            self.experience[key].pop(0)

    def copy_to_target(self):
        self.target.set_weights(self.main.get_weights())

    def train(self):
        if len(self.experience) == 0:
            return
        x, y = self.create_data()
        if x.shape[0] > 5:
            for _ in range(10):
                self.main.fit(x, y, epochs=20, callbacks=[
                    EarlyStopping(monitor='loss', min_delta=1, mode='min'),
                    TensorBoard(log_dir='./tflog/', histogram_freq=1)
                ])

            print('after')
            x, y = self.create_data()

    def __str__(self):
        from itertools import product
        product_target = [
            range(0, 100, 20),
            range(0, 120, 30),
            range(0, 100, 30),
            [0,1],
            [0,1],
            [1],
            [10],
        ]
        features = [feature for feature in product(*product_target)]
        df = pd.DataFrame(features, columns=['order', 'stock', 'waiting', 'finish', 'ufinish', 'daily_order', 'lead_time'])
        values = df.values
        df['result'] = self.main.predict(values)
        # df['exp'] = [[e[2] for e in self.experience.get(tuple(key), [])] for key in values]

        tmp = (df
            .sort_values(['order', 'stock', 'waiting', 'daily_order', 'lead_time', 'finish', 'ufinish'])
            .set_index(['order', 'stock', 'waiting', 'daily_order', 'lead_time', 'finish', 'ufinish'])
            .unstack('finish')
            .reset_index()
        )
        tmp['flag'] = (tmp[('result', 0)] > tmp[('result', 1)]).map({True: colored('注文', 'red'), False: colored('完了', 'green')})
        return tmp.head(100).to_csv(sep='\t', index=False)
    
    def save(self, path):
        self.target.save(path + '.h5')

    def load(self, path):
        from keras.models import load_model
        self.main = load_model(path + '.h5')
        self.target = load_model(path + '.h5')
