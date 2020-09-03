import os
from uuid import uuid4
from collections import defaultdict, Counter
from random import choice, random
import pickle

import pandas as pd
import numpy as np

from core.entities import Factory, Warehouse, Customer, Product, Order, ProviderRepository
np.set_printoptions(precision=3)
np.set_printoptions(threshold=10000)


class State:
    def __init__(self, factories, warehouses, customers):
        self.factories = factories
        self.warehouses = warehouses
        self.customers = customers

    def get_feature(self, warehouse_id, product_id):
        if warehouse_id is None and product_id is None:
            return [-1, -1, -1, -1, -1]

        for w in self.warehouses:
            if w.id != warehouse_id:
                continue

            product = w.product_repo[product_id]
            total_order_amount = sum([o.amount for o in w.stacking_order[product_id]])
            total_waiting_order_amount = sum([o.amount for o in w.waiting_order.values() if o.product_id == product_id])
            deadlines = [o.deadline for o in w.waiting_order.values() if o.product_id == product_id]
            deadlines = [o.amount for o in w.waiting_order.values() if o.product_id == product_id and o.deadline < 5]
            mean_deadlines = sum(deadlines) if deadlines else 0
            num_stock = w.stock[product_id]
            return [total_order_amount, num_stock, total_waiting_order_amount, product.daily_order, product.lead_time_days]

        raise Exception('warehouse not found')

    def get_features(self):
        result = []
        for w in self.warehouses:
            total_waiting_order_amount = sum([o.amount for o in w.waiting_order.values()])
            for product_id in w.stock:
                product = w.product_repo[product_id]
                total_order_amount = sum([o.amount for o in w.stacking_order[product_id]])
                num_stock = w.stock[product_id]
                result.append([total_order_amount, num_stock, total_waiting_order_amount, product.daily_order, product.lead_time_days])

        return result

    def get_keys(self):
        result = []
        for w in self.warehouses:
            for product_id in w.stock:
                result.append((w.id, product_id))

        return result


class Env:
    def __init__(self, num_customer=1, num_warehouse=1, num_provider=1, num_product=3):
        self.num_customer = num_customer
        self.num_warehouse = num_warehouse
        self.num_provider = num_provider
        self.num_product = num_product
        self.reset()
        self.num_done = 0

    def reset(self):
        self.score = 0
        self.scores = Counter()
        self.num_done = 0
        products = [
            ('2EMM01370', 'FRONT CABINET', 1, 68),
#            ('NC098UL', 'RemoteControl', 5, 68),
#            ('26032T7967', 'Stand', 7, 113),
            ('2ESA04496', 'Screw A', 20, 23),
#            ('2ESA06068', 'Screw B', 43, 113),
            # ('26032T7967', 'Stand', 7, 3),
            # ('2EMM01370', 'FRONT CABINET', 1, 4),
            # ('2ESA04496', 'Screw A', 20, 5),
            # ('2ESA06068', 'Screw B', 43, 6),
            # ('NC098UL', 'RemoteControl', 5, 7),
        ]
        products = [Product(*p) for p in products]
        self.product_repo = {p.id: p for p in products}
        self.provider_repo = ProviderRepository()

        self.factories = [
            Factory(f'工場{i}', product_repo=self.product_repo, provider_repo=self.provider_repo)
            for i in range(self.num_provider)
        ]
        self.warehouses = [
            Warehouse(f'{i}', product_repo=self.product_repo, providers=self.factories, provider_repo=self.provider_repo)
            for i in range(self.num_warehouse)
        ]
        self.customers = [
            Customer(f'顧客{i}', product_repo=self.product_repo, providers=self.warehouses, provider_repo=self.provider_repo)
            for i in range(self.num_customer)
        ]
        for factory in self.factories:
            self.provider_repo.add(factory)
        for warehouse in self.warehouses:
            self.provider_repo.add(warehouse)
        for customer in self.customers:
            self.provider_repo.add(customer)
        return State(self.factories, self.warehouses, self.customers)

    def render(self):
        result = ''
        for p in self.customers:
            result += f'{str(p)}\n\n'
        for p in self.warehouses:
            result += f'{str(p)}\n\n'
        for p in self.factories:
            result += f'{str(p)}\n\n'

        result += f'score: {self.score}'
        return result

    def step(self, actions):
        reward = Counter()

        # 発注処理
        for action in actions:
            if action and action[2] > 0:
                from_warehouse_id, product_id, amount = action
                w = self.provider_repo.get(from_warehouse_id)
                p = w.product_repo[product_id]
                w.order_to_provider(Order(w.id, product_id, amount, p.lead_time_days))
                # 注文するコスト
                reward[(action[0], action[1])] -= 2

        # # 注文終了時のみ時間を進める
        # total_order_amount = sum([a for w, p, a in actions])
        # if total_order_amount == 0:
        self.num_done += 1
        for f in self.factories:
            f.step()
        for w in self.warehouses:
            for product_id in w.product_repo:
                reward[(w.id, product_id)] += w.step_for_product(product_id)
        for c in self.customers:
            c.step()

        # 終了条件の確認
        # 1. 100以上の注文が溜まったらゲームオーバー
        # done = False
        # for w in self.warehouses:
        #     all_orders = sum(w.stacking_order.values(), [])
        #     if sum([o.amount for o in all_orders]) > 4000:
        #         done = True
        done = self.num_done > 100

        self.score += sum(reward.values())
        for k in reward:
            self.scores[k] += reward[k]

        r = self.scores if done else {k: 0 for k in reward}
        return State(self.factories, self.warehouses, self.customers), r, done, {}


def play_game(env, player, debug=False, train=True):
    obs = env.reset()
    for _ in range(365 * 3):
        actions = player.select_action(obs, train)
        next_obs, reward, done, info = env.step(actions)
        player.update_experience(reward, obs)

        if debug:
            print(pd.DataFrame(actions).set_index([0, 1]).unstack(1).to_csv(sep='\t'))
            print(env.render())
            input()

        if done:
            break
    print(f'final score: {env.score}')
    if debug:
        print(env.render())
    return env.score


def train(player):
    '''
    training player model
    player_type: gradient_boosting, deep_q, simple_multi_layer
    '''
    epochs = 500
    env = Env(num_customer=1, num_warehouse=1, num_provider=1)

    for e in range(epochs):
        print(e, end=':')

        play_game(env, player)
        player.train()
        player.save(f'./iter-{e}-model.pickle')


def test(model_file_name):
    '''
    testing model
    '''
    if not os.path.exists(model_file_name):
        raise Exception(model_file_name + ' not found')

    env = Env(num_customer=1, num_warehouse=1, num_provider=1)
    player = DLMultiPlayers()
    player.load(model_file_name)
    play_game(env, player, debug=True, train=False)


def gradient_boosting():
    from players.gb_small_player import GBMultiPlayers
    train(GBMultiPlayers())

def deep_q():
    from players.dl_player import DLMultiPlayers
    train(DLMultiPlayers())

def simple_multi_layer():
    from players.mlp_small_player import MLPMultiPlayers
    train(MLPMultiPlayers())

if __name__ == '__main__':
    from fire import Fire
    Fire({
        'train': {
            'gradient_boosting': gradient_boosting,
            'deep_q': deep_q,
            'simple_multi_layer': simple_multi_layer,
        },
        'test': test,
    })
