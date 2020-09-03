from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4
from random import choice

from termcolor import colored

@dataclass
class Product:
    id: str
    name: str
    daily_order: int
    lead_time_days: int


@dataclass
class Order:
    order_from : str
    product_id : str
    amount : int
    deadline : int
    id : str = ''

    def __post_init__(self):
        self.id = uuid4()

    def timepast(self):
        self.deadline -= 1


class ProviderRepository:
    def __init__(self):
        self.repo = {}
        self.name_id_map = {}

    def add(self, provider):
        self.repo[provider.id] = provider
        self.name_id_map[provider.name] = provider.id

    def get(self, id_or_name):
        if id_or_name in self.repo:
            return self.repo[id_or_name]
        if id_or_name in self.name_id_map:
            return self.repo[self.name_id_map[id_or_name]]


class Provider:
    '''Provider entity'''
    def __init__(self, name, product_repo=None, providers=None, provider_repo=None):
        '''
        initialize Provider entity

        prepare initial stock
        prepare initial order

        '''
        self.id = str(uuid4())[:10]
        self.name = name
        self.stock = defaultdict(int)
        self.stacking_order = defaultdict(list)
        self.waiting_order = {}

        self.providers = providers or {}
        self.product_repo = product_repo or {}
        self.provider_repo = provider_repo or ProviderRepository()

    def __str__(self):
        '''
        状態を文字列にする
        '''
        result = ''
        item_ids = self.product_repo.keys()
        item_names = [self.product_repo[item_id].name for item_id in item_ids]
        item_stocks = [
            colored(str(self.stock[item_id]),'green' if self.stock[item_id] > 0 else 'red')
            for item_id in item_ids
        ]
        item_orders = [
            colored(
                str(sum([order.amount for order in self.stacking_order[item_id]])),
                'green' if sum([order.amount for order in self.stacking_order[item_id]]) == 0 else 'red'
            )
            for item_id in item_ids
        ]
        waiting_orders = [
            str(sum([order.amount for order in self.waiting_order.values() if order.product_id == item_id]))
            for item_id in item_ids
        ]

        result += '\t'.join([self.name] + item_names) + '\n'
        result += '\t'.join(['在庫'] + item_stocks) + '\n'
        result += '\t'.join(['注文'] + item_orders) + '\n'
        result += '\t'.join(['注文済'] + waiting_orders) + '\n'
        return result

    def is_stock_enough(self, order):
        '''check if order can be fullfilled'''
        return order.amount <= self.stock[order.product_id]

    def fulfill_order(self, order):
        # print(f'{self.name} fulfilled order for order: {order.id}')
        self.stock[order.product_id] -= order.amount
        self.stacking_order[order.product_id] = [o for o in self.stacking_order[order.product_id] if o.id != order.id]
        return self.provider_repo.get(order.order_from).order_fulfilled(order)

    def order_fulfilled(self, order):
        # print(f'order: {order.id} is fulfilled')
        self.stock[order.product_id] += order.amount
        del self.waiting_order[order.id]
        return 100 # reward for product

    def receive_order(self, order):
        # print(f'order received from {self.provider_repo.get(order.order_from).name} to {self.name}')
        self.stacking_order[order.product_id].append(order)

    def send_order(self, order, provider):
        # print(f'order sended from {self.name} to {provider.name}')
        self.waiting_order[order.id] = order
        provider.receive_order(order)

    def step(self):
        pass


class Warehouse(Provider):
    '''Warehouse entity'''
    def __init__(self, name, product_repo=None, providers=None, provider_repo=None):
        '''
        initialize Warehouse entity

        prepare initial stock
        prepare initial order

        '''
        super(Warehouse, self).__init__(name, product_repo, providers, provider_repo)

        for _ in range(3):
            product = choice(list(self.product_repo.values()))
            self.stock[product.id] += product.daily_order * product.lead_time_days // 2

    def step(self):
        reward = 0
        for _, order_list in self.stacking_order.items():
            for order in order_list:
                if self.is_stock_enough(order):
                    reward += self.fulfill_order(order)
        return reward

    def step_for_product(self, product_id):
        reward = 0
        for order in self.stacking_order[product_id]:
            if self.is_stock_enough(order):
                reward += self.fulfill_order(order)
        waiting_orders = sum([o.amount for o in self.waiting_order.values() if o.product_id == product_id])
        stacking_orders = sum([o.amount for o in self.stacking_order[product_id]])
        reward += min((stacking_orders - waiting_orders) * -10, 0)
        reward += self.stock[product_id] * -0.1 if self.stock[product_id] > 10 else 0
        return reward

    def find_provider_for_product(self, product_id):
        result = []
        for provider in self.providers:
            if product_id in provider.product_repo:
                result.append(provider)
        return result

    def order_to_provider(self, order):
        provider_candidate = self.find_provider_for_product(order.product_id)
        if len(provider_candidate) == 0:
            raise Exception(f'Provider not found for product: {order.product_id}')

        self.send_order(order, choice(provider_candidate))


class Factory(Provider):
    def __init__(self, name, product_repo=None, providers=None, provider_repo=None):
        super(Factory, self).__init__(name, product_repo, providers, provider_repo)
        self.order_queue = []

    def step(self):
        '''
        order is fulfilled on deadline equals 0
        '''
        for order in self.order_queue:
            order.timepast()

        for order in self.order_queue:
            if order.deadline == 0:
                self.fulfill_order(order)

        return 0 # Factoryはrewardを生まない

    def fulfill_order(self, order):
        self.stacking_order[order.product_id] = [o for o in self.stacking_order[order.product_id] if o.id != order.id]
        return self.provider_repo.get(order.order_from).order_fulfilled(order)

    def receive_order(self, order):
        super(Factory, self).receive_order(order)
        self.order_queue.append(order)


class Customer(Provider):
    def step(self):
        '''
        create order randomly to provider
        '''
        # provider = choice(self.providers)
        # product = choice(list(self.product_repo.values()))

        num_remaining_waiting_order = len(self.waiting_order)

        for provider in self.providers:
            for product in self.product_repo.values():
                self.send_order(Order(self.id, product.id, product.daily_order, 1), provider)

        return -0.1 * num_remaining_waiting_order # minus reward for num of waiting orders
