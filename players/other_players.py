from random import choice

class InputPlayer:
    def select_action(self, obs):
        try:
            from_warehouse_id, product_id, amount = input().strip().split()
            return (from_warehouse_id, product_id, int(amount))
        except Exception as e:
            # print(e)
            return None

class RandomOrderPlayer:
    def select_action(self, obs, train):
        action_candidate = [(None, None, 0)]
        for w in obs.warehouses:
            for product_id in w.stock:
                action_candidate.append((w.id, product_id, 20))
        return [choice(action_candidate)]

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def train(self):
        pass

    def add_experience(self, s, a, r, s2):
        pass

class HeuristicPlayer:
    def select_action(self, obs):
        '''後手後手対応　足りなくなった物を注文する'''
        action_candidate = []
        all_choice = []
        for w in obs.warehouses:
            for product_id in w.stacking_order:
                total_order_amount = sum([o.amount for o in w.stacking_order[product_id]])
                num_stock = w.stock[product_id]
                if total_order_amount > num_stock:
                    action_candidate.append((w.id, product_id, (total_order_amount - num_stock)*2))

            for product_id in w.stock:
                all_choice.append((w.id, product_id, 20))
        return choice(action_candidate or all_choice)

