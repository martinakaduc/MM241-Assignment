import random
import numpy as np

from policy import Policy
from . import utils as ut

__all__ = ["QLearningPolicy"]


class QLearningPolicy(Policy):
    def __init__(self):
        self.num_stocks = 100
        self.max_product_quantity = 15
        self.state_size = (self.num_stocks + 10) * self.max_product_quantity
        self.action_size = 3000
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = np.zeros(1)
        ml_model = ut.MLModel()
        ml_model.load_model(self)

    def get_action(self, observation, info):
        state = self._extract_state(observation)
        if random.random() < self.epsilon:
            action = self._random_valid_action(observation)
        else:
            action = np.argmax(self.q_table[state])
        if action is None:  # Handle case where no valid action is found
            print("Warning: No valid action available during get_action.")

        return self._action_to_env(action, observation)

    def _extract_state(self, observation):
        stock_fill_levels = [min(int(np.sum(stock) / stock.size * 10), 10) for stock in observation["stocks"]]
        product_quantities = [min(prod["quantity"], self.max_product_quantity) for prod in observation["products"]]
        state_vector = stock_fill_levels + product_quantities
        state_hash = sum(x * (10 ** i) for i, x in enumerate(state_vector))
        return state_hash % self.state_size

    @staticmethod
    def compute_intermediate_reward(observation, next_observation):
        stocks_ratios_before = ut.compute_filled_ratio(observation["stocks"])
        stocks_ratios_after = ut.compute_filled_ratio(next_observation["stocks"])

        # Reward for increasing the filled ratio
        reward = 0
        for i in range(len(stocks_ratios_before)):
            if stocks_ratios_after[i] > stocks_ratios_before[i]:
                reward += (stocks_ratios_after[i] - stocks_ratios_before[i]) * 10

        return reward

    def learn(self, state, action, observation, next_observation, reward, next_state, done):
        intermediate_reward = self.compute_intermediate_reward(observation, next_observation)
        stocks_ratios_before = ut.compute_filled_ratio(observation["stocks"])
        stocks_ratios_after = ut.compute_filled_ratio(next_observation["stocks"])
        stock_idx = action // len(observation["products"])
        if stocks_ratios_after[stock_idx] > 0 and stocks_ratios_before[stock_idx] == 0:
            intermediate_reward -= 5  # Penalty for opening a new stock
        total_reward = reward + intermediate_reward

        # Update Q-value
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = total_reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _action_to_env(self, action, observation):
        stock_idx = action // len(observation["products"])
        product_idx = action % len(observation["products"])
        prod_size = observation["products"][product_idx]["size"]
        if action is None:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}
        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        for pos_x in range(stock_w - prod_size[0] + 1):
            for pos_y in range(stock_h - prod_size[1] + 1):
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
                # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                #     return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (pos_x, pos_y)}
        return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}

    def _random_valid_action(self, observation):
        stocks_ratios = ut.compute_filled_ratio(observation["stocks"])
        sorted_stock_indices = np.argsort(stocks_ratios)[::-1]
        for stock_idx in sorted_stock_indices:
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] == 0:
                    continue

                prod_size = product["size"]
                for pos_x in range(stock_w - prod_size[0] + 1):
                    for pos_y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            # Construct the action index and return
                            return stock_idx * len(observation["products"]) + product_idx
                        # if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                        #     # Construct the action index and return
                        #     return stock_idx * len(observation["products"]) + product_idx

        print("No valid actions available. Returning default action.")
        return None
