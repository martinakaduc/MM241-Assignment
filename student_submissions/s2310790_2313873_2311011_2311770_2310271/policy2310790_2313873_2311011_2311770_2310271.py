from policy import Policy
import torch
from torch import nn, optim
import numpy as np
from numpy import random
import os
import copy
from math import exp

class SortedList:
    """
    A list container that maintains ascending order after inserting a new element.
    """
    def __init__(self, sorted_array):
        self.sorted_array = sorted_array
    
    def add(self, new_element):
        for i, element in enumerate(self.sorted_array):
            if element >= new_element:
                self.sorted_array.insert(i, new_element)
                return
            
        self.sorted_array.append(new_element)

    def __getitem__(self, index):
        return self.sorted_array[index]
    
class CuttingStock2DModel(nn.Module):
    """
    A deep neural network model designed for the 2D cutting stock problem.

    This model consists of fully connected layers with Leaky ReLU activations, 
    and uses Softmax and Sigmoid functions in the final layers. It predicts 
    probabilities for stock selection, product selection, and product rotation 
    based on the input state.
    """
    def __init__(self, num_stocks, max_num_product_types, max_products_per_type):
        super().__init__()
        # The size of the state vector. For example, it will be 2168 when the number of stocks is 100, 
        # the maximum number of product types is 24, and the maximum number of products per type is 20.
        state_size = num_stocks * 2 + max_num_product_types * max_products_per_type * 4 + max_num_product_types * 2

        # Neural network dedicated to stock prediction.
        self.stock_layer = []
        layer_sizes = [state_size, 1536, 1024, 512, 256, 128, num_stocks]
        for i in range(1, len(layer_sizes)):
            self.stock_layer.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.stock_layer.append(nn.LeakyReLU())
        self.stock_layer = nn.Sequential(*[*self.stock_layer, nn.Linear(num_stocks, num_stocks), nn.Softmax(-1)])

        # Neural network dedicated to product prediction.
        self.product_layer = []
        layer_sizes = [state_size, 1536, 1024, 512, 256, 128, 64]
        for i in range(1, len(layer_sizes)):
            self.product_layer.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.product_layer.append(nn.LeakyReLU())
        self.product_layer = nn.Sequential(*[
            *self.product_layer, 
            nn.Linear(64, max_num_product_types), 
            nn.Softmax(-1)
        ])

        # Neural network dedicated to product rotation.
        # 0 is not rotated, 1 is rotated.
        self.rotate_layer = []
        layer_sizes = [state_size, 1024, 512, 256, 128, 32]
        for i in range(1, len(layer_sizes)):
            self.rotate_layer.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            self.rotate_layer.append(nn.LeakyReLU())
        self.rotate_layer = nn.Sequential(*[
            *self.rotate_layer, 
            nn.Linear(32, 1), 
            nn.Sigmoid()
        ])

    # Input shape: (2168,)
    #     [0 - 200]     : normalized width and height of up to 100 stocks
    #     [200 - 248]   : normalized width and height of up to 24 product types
    #     [248 - 1208]  : normalized coordinates of up to 24 * 20 products
    #     [1208 - 1688] : normalized index of the stock each product belongs to
    #     [1688 - 2168] : rotation status of each product (rotated or not)
    def forward(self, input):
        stock_prob = self.stock_layer(input) + 0.01
        stock_prob = stock_prob / stock_prob.sum(dim = -1, keepdims = True)
        product_prob = self.product_layer(input) + 0.05
        product_prob = product_prob / product_prob.sum(-1, keepdims = True)
        rotate_prob = self.rotate_layer(input) * 0.8 + 0.1

        return stock_prob, product_prob, rotate_prob

class GreedySAPolicy(Policy):
    """
    This policy first applies a greedy approach, followed by simulated annealing to refine the result. 
    The greedy approach selects the largest remaining product and places it into the largest available stock.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.first_time = True
        self.step = 0
        self.num_stocks = 0
        self.num_products = 0
        self.num_items = 0
        self.stocks = []
        self.stock_indices = None
        self.products = []
        self.product_indices = None
        self.state_matrix = []
        self.actions = []

    def get_action(self, observation, info):
        # Reset state if a new game has started.
        if info["filled_ratio"] == 0: self.reset()

        # The first time this method is called in a new game, it will calculate all the 
        # necessary cut positions for the items. After that, no further calculations are required.
        if self.first_time:
            self.num_stocks = len(observation["stocks"])
            self.num_products = len(observation["products"])
            self.num_items = 0

            for stock in observation["stocks"]:
                stock_width, stock_height = self._get_stock_size_(stock)
                self.stocks.append({
                    "width": stock_width,
                    "height": stock_height,
                    "products": [],
                    "top_bound": 0,
                    "right_bound": 0,
                    "grid": [SortedList([0, stock_width]), SortedList([0, stock_height])],
                    "occupied_cells": [[False]]
                })

            self.stock_indices = np.arange(self.num_stocks)

            for product in observation["products"]:
                self.products.append({
                    "width": product["size"][0],
                    "height": product["size"][1],
                    "demands": product["quantity"]
                })
                self.num_items += product["quantity"]
            
            self.product_indices = np.arange(self.num_products)
            self.greedy()
            self.simulated_annealing()
            self.first_time = False
        
        # Get the action.
        action = self.actions[self.step]
        self.step += 1
        return action
    
    def greedy(self):
        # Sort the stocks array in descending order of stock areas.
        self.stock_indices, _ = zip(
            *sorted(zip(self.stock_indices, self.stocks), key = lambda x: -x[1]["width"] * x[1]["height"])
        )

        # Sort the products array in descending order of item areas.
        self.product_indices, _ = zip(
            *sorted(zip(self.product_indices, self.products), key = lambda x: -x[1]["width"] * x[1]["height"])
        )

        # Place the items into stocks according to greedy algorithm.
        for product_index in self.product_indices:
            starting_stock_index = 0
            product_width = self.products[product_index]["width"]
            product_height = self.products[product_index]["height"]
            product_demands = self.products[product_index]["demands"]

            for _ in range(product_demands):
                # Find the first largest stock where the item can be placed.
                for stock_index in range(starting_stock_index, self.num_stocks):
                    stock_index = self.stock_indices[stock_index]

                    # Find the appropriate position in the stock to place the item.
                    # If a appropriate position is found, move to the next item.
                    if GreedySAPolicy.place_item(self.stocks[stock_index], product_width, product_height) is not None or \
                       GreedySAPolicy.place_item(self.stocks[stock_index], product_height, product_width) is not None:
                        break

                    starting_stock_index += 1
        
        # Trying to find smaller stocks to place items.
        GreedySAPolicy.tighten(self.stocks, self.stock_indices)
        # Insert items to state matrix.
        for i, stock in enumerate(self.stocks):
            for x, y, w, h in stock["products"]:
                self.state_matrix.append([w, h, i, x, y])
        # Convert the state matrix to a NumPy array for better performance during copying.
        self.state_matrix = np.array(self.state_matrix, dtype = np.int32)

    # Find the appropriate position in the stock to place the item and return that position,
    # if no such position is found, return None.
    @staticmethod
    def place_item(stock, product_width, product_height):
        occupied_cells = stock["occupied_cells"]
        num_row = len(occupied_cells)
        num_col = len(occupied_cells[0])
        verticals, horizontals = stock["grid"]

        # Loop through each cell of the grid and find the cluster of cells that are not occupied,
        # where the item can be placed.
        for i in range(num_row):
            for j in range(num_col):
                if not occupied_cells[i][j]:
                    # Check if there is enough space to the right of the current cell.
                    right_edge = None
                    for k in range(j + 1, num_col + 1):
                        if occupied_cells[i][k - 1]:
                            break

                        if verticals[k] >= verticals[j] + product_width:
                            right_edge = k
                            break

                    # Check if there is enough space above the current cell.
                    if right_edge != None:
                        for k in range(i + 1, num_row + 1):
                            if occupied_cells[k - 1][j]:
                                break

                            if horizontals[k] >= horizontals[i] + product_height:
                                obstacle = False
                                for check_space_i in range(i, k):
                                    for check_space_j in range(j, right_edge):
                                        if occupied_cells[check_space_i][check_space_j]:
                                            obstacle = True
                                            break
                                    if obstacle:
                                        break
                                
                                if obstacle:
                                    break
                                
                                # Update the grid after the item is placed.
                                if verticals[j] + product_width < verticals[right_edge]:
                                    verticals.add(verticals[j] + product_width)
                                    for row in occupied_cells:
                                        row.insert(right_edge, row[right_edge - 1])

                                if horizontals[i] + product_height < horizontals[k]:
                                    horizontals.add(horizontals[i] + product_height)
                                    occupied_cells.insert(k, copy.deepcopy(occupied_cells[k - 1]))

                                # Update the occupied cells after the item is placed.
                                for m in range(i, k):
                                    for n in range(j, right_edge):
                                        occupied_cells[m][n] = True
                                
                                # Place the item.
                                stock["products"].append(
                                    [verticals[j], horizontals[i], product_width, product_height]
                                )

                                # Update inner bound.
                                if stock["right_bound"] < verticals[j] + product_width:
                                    stock["right_bound"] = verticals[j] + product_width
                                if stock["top_bound"] < horizontals[i] + product_height:
                                    stock["top_bound"] = horizontals[i] + product_height
                                return verticals[j], horizontals[i]
    
    @staticmethod
    def tighten(stocks, stocks_sort):
        # Sort the stocks array in descending order of wasted areas.
        wasted_indices = np.arange(len(stocks))
        wasted_indices, _ = zip(
            *sorted(
                zip(wasted_indices, stocks), 
                key = lambda x: float("inf") if x[1]["right_bound"] * x[1]["top_bound"] == 0 
                else x[1]["right_bound"] * x[1]["top_bound"] - x[1]["width"] * x[1]["height"]
            )
        )

        # Iterate through the most wasted stocks and move all their products to other stocks if smaller.
        for stock_index in wasted_indices:
            stock = stocks[stock_index]
            # Start by checking the smallest stocks first.
            for i in stocks_sort[::-1]:
                replace_stock = stocks[i]
                if replace_stock["top_bound"] * replace_stock["right_bound"] == 0 and \
                   replace_stock["width"] >= stock["right_bound"] and replace_stock["height"] >= stock["top_bound"] and \
                   replace_stock["width"] * replace_stock["height"] < stock["width"] * stock["height"]:
                    replace_stock["products"] = stock["products"]
                    replace_stock["right_bound"] = stock["right_bound"]
                    replace_stock["top_bound"] = stock["top_bound"]

                    stock["products"] = []
                    stock["right_bound"] = 0
                    stock["top_bound"] = 0
                    break
              
    # Calculate energy of the given state.
    def energy(self, state, step = 0):
        # Define hyperparameters.
        overlap_weight = (0.95 + step / 600) ** 9
        area_weight = 1e-6
        distance_weight = 5e-10

        overlap = 0
        distance = 0
        area = 0
        stock_register = [False] * self.num_stocks

        overlap_register = [False] * self.num_items

        for i in range(self.num_items):
            wi, hi, zi, xi, yi = state[i]

            # Calculate the total area of stocks that have been cut.
            if not stock_register[zi]: 
                area += self.stocks[zi]["width"] * self.stocks[zi]["height"]
                stock_register[zi] = True

            # Calculate the total overlapping area and the distance between items.
            for j in range(i + 1, self.num_items):
                wj, hj, zj, xj, yj = state[j]
                if zi == zj:
                    overlap_area = max(0, min(xi + wi, xj + wj) - max(xi, xj)) * max(0, min(yi + hi, yj + hj) - max(yi, yj))
                    if overlap_area > 0:
                        overlap_register[i] = overlap_register[j] = True
                    overlap += overlap_area
                    distance += (xi + wi / 2 - xj - wj / 2) ** 2 + (yi + hi / 2 - yj - hj / 2) ** 2

        # Return a tuple containing energy, overlapping area, list of overlap items, and the area of stocks used.
        return (
            overlap_weight * overlap + area_weight * area + distance_weight * distance,
            overlap, overlap_register,
            area
        )

    # Calculate the probability of transitioning to the next state.
    def transition_probability(self, current_state_energy, next_state_energy, temperature):
        if next_state_energy <= current_state_energy:
            return 1
        
        return exp((current_state_energy - next_state_energy) / temperature)
    

    # Check if the position for placing the item is within the stock's bounds.
    def is_inside(self, stock_width, stock_height, product_size, position):
        return product_size[0] + position[0] <= stock_width and product_size[1] + position[1] <= stock_height
    
    # Helper function to generate a random integer centered around 0, following a normal distribution.
    def rand(self, std):
        return int(random.randn() * std)
    
    # Choose a random next state that is near the current state.
    def choose_next_state(self, current_state, overlap_register):
        next_state = current_state.copy()
        prob_list = np.array(overlap_register, dtype = np.float32)
        overlap_num = prob_list.sum()
        if overlap_num == len(current_state):
            prob_list /= len(current_state)
        elif overlap_num > 0:
            prob_list += 0.3 * overlap_num / (0.7 * len(current_state) - overlap_num).clip(1e-5)
            prob_list /= prob_list.sum()
        else:
            prob_list += 1
            prob_list /= len(current_state)

        # Swap the positions of two randomly selected items that are near each other.
        if random.randint(0, 100) > 50:
            i = random.randint(0, self.num_items)
            j = min(max(0, i + self.rand(20)), self.num_items - 1)
            wi, hi, zi, xi, yi = next_state[i]
            wj, hj, zj, xj, yj = next_state[j]

            if self.is_inside(self.stocks[zi]["width"], self.stocks[zi]["height"], (wj, hj), (xi, yi)) and \
               self.is_inside(self.stocks[zj]["width"], self.stocks[zj]["height"], (wi, hi), (xj, yj)):
                next_state[i][:2] = [wj, hj]
                next_state[j][:2] = [wi, hi]
        
        # Move a randomly selected item to another randomly selected stock, prioritizing items with overlap.
        if random.randint(0, 100) > 40:
            i = random.choice(self.num_items, p = prob_list)
            wi, hi, zi, xi, yi = next_state[i]
            new_stock_index = min(max(0, zi + self.rand(15)), self.num_stocks - 1)

            if self.is_inside(self.stocks[new_stock_index]["width"], self.stocks[new_stock_index]["height"], (wi, hi), (xi, yi)):
                next_state[i][2] = new_stock_index

        # Move a randomly selected item to a nearby random location, prioritizing items with overlap.
        if random.randint(0, 100) > 10:
            i = random.choice(self.num_items, p = prob_list)
            wi, hi, zi, xi, yi = next_state[i]

            next_state[i][3:] = [
                min(max(0, xi + self.rand(30)), self.stocks[zi]["width"] - wi),
                min(max(0, yi + self.rand(30)), self.stocks[zi]["height"] - hi)
            ]

        # Rotate a randomly selected item, prioritizing items with overlap.
        if random.randint(0, 100) > 30:
            i = random.choice(self.num_items, p = prob_list)
            wi, hi, zi, xi, yi = next_state[i]

            next_state[i][:2] = [hi, wi]

        return next_state

    # Perform the simulated annealing algorithm on the state matrix and precompute all actions.
    def simulated_annealing(self):
        # Define hyperameters.
        T = 1000
        T_min = 1
        eps = 0.95

        current_state = self.state_matrix.copy()
        min_area = self.energy(current_state)[3]

        step = 0
        while T > T_min:
            # Calculate the energy of the current and next state.
            current_state_energy, _, overlap_register, _ = self.energy(current_state, step)
            next_state = self.choose_next_state(current_state, overlap_register)
            next_state_energy, next_state_overlap, _, next_state_area = self.energy(next_state, step)

            # Based on the energies, compute the probability of transitioning to the next state.
            prob = self.transition_probability(current_state_energy, next_state_energy, T)
            if prob > random.rand():
                current_state = next_state

                # If the next state has no overlapping items and uses less stock area than the current best state, 
                # update the current best state to the next state.
                if next_state_overlap == 0 and next_state_area < min_area:
                    self.state_matrix = next_state
                    min_area = next_state_area

            # Reduce the temperature
            T *= eps
            step += 1
                
        # Compute all the nessessary actions
        for state_vector in self.state_matrix:
            self.actions.append({
                "size": state_vector[:2],
                "position": state_vector[3:],
                "stock_idx": state_vector[2]
            })

# Reinforcement learning combined with greedy policy
class RLPolicy(Policy):
    """
    This is a straightforward reinforcement policy for the 2D cutting stock problem, 
    using a naive policy gradient algorithm with entropy regularization to achieve its objective. 

    However, it is specifically designed for scenarios where the number of stocks does not exceed 100, 
    the maximum number of product types is 24, and the maximum quantity per product type is 20. 
    Outside these conditions, the policy defaults to a simple greedy approach.

    A pre-trained model is provided in the same folder as this file. 
    This policy utilizes the weights from the pre-trained model to compute actions, 
    eliminating the need for further training.
    """
    def __init__(self, is_training = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0
        self.actions = []
        self.reset_train_data()
        self.model = CuttingStock2DModel(
            num_stocks = 100, 
            max_num_product_types = 24, 
            max_products_per_type = 20
        )
        if self.device == "cuda":
            self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        self.load()
        self.is_training = is_training

    def reset_train_data(self):
        self.states = []
        self.stock_indices = []
        self.product_indices = []
        self.rotations = []
        self.rewards = []

    def get_action(self, observation, info):
        # This code runs at the start of each new game to initialize the state and precompute 
        # all actions required for the episode, ensuring that no further calculations are needed 
        # during the course of the game.
        if self.step == len(self.actions):
            # If we collect enough data, begin the training, and then collect the data again.
            if self.is_training and len(self.rewards) >= 20:
                self.train_batch()
                self.reset_train_data()
            torch.set_grad_enabled(False)
            self.rewards.append([])
            self.step = 0
            self.actions = []

            # Select the algorithm based on the environment.
            algorithm = "rl"
            if len(observation["stocks"]) != 100 or len(observation["products"]) > 24:
                algorithm = "greedy"
            else:
                for product in observation["products"]:
                    if product["quantity"] > 20:
                        algorithm = "greedy"
                        break

            if algorithm == "rl":
                state = torch.zeros(2168, device = self.device)
                stock_used_list = [False] * 100

            # stocks is a list of stock objects, where each element contains all the relevant information
            # about a particular stock.
            stocks = []
            max_stock_size = 0
            for i, stock in enumerate(observation["stocks"]):
                stock_width, stock_height = self._get_stock_size_(stock)
                if algorithm == "rl":
                    state[i], state[i + 100] = stock_width, stock_height
                stocks.append({
                    "width": stock_width,
                    "height": stock_height,
                    "grid": [SortedList([0, stock_width]), SortedList([0, stock_height])],
                    "occupied_cells": [[False]],
                    "top_bound": 0,
                    "right_bound": 0,
                    "products": []
                })
                if stock_width > max_stock_size:
                    max_stock_size = stock_width
                if stock_height > max_stock_size:
                    max_stock_size = stock_height

            # stocks_sort contains the indices of stocks sorted in descending order based on their area.
            stocks_sort = torch.arange(0, len(stocks))
            stocks_sort, _ = zip(
                *sorted(zip(stocks_sort, stocks), key = lambda x: -x[1]["width"] * x[1]["height"])
            )

            # num_products_left represents the number of products remaining to be cut in the current episode.
            # products is a list of product objects, where each element contains all the relevant information
            # about a particular product.
            products = []
            num_products_left = 0
            for i, product in enumerate(observation["products"]):
                if algorithm == "rl":
                    state[i + 200], state[i + 224] = product["size"]
                    state[1208 + i * 20 + product["quantity"]: 1228 + i * 20] = -100
                num_products_left += product["quantity"]
                products.append({
                    "width": product["size"][0],
                    "height": product["size"][1],
                    "quantity": product["quantity"],
                    "demands": product["quantity"]
                })

            if algorithm == "rl":
                state[1208 + len(observation["products"]) * 20: 1688] = -100
            
            # products_sort contains the indices of products sorted in descending order based on their area.
            products_sort = torch.arange(0, len(products))
            products_sort, _ = zip(
                *sorted(zip(products_sort, products), key = lambda x: -x[1]["width"] * x[1]["height"])
            )
        
            if algorithm == "rl":
                while num_products_left > 0:
                    # Normalize state
                    norm_state = state.clone()
                    norm_state[:200] = (norm_state[:200] / max_stock_size * 100 - 75) / 25
                    norm_state[200:248] = (norm_state[200:248] / max_stock_size * 100 - 25) / 25
                    norm_state[248:1208] = norm_state[248:1208] / max_stock_size
                    norm_state[1208:1688] /= 100
                    self.states.append(norm_state)

                    # Calculate the distributions.
                    stock_prob, product_prob, rotate_prob = self.model(norm_state)

                    # In training mode, we select the stock and product randomly based on their distributions.
                    # In production mode, we choose the stock and product with the highest probability.
                    if self.is_training:
                        stock_idx = torch.multinomial(stock_prob, 1).item()
                        product_idx = torch.multinomial(product_prob, 1).item()
                        is_rotated = int(torch.bernoulli(rotate_prob).item())
                    else:
                        stock_idx = stock_prob.argmax().item()
                        product_idx = product_prob.argmax().item()
                        is_rotated = int(rotate_prob.item() > 0.5)

                    rotate_product_idx = product_idx
                    rotate_stock_idx = stock_idx
                    rotate_is_rotated = is_rotated

                    reward = 0
                    i = 0
                    # If the selected product index is out of range or the product type has no remaining products, 
                    # the system will choose another product until a valid one is found. Each failure will reduce the 
                    # reward by 1.
                    while rotate_product_idx >= len(products) or \
                            products[rotate_product_idx]["quantity"] == 0:
                        reward -= 1
                        rotate_product_idx = products_sort[i].item()
                        i += 1
                        
                    product = products[rotate_product_idx]
                    # Select best position to place the product inside the stock and the rotation status.
                    if is_rotated:
                        position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["height"], product["width"])
                    else:
                        position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["width"], product["height"])

                    i = 0
                    # If the stock has no available space for the product, move on to the next stock and reduce
                    # the reward by 1.
                    while position is None:
                        reward -= 1
                        rotate_is_rotated = 1 - is_rotated
                        if rotate_is_rotated:
                            position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["height"], product["width"])
                        else:
                            position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["width"], product["height"])
                        if position is not None:
                            break
                        reward -= 1
                        rotate_is_rotated = is_rotated
                        rotate_stock_idx = stocks_sort[i]
                        if rotate_is_rotated:
                            position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["height"], product["width"])
                        else:
                            position = GreedySAPolicy.place_item(stocks[rotate_stock_idx], product["width"], product["height"])
                        i += 1

                    # Once an appropriate product is placed in the right stock at the correct position, 
                    # we decrease the number of products left by 1 and update the state vector.
                    num_products_left -= 1
                    offset = product["demands"] - product["quantity"]
                    state[248 + 40 * rotate_product_idx + 2 * offset] = position[0]
                    state[249 + 40 * rotate_product_idx + 2 * offset] = position[1]
                    state[1208 + rotate_product_idx * 20 + offset] = rotate_stock_idx + 1
                    state[1688 + rotate_product_idx * 20 + offset] = rotate_is_rotated

                    # If the selected stock is a new stock, decrease the reward by a factor of its area.
                    # Otherwise, if the chosen stock has already been cut, increase the reward by 1.
                    if not stock_used_list[rotate_stock_idx]:
                        reward -= (state[rotate_stock_idx] * state[rotate_stock_idx + 100] / 10000).item()
                        stock_used_list[rotate_stock_idx] = True
                    else:
                        reward += 1

                    # Save the state, the chosen action and reward at each step of an episode for later training.
                    self.stock_indices.append(stock_idx)
                    self.product_indices.append(product_idx)
                    self.rotations.append(is_rotated)
                    self.rewards[-1].append(reward)
                    product["quantity"] -= 1
            # Greedy algorithm.
            else:
                # Place the items into stocks according to greedy algorithm.
                for idx in products_sort:
                    product = products[idx.item()]
                    start_idx = 0
                    for _ in range(product["demands"]):
                        # Find the first largest stock where the item can be placed.
                        for i in range(start_idx, len(stocks)):
                            # Find the appropriate position in the stock to place the item.
                            # If a appropriate position is found, move to the next item.
                            if GreedySAPolicy.place_item(stocks[stocks_sort[i]], product["width"], product["height"]) or \
                               GreedySAPolicy.place_item(stocks[stocks_sort[i]], product["height"], product["width"]):
                                break
                            start_idx += 1

            # Find more optimal solution by tightening.
            GreedySAPolicy.tighten(stocks, stocks_sort)

            # Generate all the required actions.
            for i, stock in enumerate(stocks):
                for x, y, w, h in stock["products"]:
                    self.actions.append({
                        "stock_idx": i,
                        "size": (w, h),
                        "position": (x, y)
                    })

        # Choose next precompute action.
        action = self.actions[self.step]
        self.step += 1
        return action

    def train_batch(self):
        torch.set_grad_enabled(True)

        # Convert all the states to tensors
        states = torch.stack(self.states, 0)
        stock_indices = torch.tensor(self.stock_indices, device = self.device)
        product_indices = torch.tensor(self.product_indices, device = self.device)
        rotations = torch.tensor(self.rotations, device = self.device)

        rewards = []
        for episode_rewards in self.rewards:
            reward = sum(episode_rewards)
            rewards.extend([reward] * len(episode_rewards))

        rewards = torch.tensor(rewards, device = self.device)
        stock_probs, product_probs, rotate_probs = self.model(states)
        rotate_probs = torch.cat((1 - rotate_probs, rotate_probs), -1)

        # Calculate the log probabilies.
        stock_log_probs = torch.log(stock_probs[torch.arange(len(stock_indices)), stock_indices])
        product_log_probs = torch.log(product_probs[torch.arange(len(product_indices)), product_indices])
        rotate_log_probs = torch.log(rotate_probs[torch.arange(len(rotations)), rotations])

        # Compute the entropy loss for regularization to prevent getting stuck in a local optimum.
        stock_entropy = (-stock_probs * torch.log(stock_probs)).mean() * 4
        product_entropy = (-product_probs * torch.log(product_probs)).mean() * 1.6
        rotate_entropy = (-rotate_probs * torch.log(rotate_probs)).mean() * 0.7
        entropy_loss = -stock_entropy + product_entropy + rotate_entropy

        # Calculate the loss according to policy gradient theorem with entropy.
        loss = -((stock_log_probs + product_log_probs + rotate_log_probs) * rewards).sum() * 1e-6 + entropy_loss
        print("Loss: ", loss.item())
        # Update parameters
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.save()
        torch.cuda.empty_cache()

    # Save model
    def save(self, file_name = "cutting_stock_2d_model.ckpt"):
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, abs_path)
    
    # Load model
    def load(self, file_name = "cutting_stock_2d_model.ckpt"):
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        if os.path.exists(abs_path):
            try:
                saved_model = torch.load(abs_path, weights_only = True, map_location = self.device)
                if (type(saved_model) == dict):
                    self.model.load_state_dict(saved_model["model"])
                    self.optimizer.load_state_dict(saved_model["optimizer"])
                else:
                    self.model.load_state_dict(saved_model)
            except:
                print("Failed to load the model.")

class Policy2310790_2313873_2311011_2311770_2310271(Policy):
    """
    This policy combines a greedy approach with simulated annealing and reinforcement learning.

    - `policy_id = 1` corresponds to the greedy approach with simulated annealing.
    - `policy_id = 2` corresponds to the reinforcement learning policy.
    """
    def __init__(self, policy_id = 1, is_training = False):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy = GreedySAPolicy() if policy_id == 1 else RLPolicy(is_training)

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
