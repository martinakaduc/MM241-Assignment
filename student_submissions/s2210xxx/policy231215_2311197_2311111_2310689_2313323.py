from policy import Policy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple
from scipy import ndimage

class DQNetwork(nn.Module):
    def __init__(self, num_stocks=100, stock_size=100, learning_rate=0.001):
        super(DQNetwork, self).__init__()
        self.stock_size = stock_size
        self.num_stocks = num_stocks

        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers for product dimensions
        self.fc_product = nn.Linear(2, 128)

        # Fully connected layers for combining features
        self.fc1 = nn.Linear(128 * stock_size * stock_size + 128, 256)
        self.fc_stock = nn.Linear(256, num_stocks)  # Predict stock index
        self.fc_cut = nn.Linear(256, 2)  # Predict x, y coordinates

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, stock_matrix, product_dimensions):
        """
        Forward pass for the network.

        Parameters:
        - stock_matrix: Tensor of shape [100, 1, stock_size, stock_size]
        - product_dimensions: Tensor of shape [1, 2]

        Returns:
        - stock_idx: Predictions for stock index.
        - cut_position: Predictions for cut positions (x, y).
        """
        # Process stock matrix through convolutional layers
        x = F.relu(self.conv1(stock_matrix))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten spatial features

        # Process product dimensions
        product_features = F.relu(self.fc_product(product_dimensions))
        product_features = product_features.repeat(x.size(0), 1)  # Match batch size of x

        # Combine spatial and product features
        combined_features = torch.cat((x, product_features), dim=1)

        # Stock index prediction
        stock_features = F.relu(self.fc1(combined_features))
        stock_idx = self.fc_stock(stock_features)

        # Cut position prediction
        cut_position = self.fc_cut(stock_features)

        return stock_idx, cut_position


class Policy231215_2311197_2311111_2310689_2313323(Policy):
    def __init__(
            self, 
            policy_id = 1,
            num_stocks=100, 
            stock_size=100,
            waste_penalty: float = 0.1, 
            rotation_enabled: bool = True,
            fragmentation_weight: float = 0.3
        ):
        super().__init__()
        assert policy_id in [1, 2], "Policy must be 1 or 2"

        self.policyID = policy_id
        if self.policyID == 1:
            self.network = DQNetwork(num_stocks, stock_size)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
            self.epsilon = 1.0
            self.decay_rate = 0.99
        else:
            self.current_stock_index = 0
            self.waste_penalty = waste_penalty
            self.rotation_enabled = rotation_enabled
            self.fragmentation_weight = fragmentation_weight
            self.stock_placements = {}
        
    def _get_usable_stock_dimensions(self, stock: np.ndarray) -> Tuple[int, int]:
        """
        Get the usable width and height of the stock.
        Assumes -2 represents unusable areas.
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h
    
    def _get_usable_stock_dimensions(self, stock: np.ndarray) -> Tuple[int, int]:
        """
        Get the usable width and height of the stock.
        Assumes -2 represents unusable areas.
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _calculate_waste_score(self, stock: np.ndarray, placed_product_size: Tuple[int, int]) -> float:
        """
        Calculate an advanced waste score considering multiple factors.
        """
        # Total usable area
        total_usable_area = np.sum(stock == -1)
        
        # Area of placed product
        placed_area = np.prod(placed_product_size)
        
        # Remaining empty area
        remaining_empty_area = total_usable_area - placed_area
        
        # Calculate fragmentation score
        fragmentation_score = self._calculate_fragmentation(stock)
        
        # Waste calculation with multiple factors
        waste_ratio = max(0, remaining_empty_area / total_usable_area)
        
        # Combine waste ratio with fragmentation
        combined_waste = (
            waste_ratio * (1 - self.fragmentation_weight) + 
            fragmentation_score * self.fragmentation_weight
        )
        
        return combined_waste * (1 + self.waste_penalty)

    def _calculate_fragmentation(self, stock: np.ndarray) -> float:
        """
        Assess the fragmentation of empty spaces.
        """
        # Identify empty regions
        binary_stock = (stock == -1).astype(int)
        labeled_array, num_regions = ndimage.label(binary_stock)
        
        # If no empty regions, return 0
        if num_regions == 0:
            return 0
        
        # Calculate average empty region size
        empty_region_sizes = np.bincount(labeled_array.ravel())[1:]
        avg_region_size = np.mean(empty_region_sizes)
        
        # Fragmentation score: more regions = higher fragmentation
        return num_regions / (avg_region_size + 1)

    def find_best_placement(self, stock: np.ndarray, observation) -> Dict:
        """
        Find the most efficient placement for products in a single stock.
        
        Args:
            stock: Stock grid representation
            products: List of products to be placed
        
        Returns:
            Best placement dictionary or None
        """
        stock_w, stock_h = self._get_usable_stock_dimensions(stock)
        best_placement = None
        min_waste = float('inf')

        # Sort products by area in descending order (best-fit heuristic)
        #print(products)
        # sorted_products = sorted(
        #     [p for p in products if p['quantity'] > 0], 
        #     key=lambda p: p['size'][0] * p['size'][1], 
        #     reverse=True
        # )
        products = observation["products"]
        for product in products:
            if(product['quantity'] == 0):
                continue
            size = product["size"]
            
            prod_w, prod_h = size

            if stock_w < prod_w or stock_h < prod_h:
                continue

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), size):
                        temp_stock = stock.copy()
                        temp_stock[x:x+prod_w, y:y+prod_h] = 1
                        
                        waste_score = self._calculate_waste_score(temp_stock, size)
                            
                        if waste_score < min_waste:
                            min_waste = waste_score
                            best_placement = {
                                "product": product,
                                "size": size,
                                "position": (x, y)
                            }
                        
                
            if stock_w < prod_h or stock_h < prod_w:
                continue
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(stock, (x, y), size[::-1]):

                        size = size[::-1]
                        

                        temp_stock_ = stock.copy()
                        temp_stock_[x:x+prod_h, y:y+prod_w] = 1
                            
                        waste_score = self._calculate_waste_score(temp_stock_, size[::-1])
                            
                        if waste_score < min_waste:
                            min_waste = waste_score
                            product["size"] = size
                            best_placement = {
                                "product": product,
                                "size": size,
                                "position": (x, y)
                           }
        
        print(best_placement)
        return best_placement

    def is_stock_fully_utilized(self, stock: np.ndarray, placed_products: List[Dict]) -> bool:
        """
        Check if the stock is considered fully utilized.
        """
        remaining_area = np.sum(stock == -1)
        max_product_size = max(
            [np.prod(prod['size']) for prod in placed_products], 
            default=0
        )
        return remaining_area < max_product_size * 1.5

    def get_action(self, observation, info):
        """
        Predict action for a single product.
        """
        if self.policyID == 1:
            # Encode stocks
            stocks = np.array(observation['stocks'])  # Convert tuple to NumPy array
            stocks = np.where(stocks == -1, 1, 0)  # Replace -1 with 1 and others with 0
            
            stock_matrix = torch.tensor(stocks, dtype=torch.float32).unsqueeze(1)

            # Define Sobel and Laplacian filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

            # Apply convolution for edge and corner detection
            edges_x = F.conv2d(stock_matrix, sobel_x, padding=1)  # Horizontal edges
            edges_y = F.conv2d(stock_matrix, sobel_y, padding=1)  # Vertical edges
            edges = torch.sqrt(edges_x**2 + edges_y**2)    # Edge magnitude

            corners = F.conv2d(stock_matrix, laplacian, padding=1).abs()  # Laplacian for corners

            # Amplify corners more than edges for stronger highlight
            highlighted = stock_matrix + edges + 2 * corners  # Adjust corner weight (e.g., 2x)

            # Normalize the matrix (optional for visualization)
            stock_matrix = (highlighted - highlighted.min()) / (highlighted.max() - highlighted.min())

            for product in observation['products']:
                # Find a product with quantity != 0
                if product['quantity'] == 0:
                    continue
                
                # Store the product for later fetch
                placed_product = product
                product_dimensions = torch.tensor(product['size'], dtype=torch.float32)

                # Get output for normal product
                # Forward pass
                pred_stock, pred_cut = self.network(stock_matrix, product_dimensions)

                # Decode outputs for action
                batch_idx = torch.argmax(pred_stock.max(dim=1).values).item()  # Batch index of the largest value
                stock_idx = torch.argmax(pred_stock[batch_idx]).item()         # Stock index within that batch
                cut_position = pred_cut[stock_idx]

                # Rotate the product
                # Forward pass
                rotated_pred_stock, rotated_pred_cut = self.network(stock_matrix, product_dimensions.flip(-1))

                # Decode outputs for action
                rotated_batch_idx = torch.argmax(rotated_pred_stock.max(dim=1).values).item()  # Batch index of the largest value
                rotated_stock_idx = torch.argmax(rotated_pred_stock[rotated_batch_idx]).item()         # Stock index within that batch
                rotated_cut_position = rotated_pred_cut[rotated_stock_idx]

                # If the rotated product got a better score, choose it
                reward = self.evaluate_action(stocks, product_dimensions, stock_idx, cut_position)
                rotated_reward = self.evaluate_action(stocks, product_dimensions.flip(-1), rotated_stock_idx, rotated_cut_position)

                if (reward < rotated_reward):
                    # Change all 5 value calculated
                    stock_idx = rotated_stock_idx
                    cut_position = rotated_cut_position
                    reward = rotated_reward
                    pred_stock = rotated_pred_stock
                    pred_cut = rotated_pred_cut

                # Convert to int
                cut_position = cut_position.to(dtype=torch.int32)
                    
                # Exploration vs. Exploitation
                if random.random() < self.epsilon:
                    # Greedy action for exploration
                    target_stock, target_cut = self.get_greedy_action(observation, product)
                    target_stock = torch.tensor([target_stock]).repeat(100)
                    target_cut = torch.tensor(target_cut).unsqueeze(0).repeat(100, 1).float()

                    # Loss calculation
                    stock_loss = F.cross_entropy(pred_stock, target_stock)
                    cut_loss = F.mse_loss(pred_cut, target_cut)
                    loss = stock_loss + cut_loss

                    # Backpropagation and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Decay epsilon if the action was successful. Average good move is ~10, we take 70% to mark it successful
                    if reward > 7:
                        self.epsilon *= self.decay_rate
                    
                break

            return {"stock_idx": stock_idx, "size": placed_product['size'], "position": cut_position}
        else:
            # Get current stock
            current_stock = observation["stocks"][self.current_stock_index]
            
            # Initialize products tracking for current stock
            if self.current_stock_index not in self.stock_placements:
                self.stock_placements[self.current_stock_index] = []
            
            # Find best placement for current stock
            placement = self.find_best_placement(current_stock, observation)
            #print(placement)
            # If placement found, return it
            if placement:
                #print("chay")
                return {
                    "stock_idx": self.current_stock_index,
                    "size": placement['size'],
                    "position": placement['position']
                }
            else:
                    self.current_stock_index += 1
                
            
            # If no more placements possible in current stock
            #if self.is_stock_fully_utilized(current_stock, self.stock_placements[self.current_stock_index]):
                # Move to next stock
                
                #self.current_stock_index += 1
                # If all stocks processed, return no placement
            if self.current_stock_index >= len(observation["stocks"]):
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def get_greedy_action(self, observation, product):
        """
        Greedy approach for selecting stock index and cut position for a single product.
        """
        # Logic to find the best greedy placement
        stock_idx = -1
        pos_x, pos_y = 0, 0
        prod_size = product["size"]

        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            return i, (x, y)

        return stock_idx, (pos_x, pos_y)

    def evaluate_action(self, stock_matrix, product_size, stock_idx, cut_position):
        """
        Evaluate the reward for a given action, with reversed logic (1 = free space, 0 = occupied space).
        
        Parameters:
        - stock_matrix: Tensor of current stock states [num_stocks, stock_size, stock_size].
        - product_size: Tuple representing the product size (width, height).
        - stock_idx: Chosen stock index.
        - cut_position: Tuple of (x, y) for the cut position.
        
        Returns:
        - reward: Scalar reward value for the action.
        """
        product_w, product_h = product_size
        product_w = int(product_w.item())
        product_h = int(product_h.item())
        stock = stock_matrix[stock_idx]  # Selected stock
        stock_h, stock_w = stock.shape

        x, y = cut_position
        x = int(x.item())
        y = int(y.item())
        reward = 0

        # Check if placement is within bounds
        if x + product_w > stock_w or y + product_h > stock_h:
            return -10  # Flat penalty for out-of-bounds placement

        # Extract the stock area where the product will be placed
        placement_area = stock[y:y+product_h, x:x+product_w]

        # Calculate overlap (this time overlap is when the space is occupied, i.e., == 0 in reversed logic)
        overlap_area = (placement_area == 0).sum().item()  # overlap if there's 0 (occupied space)
        product_area = product_w * product_h

        # Calculate free space (this time free space is when the space is free, i.e., == 1)
        free_space = (placement_area == 1).sum().item()  # free space if there's 1 (free space)

        # Check for invalid placement (overlap or insufficient space)
        if overlap_area > 0 or free_space < product_area:
            reward += -5 * overlap_area  # Overlap penalty
            reward += -10  # Flat penalty for invalid placement
            return reward

        # Reward for successful placement
        placed_area = min(free_space, product_area)
        reward += (placed_area / product_area) * 10  # Base reward for successful placement

        # Continuity reward (incentivize filling one stock first)
        stock_utilization = stock.sum().item() / (stock_h * stock_w)
        reward += stock_utilization * 2  # Continuity bonus

        return reward
