from policy import Policy
import numpy as np
import time

class Policy2311671_2311815_2311972_2312738_2311660(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Queue to store actions
        self.action_queue = []
        self.used_stocks = []  # Tracks which stocks are used
        self.largest_stock_idx = -1  # Cache for the current largest stock
        self.total_product_num = 0
        self.acceptable_ratio = 0.9
        self.id_policy = policy_id

        self.discount_factor = 0.95
        self.theta = 1e-4
        self.max_iterations = 1000
        self.timeout = 20
        self.value_function = {}  # To store the value of each state
        self.sorted_stocks = []  # To store the sorted stocks

        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.id_policy == 1:
            return self.get_action1(observation, info)
        elif self.id_policy == 2:
            return self.get_action3(observation, info)
        pass

    def get_action1(self, observation, info):
        if self.action_queue:
            return self.action_queue.pop(0)

        self.get_action2(observation, info)
        if not self.action_queue:
            return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
        
        return self.action_queue.pop(0)

    def get_action2(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        
        if not self.used_stocks:
            self.used_stocks = [0] * len(stocks) # 0 = unused, 1 = used, 2 = not good enough

        # Step 1: Find the largest unused stock
        max_area = 0
        self.largest_stock_idx = -1
        for i, stock in enumerate(stocks):
            if self.used_stocks[i] == 0:
                stock_w, stock_h = self._get_stock_size_(stock)
                stock_area = stock_w * stock_h
                if stock_area > max_area:
                    max_area = stock_area
                    self.largest_stock_idx = i

        if self.largest_stock_idx == -1:
            self.acceptable_ratio -= 0.1
            if self.acceptable_ratio < 0.4:
                for i, stock in enumerate(stocks):
                    if self.used_stocks[i] == 2:
                        self.used_stocks[i] = 0
            for i, stock in enumerate(stocks):
                if self.used_stocks[i] == 2:
                    self.used_stocks[i] = 0
            self.get_action2(observation, info)

        if self.largest_stock_idx == -1:
            return  # No stocks available

        stock = stocks[self.largest_stock_idx]
        tmp_stock = np.copy(stock)
        # Step 2: Perform branch-and-bound to find the configuration with the least waste
        best_waste = np.sum(stock == -1)
        best_queue = []
        # Try with all products first
        original_quantities = [p["quantity"] for p in products]
        self.action_queue = []
        waste_all = self._place_products_and_calculate_waste(tmp_stock, products)
        best_waste = waste_all
        best_queue = self.action_queue.copy()
        temp_best = best_waste + 1
        used_products = self.calc_used(products, original_quantities)
        self.total_product_num = 0
        for ori in original_quantities:
            self.total_product_num += ori

        if self.action_queue.__len__() == self.total_product_num:
            self.action_queue = []
            self._reset_product_quantities(products, original_quantities)
            self._place_last_stock_(observation, original_quantities, info)
            return
        # Reset quantities and try skipping one product at a time
        while temp_best > best_waste:
            temp_best = best_waste
            for i in range(len(products)):
                self._reset_product_quantities(products, original_quantities)
                tmp_stock = np.copy(stock)  # Reset stock
                if used_products[i] == 0:
                    continue  # Skip products that are not used
                products[i]["quantity"] = used_products[i] - 1
                self.action_queue = []
                waste_partial = self._place_products_and_calculate_waste(tmp_stock, products)
                if waste_partial < best_waste:
                    best_waste = waste_partial
                    best_queue = self.action_queue.copy()
        self._reset_product_quantities(products, original_quantities)

        stock_area = self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
        if best_waste / stock_area > 1 - self.acceptable_ratio and self.used_stocks[self.largest_stock_idx] == 0:
            self.used_stocks[self.largest_stock_idx] = 2
            self.largest_stock_idx = -1
            self.action_queue = []
            self.get_action2(observation, info)
            return
        self.action_queue = best_queue
        self.used_stocks[self.largest_stock_idx] = 1
        self.largest_stock_idx = -1

    def _place_products_and_calculate_waste(self, stock, products):
        sorted_products = sorted(
            products,
            key=lambda p: p["size"][0] * p["size"][1],  # Sort by area (width * height)
            reverse=True
        )
        for product in sorted_products:
            while product["quantity"] > 0:
                placed = self._place_product_in_stock(stock, product, rotate=False)
                if not placed:
                    break  # Move to the next product if this one can't be placed anymore

        return self._calculate_waste(stock)

    def _place_product_in_stock(self, stock, product, rotate=False):

        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product["size"]
        if rotate:
            prod_w, prod_h = product["size"][::-1]
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):

                if np.all(stock[x : x + prod_w, y : y + prod_h] == -1):
                    stock[x : x + prod_w, y : y + prod_h] = 0
                    product["quantity"] -= 1

                    self.action_queue.append({
                        "stock_idx": self.largest_stock_idx,
                        "size": [prod_w, prod_h], 
                        "position": (x, y)
                    })
                    return True      
        if rotate:
            return False
        else:
            return self._place_product_in_stock(stock, product, rotate=True)

    def _calculate_waste(self, stock):
        return np.sum(stock == -1)
    
    def _calculate_fill(self, stock):
        return np.sum(stock == 0)

    def _reset_product_quantities(self, products, original_quantities):
        for i, quantity in enumerate(original_quantities):
            products[i]["quantity"] = quantity

    def calc_used(self, products, original_quantities):
        used_products = []
        for i in range(len(products)):
            used_products.append(original_quantities[i] - products[i]["quantity"])
        return used_products
    
    def _place_last_stock_(self, observation, original_quantities, info):
        stocks = observation["stocks"]
        products = observation["products"]
        area = 0
        for product in products:
            area += product["size"][0] * product["size"][1] * product["quantity"]
        valid_stocks = [
        (i, stock) for i, stock in enumerate(stocks)
        if self.used_stocks[i] != 1 and np.prod(self._get_stock_size_(stock)) >= area
        ]
        valid_stocks.sort(key=lambda s: np.prod(self._get_stock_size_(s[1])))
        for stock_idx, stock in valid_stocks:
            tmp_stock = np.copy(stock)
            self._reset_product_quantities(products, original_quantities)
            self.action_queue = []
            self.largest_stock_idx = stock_idx
            self._place_products_and_calculate_waste(tmp_stock, products)
            if self.action_queue.__len__() == self.total_product_num:
                self.used_stocks[stock_idx] = 1
                self._reset_product_quantities(products, original_quantities)
                self.used_stocks = []
                self.largest_stock_idx = -1
                return

    def update_sorted_stocks(self, observation):
        """
        Sort the products by size (area) in descending order.
        """
        self.sorted_stocks = sorted(
            enumerate(observation["products"]),  # Iterate over the products
            key=lambda x: x[1]["size"][0] * x[1]["size"][1],  # Sort by area (size)
            reverse=True  # Sort in descending order (largest area first)
        )

    def evaluate_policy(self, observation):
        """
        Policy evaluation: Update the value function based on the current policy.
        """
        iteration = 0
        while iteration < self.max_iterations:
            delta = 0
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_state = tuple(map(tuple, stock))  # Convert stock to a hashable state
                if stock_state not in self.value_function:
                    self.value_function[stock_state] = 0

                v = self.value_function[stock_state]
                new_value = 0
                for prod in observation["products"]:
                    if prod["quantity"] > 0:
                        reward = -prod["size"][0] * prod["size"][1]  # Negative space used as cost
                        prob = 1 / len(observation["products"])  # Assuming uniform probability
                        new_value += prob * (reward + self.discount_factor * v)

                self.value_function[stock_state] = new_value
                delta = max(delta, abs(v - new_value))

            iteration += 1
            if delta < self.theta:
                break

    def improve_policy(self, observation):
        best_action = None
        best_value = float('-inf')
        start = time.time()
        # print("Improving policy...")

        # Sort products by size (area), descending order (largest first)
        sorted_products = sorted(observation["products"], key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        stock_count = len(observation["stocks"])
        stock_chunks = [observation["stocks"][i:i + 2] for i in range(0, stock_count, 2)]
        for chunk_idx, chunk in enumerate(stock_chunks):
            # print(f"Processing stock chunk {chunk_idx + 1}/{len(stock_chunks)}...")
            for prod in sorted_products:  # Use the sorted list of products

                for stock_idx, stock in enumerate(chunk):
                    global_stock_idx = chunk_idx * 2 + stock_idx  # Global index in observation["stocks"]
                    stock_state = tuple(map(tuple, stock))
                    if stock_state not in self.value_function:
                        continue

                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        stock_w, stock_h = stock.shape

                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    reward = -prod_size[0] * prod_size[1]
                                    future_value = self.discount_factor * self.value_function.get(stock_state, 0)
                                    value = reward + future_value
                                    if value > best_value:
                                        best_value = value
                                        best_action = {"stock_idx": global_stock_idx, "size": prod_size, "position": (x, y)}
                                    
                                    if time.time() - start > self.timeout:
                                        return best_action
                                    
                if best_action:
                    return best_action
                    

        return best_action

    def get_action3(self, observation, info):
        """
        Get the next action based on policy evaluation and improvement.
        """
        self.update_sorted_stocks(observation)  # Sort stocks before processing
        self.evaluate_policy(observation)
        return self.improve_policy(observation)

    def _can_place_(self, stock, position, size):
        """
        Check if a product can be placed at the given position in the stock.
        """
        x, y = position
        w, h = size
        if x + w > stock.shape[0] or y + h > stock.shape[1]:
            return False
        return np.all(stock[x:x+w, y:y+h] == -1)
