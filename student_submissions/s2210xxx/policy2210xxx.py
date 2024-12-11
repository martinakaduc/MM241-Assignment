from math import fabs
from policy import Policy
import numpy as np


class Policy2210xxx(Policy):
    def __init__(self, policy_id):
        self.stock_products = []
        self.check = False
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            list_prods = list(observation["products"]) 
            list_prods.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            stocks = observation["stocks"]
            if not self.check:  
                self.stock_products = [[] for _ in range(len(stocks))]
                self.check = True  
                self.stocks = stocks[:] 
            
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    placed = False
                    
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        placed = True
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        prod_size = prod_size[::-1]
                                        placed = True
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                    if pos_x is not None and pos_y is not None:
                        if placed:
                            placed = False
                            product_found = False             
                            if self.stock_products[stock_idx]:
                                for product in self.stock_products[stock_idx]:
                                    if tuple(product['size']) == tuple(prod_size): 
                                        product['quantity'] += 1 
                                        product_found = True 
                            if not product_found:
                                self.stock_products[stock_idx].append({"size": prod_size, "quantity": 1})  
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            pass
        elif self.policy_id == 2:
            list_prods = list(observation["products"]) 
            list_prods.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            stocks = observation["stocks"]
            if not self.check:  
                self.stock_products = [[] for _ in range(len(stocks))]
                self.check = True  
                self.stocks = stocks[:] 

            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    best_fit = None
                    min_leftover_space = float('inf')

                    for stock_idx, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):                                    
                                        leftover_space = (stock_w - prod_w) * (stock_h - prod_h)
                                        if leftover_space < min_leftover_space:
                                            min_leftover_space = leftover_space
                                            best_fit = (stock_idx, (x, y))
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size):                                    
                                        leftover_space = (stock_w - prod_h) * (stock_h - prod_w)
                                        if leftover_space < min_leftover_space:
                                            print("xoay")
                                            min_leftover_space = leftover_space
                                            best_fit = (stock_idx, (x, y))          
                    if best_fit is not None:
                        stock_idx, (pos_x, pos_y) = best_fit
                        product_found = False 
                        if self.stock_products[stock_idx]:
                            for product in self.stock_products[stock_idx]:
                                if tuple(product['size']) == tuple(prod_size): 
                                    product['quantity'] += 1 
                                    product_found = True 
                                    break  

                        if not product_found:
                            self.stock_products[stock_idx].append({"size": prod_size, "quantity": 1}) 
                    break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            pass
    def print_stocks(self):
        print("First Stocks with Products:")

        
        for i, products in enumerate(self.stock_products):
            if len(products) > 0:
                stock = self.stocks[i]  
                stock_w, stock_h = self._get_stock_size_(stock)
                total_stock_area = stock_w * stock_h
                total_product_area = sum(prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in products)
                waste_size = total_stock_area - total_product_area

                print(f"Stock Index: {i}")
                print(f"Stock Size: {stock_h}x{stock_w}")
                print(f"Waste Size: {waste_size}")
                print("Products in this stock:")

                for prod in products:
                    prod_w, prod_h = prod["size"]
                    quantity = prod["quantity"]
                    print(f"  Product Size: {prod_w}x{prod_h}, Quantity: {quantity}")

                print("-" * 30)

