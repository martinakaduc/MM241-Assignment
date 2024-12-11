import random
import numpy as np
from policy import Policy


class Policy2312562_213826_2213960_2213416(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.init = False
            self.index = 0
            self.n_population = 30
            self.generation = 25
            

    def get_action(self, observation, info):
        # Student code here
        if  self.policy_id == 2 and self.init == False:
            self.stocks = observation["stocks"]
            self.products = observation["products"]
            self.population = self.initialize_population([product.copy() for product in self.products], [np.copy(stock) for stock in self.stocks])
            print("population initialized!")
            for i in range(self.generation):
                self.population = self.crossover(self.population)
                print(f"generation {i+1} generated!")
            self.solution = self.population[0]
            self.nb_products = len(self.solution)
            self.init = True

        if self.policy_id == 1:
            return self.ffd_get_action(observation, info)
        
        if self.policy_id == 2 and self.init == True:
            action = self.ga_get_action(self.index)
            if self.index == self.nb_products - 1:
                self.index = 0
                self.init = False
            else:
                self.index += 1
            return action
    def ffd_get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        # 1. Sắp xếp danh sách các sản phẩm theo diện tích giảm dần
        
        sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        # 2. Duyệt qua từng sản phẩm trong danh sách đã được sắp xếp
        for product in sorted_products:
            size = product["size"]
            prod_w, prod_h = size
            product_size = prod_w * prod_h
            if product["quantity"] == 0:
                continue

            pos_x, pos_y = None, None
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), size):
                               pos_x, pos_y = x, y
                               break
                        if pos_x is not None and pos_y is not None:
                                break
                    if pos_x is not None and pos_y is not None:  
                        return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}
                
                if stock_w >= prod_h and stock_h >= prod_w:
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), size[::-1]):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        return {"stock_idx": stock_idx, "size": size[::-1], "position": (pos_x, pos_y)}


        # Nếu không còn sản phẩm nào để xử lý, kết thúc
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    # Student code here
    # You can add more functions if needed
#---------------------------------- GENETIC ALGORITHM--------------------------------------------------------------#
# check vị trí trống trong stock
    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1) 
    
    # đánh dấu đã đặt vào vị trí trên stock
    def _place_rectangle_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        for x in range(pos_x, pos_x + prod_w):
            for y in range(pos_y, pos_y + prod_h):
                stock[x, y] = 1
    
    # làm rỗng vị trí trên stock
    def _remove_rectangle_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = -1

    def get_stock_size(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h
    
    # khởi tạo một giải pháp
    def initial_individual(self, products, stocks):

        placement = []

        for prod in products:
            prod_quantity = prod["quantity"]
            prod_size = prod["size"]
            for _ in range(prod_quantity):
                pos_x, pos_y = None, None
                for _ in range(100):
                    stock_idx = random.randint(0, len(stocks) - 1)
                    stock = stocks[stock_idx]

                    stock_w, stock_h = self.get_stock_size(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        if stock_w >= prod_h or stock_h >= prod_w:
                            print("quay tay")
                            prod_size = prod_size[::-1]
                        else: 
                            continue

                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)

                    if not self._can_place_(stock, (pos_x, pos_y), prod_size):
                        continue
                    else:
                        placement.append((stock_idx, prod_size, (pos_x, pos_y)))
                        self._place_rectangle_(stock, (pos_x, pos_y), prod_size)
                        break
        
        return placement
    def ga_get_action(self, index):
        stock_idx, prod_size, position = self.solution[index]
        return {"stock_idx": stock_idx, "size": prod_size, "position": position}
    
    #tạo một quần thể chứa các solutions
    def initialize_population(self, products, stocks):
        population = []
        initial_solution = self.initial_individual([product.copy() for product in self.products], [np.copy(stock) for stock in stocks])
        population.append(initial_solution)
        for _ in range(self.n_population - 1):
            new_solution = self.initial_individual([product.copy() for product in self.products], [np.copy(stock) for stock in stocks])
            population.append(new_solution)

        sorted_population = sorted(population, key=self.evaluate_fitness)
        return sorted_population
        
    def evaluate_fitness(self, solution):
        used_stock = set(action[0] for action in solution)
        return len(used_stock)

    def repair_solution(self, solution, stocks):
        repaired_solution = []
        products = [product.copy() for product in self.products]
        # thêm các sản phẩm trong solution vào stock
        for action in solution:
            stock_idx, size, pos = action
            #check sản phẩm có sử dụng được hay không
            for prod in products:
                if np.array_equal(prod["size"], size) and prod["quantity"] > 0:
                    if self._can_place_(stocks[stock_idx], pos, size):
                        self._place_rectangle_(stocks[stock_idx], pos, size)
                        repaired_solution.append((stock_idx, size, pos))
                        prod["quantity"] -= 1
                        break

        # Bước 2: Thêm các sản phẩm bị thiếu (First-Fit)
        sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        for prod in sorted_products:
            quantity = prod["quantity"]
            prod_size = prod["size"]

            if quantity <= 0:
                continue

            prod_w, prod_h = prod_size
            for _ in range(quantity):
                for stock_idx, stock in enumerate(stocks):
                    placed = False  # Đánh dấu nếu đã đặt thành công
                    stock_w, stock_h = self.get_stock_size(stock)
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Duyệt các vị trí hợp lệ (First-Fit)
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                self._place_rectangle_(stock, (x, y), prod_size)
                                repaired_solution.append((stock_idx, prod_size, (x, y)))
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

        return repaired_solution


    def crossover(self, population):
        elite_population = population[:max(1, len(population) // 5)]
        new_population = []
        new_population.extend(elite_population)
        
        for i in range(self.n_population - len(new_population)):
            parent1 = random.choice(elite_population)
            parent2 = random.choice(elite_population)

            while np.array_equal(parent1, parent2):
                parent2 = random.choice(elite_population)
            # Lai ghép bằng cách kết hợp các action từ cả hai cha mẹ
            child_solution = []
            child_solution = self.two_point_crossover(parent1, parent2)
            child_solution = self.random_walk_mutate(child_solution, len(child_solution) // 10)

            # Sửa lỗi nếu có chồng lấn
            repaired_solution = self.repair_solution(child_solution, [np.copy(stock) for stock in self.stocks])

            new_population.append(repaired_solution)

        # Sắp xếp lại quần thể theo fitness
        new_population = sorted(new_population, key=self.evaluate_fitness)
        return new_population
    def two_point_crossover(self, parent1, parent2):
        # Chọn hai điểm cắt ngẫu nhiên
        length = len(parent1)
        if length < 2:
            return parent1  # Không đủ phần tử để lai ghép

        cut1 = random.randint(0, length - 1)
        cut2 = random.randint(cut1, length - 1)

        # Tạo con
        child = parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:]  

        return child
    def random_walk_mutate(self, solution, num_products_to_remove=10):
   
        # Tạo bản sao của giải pháp và stocks để sửa đổi
        if random.random() > 0.1:
            return solution
        mutated_solution = []
        mutated_solution.extend(solution)
        stock_copies = [np.copy(stock) for stock in self.stocks]
        for action in solution:
            stock_idx, size, pos = action
            self._place_rectangle_(stock_copies[stock_idx], pos, size)
    
        # Tháo gỡ các sản phẩm trong solution
        removed_products = []
        for _ in range(num_products_to_remove):
            if not mutated_solution:
                break  # Không còn sản phẩm để tháo gỡ
            idx = random.randint(0, len(mutated_solution) - 1 - num_products_to_remove)
            stock_idx, prod_size, position = mutated_solution[idx]
            mutated_solution.pop(idx)
            self._remove_rectangle_(stock_copies[stock_idx], position, prod_size)
            removed_products.append(prod_size)
    
        # Lắp lại các sản phẩm bị tháo gỡ theo chiến lược First-Fit
        sorted_products = sorted(removed_products, key=lambda p: p[0] * p[1], reverse=True)
        for prod_size in sorted_products:
            prod_w, prod_h = prod_size
            placed = False
            for stock_idx, stock in enumerate(stock_copies):
                stock_w, stock_h = self.get_stock_size(stock)
                
                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), size):
                                self._place_rectangle_(stock, (x, y), prod_size)
                                mutated_solution.append((stock_idx, prod_size, (x, y)))
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
                
                if stock_w >= prod_h and stock_h >= prod_w:
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                self._place_rectangle_(stock, (x, y), prod_size[::-1])
                                mutated_solution.append((stock_idx, prod_size[::-1], (x, y)))
                                placed = True
                                break
                        if placed:
                            break

                    if placed:
                        break

        return mutated_solution
