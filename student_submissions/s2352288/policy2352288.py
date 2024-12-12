from policy import Policy
import numpy as np
import random
from scipy.optimize import linprog
from scipy.ndimage import label

class Policy2352288(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()
        self.policy_id = policy_id
        
        if policy_id == 1:
            self.strategy = self.AlbanoSapuppoAlgorithm()
        elif policy_id == 2:
            self.strategy = self.GeneticAlgorithm()

    def get_action(self, observation, info):
        return self.strategy.get_action(observation, info)

    class AlbanoSapuppoAlgorithm(Policy):
        def __init__(self):
            super().__init__()
            
        def get_action(self, observation, info):
            list_prods = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)
            stocks = observation["stocks"]

            best_stock_idx = -1
            best_position = None
            best_prod_size = None
            min_waste = float("inf")

            for prod in list_prods:
                if prod["quantity"] <= 0:
                    continue

                original_size = prod["size"]
                rotated_size = original_size[::-1]  # Xoay 90 độ (đổi chiều rộng và chiều cao)

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra trạng thái nguyên bản
                    candidate_positions = self._find_candidate_positions(stock, original_size, stock_w, stock_h)
                    for position in candidate_positions:
                        waste = self._compute_waste(stock, position, original_size)
                        if waste < min_waste:
                            min_waste = waste
                            best_stock_idx = stock_idx
                            best_position = position
                            best_prod_size = original_size

                    # Kiểm tra trạng thái xoay 90 độ
                    candidate_positions = self._find_candidate_positions(stock, rotated_size, stock_w, stock_h)
                    for position in candidate_positions:
                        waste = self._compute_waste(stock, position, rotated_size)
                        if waste < min_waste:
                            min_waste = waste
                            best_stock_idx = stock_idx
                            best_position = position
                            best_prod_size = rotated_size

            if best_stock_idx != -1:
                return {
                    "stock_idx": best_stock_idx,
                    "size": best_prod_size,
                    "position": best_position,
                }

            return None  # Không tìm được vị trí hợp lệ


        def _find_candidate_positions(self, stock, prod_size, stock_w, stock_h):
            """
            Tìm các vị trí ứng viên đơn giản (cạnh biên trái trên).
            """
            valid_positions = []
            prod_w, prod_h = prod_size

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    # Chỉ xét các vị trí ở góc trái trên hoặc cạnh biên
                    if (x == 0 or y == 0) and self._can_place_(stock, (x, y), prod_size):
                        valid_positions.append((x, y))

            return valid_positions

        def _compute_waste(self, stock, position, prod_size):
            """
            Tính toán vật liệu thừa khi đặt sản phẩm tại vị trí cụ thể.
            """
            pos_x, pos_y = position
            prod_w, prod_h = prod_size
            temp_stock = stock.copy()

            # Đặt sản phẩm tạm thời
            temp_stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = 0

            # Tính vật liệu thừa
            unused_area = np.sum(temp_stock == -1)
            return unused_area

        def _get_stock_size_(self, stock):
            """
            Lấy kích thước của stock (chiều rộng, chiều cao).
            """
            return stock.shape[1], stock.shape[0]

        def _can_place_(self, stock, position, prod_size):
            """
            Kiểm tra xem sản phẩm có thể đặt tại vị trí cụ thể không.
            """
            pos_x, pos_y = position
            prod_w, prod_h = prod_size

            # Kiểm tra kích thước vùng có đủ để đặt không
            if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
                return False

            # Kiểm tra vùng có bị chiếm bởi sản phẩm khác không
            return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)
    class GeneticAlgorithm(Policy):
        def __init__(self):
            super().__init__()
            self.beam_width = 25
            self.mutation_rate = 0.35
            self.crossover_rate = 0.95
            self.generations = 100
            self.used_stocks = set()
            
        def get_action(self, observation, info):
            list_prods = observation["products"]
        
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    beam = self._initialize_population(observation, self.beam_width)
                    
                    for gen in range(self.generations):
                        fitness_scores = self._evaluate_fitness(beam, observation, prod_size)
                        top_indices = np.argsort(fitness_scores)[-self.beam_width:]
                        beam = [beam[i] for i in top_indices]
                        
                        offspring = self._crossover(beam)
                        mutated_offspring = self._mutation(offspring, observation)
                        
                        combined_population = beam + mutated_offspring
                        fitness_scores = self._evaluate_fitness(combined_population, observation, prod_size)
                        top_indices = np.argsort(fitness_scores)[-self.beam_width:]
                        beam = [combined_population[i] for i in top_indices]
                    
                    best_solution = max(beam, key=lambda x: self._evaluate_individual(x, observation, prod_size))
                    stock_idx, pos_x, pos_y = best_solution
                    
                    if (stock_idx >= 0 and pos_x is not None and pos_y is not None and 
                        self._can_place_(observation["stocks"][stock_idx], (pos_x, pos_y), prod_size)):
                        self.used_stocks.add(stock_idx)
                        best_solution = self._local_search(best_solution, observation, prod_size)
                        stock_idx, pos_x, pos_y = best_solution
                        break
            
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        def _initialize_population(self, observation, size):
            stock_areas = [
                (i, self._get_stock_size_(observation["stocks"][i])[0] * self._get_stock_size_(observation["stocks"][i])[1])
                for i in range(len(observation["stocks"]))
            ]
            stock_areas.sort(key=lambda x: x[1], reverse=True)

            population = []
            for _ in range(size):
                for stock_idx, _ in stock_areas:
                    stock = observation["stocks"][stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    pos_x = np.random.randint(0, max(0, stock_w))
                    pos_y = np.random.randint(0, max(0, stock_h))
                    population.append((stock_idx, pos_x, pos_y))
                    if len(population) >= size:
                        break
                if len(population) >= size:
                    break
            return population

        def _evaluate_fitness(self, population, observation, prod_size):
            return [self._evaluate_individual(ind, observation, prod_size) for ind in population]
        
        def _evaluate_individual(self, individual, observation, prod_size):
            stock_idx, pos_x, pos_y = individual
            stock = observation["stocks"][stock_idx]
            if self._can_place_(stock, (pos_x, pos_y), prod_size):
                reuse_bonus = 10 if stock_idx in self.used_stocks else 0
                trim_loss_penalty = self._calculate_trim_loss(stock, (pos_x, pos_y), prod_size)
                return 1 + reuse_bonus - trim_loss_penalty
            return 0

        def _calculate_trim_loss(self, stock, position, prod_size):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size
            trim_loss = (stock_w * stock_h - prod_w * prod_h) / (stock_w * stock_h)
            return trim_loss

        def _crossover(self, beam):
            offspring = []
            for i in range(len(beam)):
                for j in range(i + 1, len(beam)):
                    if np.random.random() < self.crossover_rate:
                        p1, p2 = beam[i], beam[j]
                        point = np.random.randint(0, 2)
                        child1 = p1[:point] + p2[point:]
                        child2 = p2[:point] + p1[point:]
                        offspring.extend([child1, child2])
            return offspring

        def _mutation(self, population, observation):
            mutated = []
            stock_areas = [
                (i, self._get_stock_size_(observation["stocks"][i])[0] * self._get_stock_size_(observation["stocks"][i])[1])
                for i in range(len(observation["stocks"]))
            ]
            stock_areas.sort(key=lambda x: x[1], reverse=True)

            for ind in population:
                if np.random.random() < self.mutation_rate:
                    stock_idx, pos_x, pos_y = ind
                    component = np.random.randint(0, 2)
                    if component == 0:
                        stock_idx = stock_areas[np.random.randint(len(stock_areas))][0]
                    elif component == 1:
                        pos_x = max(0, pos_x + np.random.randint(-1, 2))
                    else:
                        pos_y = max(0, pos_y + np.random.randint(-1, 2))
                    mutated.append((stock_idx, pos_x, pos_y))
                else:
                    mutated.append(ind)
            return mutated

        def _local_search(self, solution, observation, prod_size):
            stock_idx, pos_x, pos_y = solution
            stock = observation["stocks"][stock_idx]
            best_solution = solution
            best_trim_loss = self._calculate_trim_loss(stock, (pos_x, pos_y), prod_size)
            
            for dx in range(-30, 31):  # Increased range for more thorough search
                for dy in range(-30, 31):
                    new_pos_x = pos_x + dx
                    new_pos_y = pos_y + dy
                    if self._can_place_(stock, (new_pos_x, new_pos_y), prod_size):
                        new_trim_loss = self._calculate_trim_loss(stock, (new_pos_x, new_pos_y), prod_size)
                        if new_trim_loss < best_trim_loss:
                            best_trim_loss = new_trim_loss
                            best_solution = (stock_idx, new_pos_x, new_pos_y)
            
            return best_solution
