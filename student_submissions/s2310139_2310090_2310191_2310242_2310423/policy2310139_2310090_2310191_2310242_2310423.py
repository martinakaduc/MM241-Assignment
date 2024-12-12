from policy import Policy
from math import floor, ceil
from copy import deepcopy
from random import randint, shuffle, random, choice, choices
import time
import numpy as np

class Policy2310139_2310090_2310191_2310242_2310423(Policy):
    def __init__(self, policy_id=1, populationSize=300, penalty=2, mutationRate=0.1):
        """
        Initialize the Policy based on the given policy_id.
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if self.policy_id == 1:
            # Parameters for Genetic Algorithm (your original implementation)
            self.MAX_ITERATIONS = 2000
            self.POPULATION_SIZE = populationSize
            self.penalty = penalty
            self.mutationRate = mutationRate
            self.stockLength = 0
            self.stockWidth = 0
            self.lengthArr = []
            self.widthArr = []
            self.demandArr = []
            self.N = None
        if policy_id == 2:
            self.stockLength = 0
            self.stockWidth = 0
            self.lengthArr = []
            self.widthArr = []
            self.demandArr = []
            self.demand = [0]  # Demand should be a list
            self.dual_prices = []  # Initialize dual prices after item_sizes
            self.columns = []  # Store generated columns
            self.column_frequencies = {}
            self.stock_idx = 0
            
#----------------------------------------------------------------------------------------------------------------------------
    # Helper functions for the Genetic Algorithm
#----------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------
    #Part 1(Nguyễn Hoàng Gia Bảo)
    #------------------------------------------------------------------------------------------------------------------------
    def generate_efficient_patterns(self):
        patterns = []
        stack = [([0] * self.N, 0, 0)]

        while stack:
            current_pattern, length_used, width_used = stack.pop()

            for i in range(self.N):
                max_repeat = min(
                    (self.stockLength - length_used) // self.lengthArr[i],
                    (self.stockWidth - width_used) // self.widthArr[i],
                    self.demandArr[i]
                )
                if max_repeat > 0:
                    new_pattern = current_pattern.copy()
                    new_pattern[i] += max_repeat
                    patterns.append(new_pattern)
                    stack.append((new_pattern, length_used + max_repeat * self.lengthArr[i],
                                  width_used + max_repeat * self.widthArr[i]))

        return patterns

    def calculate_max_pattern_repetition(self, patternsArr):
        """
            Tính số lần tối đa mỗi mẫu (pattern) có thể được sử dụng.

            Tham số:
            - patternsArr: Danh sách các mẫu.

            Kết quả:
            - Trả về danh sách số lần tối đa mỗi mẫu có thể lặp lại.
            """
        result = [] # Danh sách lưu trữ số lần lặp tối đa cho từng mẫu
        for pattern in patternsArr:
                maxRep = 0
                for i in range(len(pattern)):
                    if pattern[i] > 0:
                        # Tính số lần cần lặp để đáp ứng nhu cầu
                        neededRep = ceil(self.demandArr[i] / pattern[i])
                        if neededRep > maxRep:
                            maxRep = neededRep
                result.append(maxRep)
        return result


    def initialize_population(self, maxRepeatArr):
        # initPopulation = []
        # for _ in range(self.POPULATION_SIZE):
        #     chromosome = []
        #     indices = list(range(len(maxRepeatArr)))
        #     shuffle(indices)
        #     for idx in indices:
        #         chromosome.append(idx)
        #         chromosome.append(max(1, maxRepeatArr[idx]))
        #     initPopulation.append(chromosome)
        # return initPopulation
        initPopulation = []
        for _ in range(self.POPULATION_SIZE):
            chromosome = []
            for i in np.argsort(-np.array(self.lengthArr) * np.array(self.widthArr)):  # Sắp xếp mẫu lớn trước
                chromosome.append(i)
                chromosome.append(randint(1, maxRepeatArr[i]))
            initPopulation.append(chromosome)
        return initPopulation

    #------------------------------------------------------------------------------------------------------------------------
    #Part 2 (Võ Duy Ân)
    #------------------------------------------------------------------------------------------------------------------------
    def evaluate_fitness(self, chromosome, patterns_arr):
        P = self.penalty
        unsupplied_sum = 0
        provided = [0] * self.N
        total_unused_area = 0  # Track unused area

        if self.stockLength == 0 or self.stockWidth == 0:
            raise ValueError("Stock dimensions (length or width) are not properly initialized.")

        stock_area = self.stockLength * self.stockWidth
        if stock_area == 0:
            stock_area = 1  # Fallback to prevent division by zero

        for i in range(0, len(chromosome), 2):
            pattern_index = chromosome[i]
            repetition = chromosome[i + 1]
            pattern = patterns_arr[pattern_index]

            for j in range(len(pattern)):
                provided[j] += pattern[j] * repetition

            # Simulate stock usage for unused area calculation
            pattern_area = sum(
                pattern[j] * self.lengthArr[j] * self.widthArr[j] for j in range(len(pattern))
            )
            total_unused_area += stock_area - pattern_area * repetition

        for i in range(self.N):
            unsupplied = max(0, self.demandArr[i] - provided[i])
            unsupplied_sum += unsupplied * self.lengthArr[i] * self.widthArr[i]

        fitness = (
            0.7 * (1 - total_unused_area / stock_area)  # Prioritize material usage
            - 0.3 * (P * unsupplied_sum / sum(self.demandArr))  # Penalize unsupplied products proportionally
        )

        return fitness

    def run(self, population, patterns_arr, max_repeat_arr, problem_path, queue=None):
        """
        Run the Genetic Algorithm to solve the 2D cutting stock problem.

        Parameters:
        - population: Initial population of chromosomes.
        - patterns_arr: List of feasible patterns.
        - max_repeat_arr: Maximum repetitions for each pattern.
        - problem_path: Problem instance file path (not used directly here).
        - queue: Optional queue for communication during execution.

        Returns:
        - Best solution, its fitness, fitness history, and execution time.
        """
        start_time = time.time()
        best_results = []
        num_iters_same_result = 0
        last_result = float('inf')

        for count in range(self.MAX_ITERATIONS):
            # Evaluate fitness for the population
            fitness_pairs = [(ch, self.evaluate_fitness(ch, patterns_arr)) for ch in population]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)

            # Track the best result
            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)

            # Convergence check
            if abs(best_fitness - last_result) < 1e-5:
                num_iters_same_result += 1
            else:
                num_iters_same_result = 0
            last_result = best_fitness

            # Early termination if converged
            if num_iters_same_result >= 100 or best_fitness == 1:
                break

            # Preserve top 3 (elitism)
            next_generation = [fitness_pairs[i][0] for i in range(3)]

            # Create new population
            while len(next_generation) < self.POPULATION_SIZE:
                parent1 = None
                parent2 = None

                try:
                    if random() < 0.5:
                        parent1 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                        parent2 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                    else:
                        parent1 = self.select_parents2([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                        parent2 = self.select_parents2([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                except ValueError as e:
                    # Fallback to random selection if selection fails
                    parent1 = choice([fp[0] for fp in fitness_pairs])
                    parent2 = choice([fp[0] for fp in fitness_pairs])

                # Perform crossover and mutation
                if parent1 and parent2:
                    child1 = self.mutate(self.crossover(parent1, parent2), self.mutationRate, max_repeat_arr)
                    child2 = self.mutate(self.crossover(parent2, parent1), self.mutationRate, max_repeat_arr)
                    next_generation.extend([child1, child2])


            # Update population for the next iteration
            population = deepcopy(next_generation[:self.POPULATION_SIZE])

            # Communicate progress if using a queue
            if queue is not None:
                queue.put((count, best_solution, best_fitness, time.time() - start_time))

        end_time = time.time()

        return best_solution, best_fitness, best_results, end_time - start_time


    #------------------------------------------------------------------------------------------------------------------------
    #Part 3 (Phạm Duy Anh)
    #------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def select_parents1(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:  # Handle zero fitness case
            return choice(population)

        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        return choices(population, probabilities)[0]

    @staticmethod
    def select_parents2(population, fitness_scores, tournament_size = 5):
        """
        Lựa chọn cha mẹ bằng phương pháp Tournament.

        Tham số:
        - population: Quần thể hiện tại.
        - fitness_scores: Điểm fitness của các cá thể trong quần thể.
        - tournament_size: Kích thước nhóm trong tournament.

        Kết quả:
        - Trả về một cá thể được chọn làm cha/mẹ.
        """
        indices = choices(range(len(population)), k=tournament_size)
        tournament = [population[i] for i in indices]
        tournament_scores = [fitness_scores[i] for i in indices]

        # Tìm cá thể tốt nhất trong nhóm tournament
        best_index = tournament_scores.index(max(tournament_scores))
        return tournament[best_index]
    
    @staticmethod
    def crossover(parent1, parent2):
        """
        Lai ghép hai cá thể để tạo ra một cá thể con mới.

        Tham số:
        - parent1: Cá thể cha.
        - parent2: Cá thể mẹ.

        Kết quả:
        - Trả về một cá thể con.
        """
        if parent1 is None or parent2 is None:
            raise ValueError("Parents must not be None")

        child = []
        for i in range(len(parent1)):
            # Chọn gen từ cha hoặc mẹ
            if random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    @staticmethod
    def mutate(chromosome, mutation_rate, max_repeat_arr):
        """
        Thực hiện đột biến ngẫu nhiên trên một cá thể.

        Tham số:
        - chromosome: Cá thể cần đột biến.
        - mutation_rate: Xác suất đột biến.
        - max_repeat_arr: Giới hạn số lần lặp tối đa của từng mẫu.

        Kết quả:
        - Trả về cá thể sau khi đột biến.
        """
        mutated_chromosome = chromosome[:]
        for i in range(0, len(chromosome), 2):  # Xét từng cặp (pattern_index, repetition)
            if random() < mutation_rate and i + 1 < len(chromosome) :
                # Thay đổi số lần lặp trong giới hạn
                pattern_index = mutated_chromosome[i]
                mutated_chromosome[i+1] = randint(1, max_repeat_arr[pattern_index])
        return mutated_chromosome
    
    def select_new_population(self,population, fitness_scores, patterns_arr, mutation_rate, max_repeat_arr, selection_type="tournament"):
        """
        Tạo quần thể mới bằng cách chọn lọc, lai ghép và đột biến.

        Tham số:
        - population: Quần thể hiện tại.
        - fitness_scores: Điểm fitness của các cá thể trong quần thể.
        - patterns_arr: Danh sách các mẫu khả thi.
        - mutation_rate: Xác suất đột biến.
        - max_repeat_arr: Số lần lặp tối đa cho từng mẫu.
        - selection_type: Phương pháp chọn lọc ('tournament' hoặc 'roulette').

        Kết quả:
        - Trả về quần thể mới.
        """
        new_population = []
        for _ in range(len(population) // 2):  # Số cặp cha mẹ
            # Chọn cha mẹ
            if selection_type == "tournament":
                parent1 = self.select_parents1(population, fitness_scores)
                parent2 = self.select_parents1(population, fitness_scores)
            elif selection_type == "roulette":
                parent1 = self.select_parents2(population, fitness_scores)
                parent2 = self.select_parents2(population, fitness_scores)

            # Lai ghép
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)

            # Đột biến
            child1 = self.mutate(child1, mutation_rate, max_repeat_arr)
            child2 = self.mutate(child2, mutation_rate, max_repeat_arr)

            # Thêm cá thể con vào quần thể mới
            new_population.extend([child1, child2])

        return new_population

#----------------------------------------------------------------------------------------------------------------------------
    #helper functions for the column generation algorithm
#----------------------------------------------------------------------------------------------------------------------------

    def solve_subproblem(self):
        """
        Solve the subproblem to generate a new cutting pattern.
        """
        if not self.lengthArr or not self.widthArr or not self.dual_prices:
            return (0, 0)

        max_reduced_cost = float('-inf')
        best_cut = None

        for i in range(len(self.lengthArr)):
            width, height = self.lengthArr[i], self.widthArr[i]
            if width <= self.stockLength and height <= self.stockWidth:
                reduced_cost = self.dual_prices[i] - 1
                if reduced_cost > max_reduced_cost:
                    max_reduced_cost = reduced_cost
                    best_cut = (width, height)

        return best_cut if max_reduced_cost > 0 else (0, 0)



    def update_dual_prices(self):
        """
        Update the dual prices based on demand satisfaction.
        """
        for i in range(len(self.lengthArr)):
            size = (self.lengthArr[i], self.widthArr[i])
            usage = self.column_frequencies.get(size, 0)
            demand = self.demandArr[i]
            adjustment = (demand - usage) / (demand + 1) if demand > 0 else 0
            self.dual_prices[i] = max(0, self.dual_prices[i] * (1 + adjustment))


    def is_pattern_feasible(self, pattern):
        """
        Check if a cutting pattern is feasible within stock dimensions.
        """
        total_length = sum(pattern[i] * self.lengthArr[i] for i in range(len(pattern)))
        total_width = sum(pattern[i] * self.widthArr[i] for i in range(len(pattern)))
        return total_length <= self.stockLength and total_width <= self.stockWidth


    def generate_efficient_patterns(self):
        """
        Generate cutting patterns based on stock size and item dimensions.

        Returns:
        - List of efficient cutting patterns.
        """
        patterns = []
        for i in range(len(self.lengthArr)):
            for j in range(len(self.widthArr)):
                # Check if item fits within stock dimensions
                if self.lengthArr[i] <= self.stockLength and self.widthArr[j] <= self.stockWidth:
                    pattern = [0] * len(self.lengthArr)
                    pattern[i] += 1  # Increment the pattern count for this item
                    patterns.append(pattern)
        return patterns

    def calculate_max_pattern_repetition(self, patternsArr):
        """
        Calculate the maximum repetitions for each pattern based on demand.

        Args:
        - patternsArr: List of cutting patterns.

        Returns:
        - List of maximum repetitions for each pattern.
        """
        max_repetitions = []
        for pattern in patternsArr:
            repetitions = []
            for i, count in enumerate(pattern):
                if count > 0:
                    # Calculate how many times this pattern can be repeated
                    repetitions.append(self.demandArr[i] // count)
            max_repetitions.append(min(repetitions) if repetitions else 0)
        return max_repetitions


    def get_action(self, observation, info):
            """
            Entry point for decision making based on the observation.
            """
            if self.policy_id == 1:
                # Use the existing implementation for Genetic Algorithm
                list_prods = observation["products"]
                stocks = observation["stocks"]

                if not stocks or not list_prods:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

                self.lengthArr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
                self.widthArr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
                self.demandArr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
                self.N = len(self.lengthArr)

                if self.N == 0:
                    return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

                first_stock = stocks[0]
                self.stockLength, self.stockWidth = self._get_stock_size_(first_stock)

                patterns_arr = self.generate_efficient_patterns()
                max_repeat_arr = self.calculate_max_pattern_repetition(patterns_arr)
                population = self.initialize_population(max_repeat_arr)

                best_solution, _, _, _ = self.run(population, patterns_arr, max_repeat_arr,None)

                for i in range(0, len(best_solution), 2):
                    pattern_index = best_solution[i]
                    repetition = best_solution[i + 1]
                    pattern = patterns_arr[pattern_index]

                    for stock_idx, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        for x in range(stock_w):
                            for y in range(stock_h):
                                if pattern_index >= len(self.lengthArr):
                                    continue
                                prod_size = (self.lengthArr[pattern_index], self.widthArr[pattern_index])
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            if self.policy_id == 2:
        # Initialization
                if not self.dual_prices or not self.lengthArr or not self.widthArr or not self.demandArr:
                    # self.stockLength = observation.get("stock_length", 100)
                    # self.stockWidth = observation.get("stock_width", 100)
                    self.lengthArr = [prod["size"][0] for prod in observation.get("products", [])]
                    self.widthArr = [prod["size"][1] for prod in observation.get("products", [])]
                    self.demandArr = [prod["quantity"] for prod in observation.get("products", [])]
                    self.demand = self.demandArr[:]
                    self.dual_prices = [1.0] * len(self.lengthArr)
                    self.columns = []
                    self.column_frequencies = {}
                list_prods = observation["products"]
                stocks = observation["stocks"]
                first_stock = stocks[0]
                self.stockLength, self.stockWidth = self._get_stock_size_(first_stock)
                # Generate patterns and calculate repetitions
                patterns = self.generate_efficient_patterns()
                max_repetitions = self.calculate_max_pattern_repetition(patterns)

                # Select the best pattern
                for stock_idx in range(len(observation.get("stocks", []))):
                    stock = observation["stocks"][stock_idx]
                    for pattern, max_rep in zip(patterns, max_repetitions):
                        if max_rep > 0:
                            for x in range(self.stockLength):
                                for y in range(self.stockWidth):
                                    feasible = True
                                    for i, count in enumerate(pattern):
                                        if count > 0:
                                            prod_size = (self.lengthArr[i], self.widthArr[i])
                                            if not self._can_place_(stock, (x, y), prod_size):
                                                feasible = False
                                                break

                                    if feasible:
                                        # Update usage and dual prices
                                        for i, count in enumerate(pattern):
                                            if count > 0:
                                                self.demandArr[i] -= count
                                                self.column_frequencies[(self.lengthArr[i], self.widthArr[i])] = \
                                                    self.column_frequencies.get((self.lengthArr[i], self.widthArr[i]), 0) + count

                                        # Update dual prices and track the column
                                        self.update_dual_prices()
                                        self.columns.append(pattern)

                                        return {
                                            "stock_idx": stock_idx,
                                            "size": prod_size,  # Use the actual product size
                                            "position": (x, y),
                                        }

                # No feasible placement found
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}




