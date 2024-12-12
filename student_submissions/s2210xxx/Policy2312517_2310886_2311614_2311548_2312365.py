import numpy as np
from policy import Policy
import random


class Policy2312517_2310886_2311614_2311548_2312365(Policy):
    def __init__(self, policy_id=2):
        """
        Initialize the policy based on the given policy ID.
        - policy_id = 1: Genetic Policy
        - policy_id = 2: Max-Fill Area Policy
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Attributes for Genetic Policy
        if self.policy_id == 1:
            self.population_size = 10
            self.generations = 5
            self.crossover_rate = 0.8
            self.mutation_rate = 0.2
            self.elite_size = 2
            self.best_solution = []
            self.program_counter = 0

        # No specific initialization needed for Stock filled_rate optimize heuristic (policy_id = 2)

    def get_action(self, observation, info):
        """
        Decide which action function to use based on the policy_id.
        """
        if self.policy_id == 1:
            return self.genetic_get_action(observation, info)
        elif self.policy_id == 2:
            return self.max_fill_get_action(observation, info)

# -----------------------------------------------------------------------------
#                   STOCK_FILL_RATE_OPTIMIZATION_HEURISTIC
# -----------------------------------------------------------------------------

    def _get_empty_areas(self, stock):
        """
        Analyze the available empty areas in the stock and return a list of empty rectangular regions.
        """
        empty_areas = []
        visited = np.zeros_like(stock, dtype=bool)

        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                if stock[x, y] == -1 and not visited[x, y]:
                    # find largest rectangular starting from (x, y)
                    area_w, area_h = self._find_rectangle(stock, visited, x, y)
                    empty_areas.append((x, y, area_w, area_h))

        return empty_areas

    def _find_rectangle(self, stock, visited, start_x, start_y):
        """
        Find the largest empty rectangle starting from (start_x, start_y).
        """
        max_x = start_x
        max_y = start_y

        # Find width
        while max_x < stock.shape[0] and stock[max_x, start_y] == -1:
            max_x += 1

        # Find height
        while max_y < stock.shape[1] and np.all(stock[start_x:max_x, max_y] == -1):
            max_y += 1

        # marked as explored
        visited[start_x:max_x, start_y:max_y] = True

        return max_x - start_x, max_y - start_y

    def max_fill_get_action(self, observation, info):
        """
        return optimal action with best fill rate
        """
        stocks = observation["stocks"]
        products = observation["products"]

        best_action = None
        max_fill_rate = -1  # best fill rate

        # sort in decsending order of product area
        sorted_products = sorted(
            enumerate(products),
            key=lambda x: x[1]["size"][0] * x[1]["size"][1],
            reverse=True,
        )

        for product_idx, product in sorted_products:
            if product["quantity"] <= 0:
                continue

            size = product["size"]

            for stock_idx, stock in enumerate(stocks):
                stock_width, stock_height = self._get_stock_size_(stock)

                if size[0] > stock_width or size[1] > stock_height:
                    continue

                #analyze empty area on the stock
                empty_areas = self._get_empty_areas(stock)

                for area in empty_areas:
                    area_x, area_y, area_w, area_h = area

                    # Check if product can be placed on empty area
                    for width, height in [size, size[::-1]]:
                        if width <= area_w and height <= area_h:
                            fill_rate = (width * height) / (area_w * area_h)

                            # prioritize actio with highest fill rate
                            if fill_rate > max_fill_rate:
                                max_fill_rate = fill_rate
                                best_action = {
                                    "stock_idx": stock_idx,
                                    "size": np.array([width, height]),
                                    "position": np.array([area_x, area_y]),
                                }

        return best_action

# -----------------------------------------------------
#                     GENETIC POLICY
# -----------------------------------------------------

    class Individual:
        def __init__(self, actions, stock_usage, fitness):
            self.actions = actions  # List of actions (placements of products)
            self.stock_usage = stock_usage  # Matrix representing stock usage
            self.fitness = fitness  # Fitness score

    def genetic_get_action(self,observation,info):
         # If best_solution is empty, generate it
        if not self.best_solution and self.program_counter == 0:
            self.best_solution = self.gen_population(observation, info)
            self.program_counter+=1

        if self.best_solution:
          # Loop through the actions and check for overlap
          for action in self.best_solution:
            if self.is_valid_action(action, observation["stocks"]):
                return self.best_solution.pop(0)

        # Fallback if no actions are available
        if self.program_counter == 1:
          self.program_counter+=1
        return self.max_fill_get_action(observation, info)
  
    def gen_population(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # Step 1: Generate initial population for 100% of total product quantity
        population = self._generate_initial_population(products, stocks)

        # Step 2: Genetic Algorithm Evolution
        best_individual = None
        for gen in range(self.generations):
           population = self._evolve_population(population, products, stocks)
           best_individual = min(population, key=lambda ind: ind.fitness)

        # Step 3: Return the best solution's actions
        if best_individual and best_individual.actions:
          return best_individual.actions

        # Step 4: return None to activate Fallback mechanism if no best solution found for all product
        return None

    def _generate_initial_population(self, products, stocks):
        """
        Generate a more diverse initial population by introducing randomness into placement.
        """
        population = []
        for _ in range(self.population_size):
            # Generate initial population with some randomness
            actions, stock_usage = self._generate_heuristic_solution(products, stocks)
            fitness = self._evaluate_fitness(actions, stock_usage)
            population.append(self.Individual(actions, stock_usage, fitness))

        return population

    def _generate_heuristic_solution(self, products, stocks):
        """
        Generate a heuristic solution by placing all products in stocks using a Greedy strategy.
        """
        actions = []
        stock_usage = self.deep_copy_stock_usage(stocks)

        # Sort products by size (larger products first)
        products_sorted = sorted(products, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        for product in products_sorted:
            product_quantity = product["quantity"]

            for _ in range(product_quantity):
                placed = False
                prod_w, prod_h = product["size"]

                for stock_idx, stock in enumerate(stock_usage):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for _ in range(100):  # Try 100 random placements
                        pos_x = random.randint(0, max(0, stock_w - prod_w))
                        pos_y = random.randint(0, max(0, stock_h - prod_h))

                        if self._can_place_(stock, (pos_x, pos_y), product["size"]) :
                            actions.append({"stock_idx": stock_idx, "size": product["size"], "position": (pos_x, pos_y)})
                            stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = 0  # Mark as occupied
                            placed = True
                            break
                        # Try placing in rotated orientation
                        elif self._can_place_(stock, (pos_x, pos_y), (prod_h, prod_w)):
                           actions.append({"stock_idx": stock_idx, "size": (prod_h, prod_w), "position": (pos_x, pos_y)})
                           stock[pos_x:pos_x + prod_h, pos_y:pos_y + prod_w] = 0  # Mark as occupied
                           placed = True
                           break

                    if placed:
                        break

        return actions, stock_usage
    
    def _evaluate_fitness(self, actions, stock_usage):
      """
      Fitness is based on the number of unique stocks used, unused area, and wasted area.
      The goal is to minimize the unused space and wasted space within the used stocks.
      """
      used_stocks = set(a["stock_idx"] for a in actions)  # Get unique stock indices used
      unused_area = 0
      wasted_area = 0  # Area that is wasted due to inefficient placement

      for stock_idx in used_stocks:
        stock = stock_usage[stock_idx]
        # Calculate unused area (where stock == -1 means empty)
        unused_area += np.sum(stock == -1)
     
      # Fitness function: minimize the number of stocks used, penalize unused and wasted areas
      fitness =  unused_area
      return fitness

    def _evolve_population(self, population, products, stocks):
        """
        Perform selection, crossover, and mutation to evolve the population.
        """
        new_population = []

        # Elite Selection
        population_sorted = sorted(population, key=lambda ind: ind.fitness)
        new_population.extend(population_sorted[:self.elite_size])

        # Selection and Reproduction
        while len(new_population) < self.population_size:
            parent1 = self._rank_selection(population)
            parent2 = self._rank_selection(population)

            if random.random() < self.crossover_rate:
                child_actions = self._two_point_crossover(parent1.actions, parent2.actions)
            else:
                child_actions = parent1.actions

            if random.random() < self.mutation_rate:
                child_actions, stock_usage = self._mutate(child_actions, products, stocks)
            else:
                stock_usage = parent1.stock_usage
            
            purified_action = self.purify_solutions(child_actions)
            fitness = self._evaluate_fitness(purified_action, stock_usage)
            new_population.append(self.Individual(purified_action, stock_usage, fitness))

        return new_population

    def _rank_selection(self, population):
        """
        Rank selection.
        """
        population_sorted = sorted(population, key=lambda ind: ind.fitness)
        total_rank = sum(range(1, len(population) + 1))
        selection_probs = [rank / total_rank for rank in range(1, len(population) + 1)]
        return random.choices(population_sorted, weights=selection_probs, k=1)[0]

    def _two_point_crossover(self, parent1, parent2):
       """
       Perform two-point crossover between two parents' actions.
       """
       length = min(len(parent1), len(parent2))
       if length < 2:
           return parent1 if random.random() < 0.5 else parent2

       point1, point2 = sorted(random.sample(range(length), 2))
       child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
       return child

    def _mutate(self, actions, products, stocks):
        """
        Perform mutation by modifying an action and updating stock_usage.
        """
        if actions:
            idx = random.randint(0, len(actions) - 1)
            random_solution, stock_usage = self._generate_heuristic_solution(products, stocks)
            if random_solution:
                actions[idx] = random_solution[random.randint(0, len(random_solution) - 1)]
        return actions, stock_usage

    
    #-----------------------SUPPORT FUNCTION FOR GENETIC POLICY--------------------------

    def _can_place_(self, stock, position, prod_size):
      """
      Check if a product can be placed at the given position in the stock.
      Ensure the product fits within the boundaries and the area is empty (filled with -1).
      """
      #Just an improvement of the given _can_place_ function
      pos_x, pos_y = position  # Position where the product will be placed (top-left corner)
      prod_w, prod_h = prod_size  # Width and height of the product

      # Get the stock's dimensions
      stock_w, stock_h = self._get_stock_size_(stock)

      # Check if the product fits within the boundaries of the stock
      if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
        # If the product exceeds the boundaries, return False
        return False

      # Check if the area in the stock is empty (i.e., filled with -1)
      return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
    
    def deep_copy_stock_usage(self,stocks):
      copied_stocks = []
      for stock in stocks:
        copied_stocks.append(stock.copy())  # Sử dụng .copy() của numpy
      return copied_stocks
    
    def is_valid_action(self, action, stocks):
      """
      Check if the given action overlaps with any already placed product on the stock.
      """
      stock_idx = action["stock_idx"]
      prod_w, prod_h = action["size"]
      pos_x, pos_y = action["position"]
    
      stock = stocks[stock_idx]

      # Check the stock grid for overlap
      if np.any(stock[pos_y:pos_y + prod_h, pos_x:pos_x + prod_w] >= 0):  # Check if any value >=0 (meaning occupied)
        return False
      
      return True
    
    def purify_solutions(self, actions):
      """
      Purify the actions to ensure no duplicates while keeping them as dictionaries.
      """
      purified_actions = []
      seen = set()

      for action in actions:
        # Convert action["size"] (which is a numpy array) to a tuple
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        # Create a unique key from stock_idx, size (as tuple), and position using frozenset
        action_key = (stock_idx, (x,y), (width,height))

        # If this key has not been seen, add the action to the purified list
        if action_key not in seen:
            purified_actions.append(action)
            seen.add(action_key)

      return purified_actions
    

