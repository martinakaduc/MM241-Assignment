from policy import Policy
from scipy.optimize import linprog
from itertools import product

class Policy2310270(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Student code here
        if policy_id == 1:
            self.description = "Brach_And_Price_Policy"
        elif policy_id == 2:
            self.description = "First_Fit_Decreasing_Policy"

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.BranchAndPricePolicy(observation)
        elif self.policy_id == 2:
            return self.FirstFitDecreasingPolicy(observation)

class BranchAndPricePolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Bước 1: Khởi tạo bài toán chính
        demand = observation["demand"]
        dimensions = observation["dimensions"]
        stock_dimensions = observation["stock_dimensions"]
        patterns = self.generate_initial_patterns(demand, dimensions, stock_dimensions)

        while True:
            # Bước 2: Giải bài toán chính
            x, dual_prices = self.solve_master_problem(patterns, demand)

            # Bước 3: Kiểm tra tính nguyên
            if self.is_integer_solution(x):
                return self.format_result(x, patterns)

                # Bước 4: Giải bài toán sinh cột
            new_pattern = self.solve_pricing_problem(dual_prices, dimensions, stock_dimensions)
            if not new_pattern:
                break  # Không có mẫu mới => tối ưu

            # Bước 5: Cập nhật bài toán chính
            patterns.append(new_pattern)

            # Bước 6: Phân nhánh nếu cần thiết
        branches = self.perform_branching(x, patterns)
        solutions = []
        for branch in branches:
            solutions.append(self.branch_and_price(demand, dimensions, stock_dimensions))

            # Bước 7: Trả về nghiệm tốt nhất
        best_solution = min(solutions, key=lambda sol: self.objective_value(sol))
        return self.format_result(best_solution, patterns)

    def generate_initial_patterns(self, demand, dimensions, stock_dimensions):
        W, H = stock_dimensions
        patterns = []
        for i, (w, h) in enumerate(dimensions):
            num_items = min(W // w, H // h)
            pattern = [0] * len(demand)
            pattern[i] = num_items
            patterns.append(pattern)
        return patterns

    def solve_master_problem(self, patterns, demand):
        num_patterns = len(patterns)
        num_items = len(demand)

        # Hàm mục tiêu
        c = [1] * num_patterns

        # Ràng buộc
        A = [[patterns[j][i] for j in range(num_patterns)] for i in range(num_items)]
        b = demand

        # Giải bài toán LP
        result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method='highs')

        x = result.x  # Nghiệm của bài toán
        dual_prices = result.dual_values  # Giá kép
        return x, dual_prices

    def solve_pricing_problem(self, dual_prices, dimensions, stock_dimensions):
        W, H = stock_dimensions
        best_pattern = None
        max_reduced_cost = 0

        # Duyệt qua tất cả các mẫu cắt khả thi
        for pattern in self.generate_all_feasible_patterns(dimensions, stock_dimensions):
            reduced_cost = sum(dual_prices[i] * pattern[i] for i in range(len(pattern)))
            if reduced_cost > max_reduced_cost:
                max_reduced_cost = reduced_cost
                best_pattern = pattern

                # Nếu không tìm thấy mẫu tốt hơn, trả về None
        if max_reduced_cost <= 0:
            return None

        return best_pattern

    def generate_all_feasible_patterns(self, dimensions, stock_dimensions):
        W, H = stock_dimensions
        feasible_patterns = []
        for pattern in product(range(W + 1), repeat=len(dimensions)):
            if self.is_pattern_feasible(pattern, dimensions, stock_dimensions):
                feasible_patterns.append(pattern)
        return feasible_patterns

    def is_pattern_feasible(self, pattern, dimensions, stock_dimensions):
        W, H = stock_dimensions
        total_width = total_height = 0
        for count, (w, h) in zip(pattern, dimensions):
            total_width += count * w
            total_height += count * h
        return total_width <= W and total_height <= H

    def perform_branching(self, x, patterns):
        branches = []
        for i, value in enumerate(x):
            if not self.is_integer_solution([value]):
                # Tạo 2 nhánh: ép biến x[i] thành nguyên
                branch1 = patterns.copy()
                branch1[i] = int(value)
                branch2 = patterns.copy()
                branch2[i] = int(value) + 1
                branches.extend([branch1, branch2])
                break
        return branches

    def is_integer_solution(self, x):
        return all(abs(value - round(value)) < 1e-6 for value in x)

    def format_result(self, x, patterns):
        stock_idx = -1
        pos_x, pos_y = None, None
        prod_size = [0, 0]

        # Tìm mẫu cắt đầu tiên có nghiệm nguyên
        for i, count in enumerate(x):
            if count > 0:
                stock_idx = i  # Chỉ số kho
                prod_size = patterns[i]  # Kích thước sản phẩm
                # Tìm vị trí (pos_x, pos_y) cho sản phẩm
                pos_x, pos_y = 0, 0  # Cần logic để xác định vị trí thực tế
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def objective_value(self, solution):
        return sum(solution)

class FirstFitDecreasingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Loop through all products sorted by size (decreasing)
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Check if the product can fit in the stock
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    # Check for rotated product
                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            prod_size = prod_size[::-1]  # Update product size to rotated
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

