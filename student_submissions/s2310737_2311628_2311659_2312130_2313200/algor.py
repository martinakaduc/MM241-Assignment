from policy import Policy

import numpy as np
from scipy.optimize import linprog

# Giải thuật Worst Fit:
#   Sắp xếp các stock giảm dần.
#   Đồng thời sắp xếp các sản phẩm giảm dần.
#
#   Duyệt qua danh sách trên và đặt các sản phẩm vào các stock
#   theo danh sách trên. Khi không thể đặt sản phẩm bất kì vào
#   stock hiện tại nữa, ta chuyển sang stock tiếp theo.
#
#   Tiêu chí sắp xếp: Cạnh lớn nhất.
#   Giải thuật đồng thời xoay các sản phẩm theo chiều của stock,
#   nhưng có thể xoay lại sản phẩm nếu không vừa.
class WorstFit(Policy):
    def __init__(self, policy_id=1):

        assert policy_id in [1, 2]

        super().__init__()

        # lưu lại cho lần gọi get_action sau.
        self.stock_idx = 0

        # Sắp xếp các stock và sản phẩm bằng chỉ số
        self.stock_indices = []
        self.prods_indices = []

    def get_action(self, observation, info):

        list_stock = observation["stocks"]
        list_prods = observation["products"]

        # Stock mới
        if info["filled_ratio"] == 0.0:
            self.prepare(list_prods, list_stock)

        self.current_trim_loss = info["trim_loss"]

        # Thử đặt sản phẩm vào stock
        action = self.get_core_action(list_prods, list_stock)

        return action

    def prepare(self, list_prods, list_stock):
        self.stock_idx = 0

        self.stock_indices = [i for i in range(len(list_stock))]
        self.prods_indices = [i for i in range(len(list_prods))]

        # Sắp xếp các stock theo chiều lớn nhất giảm dần
        self.stock_indices.sort(
            key=lambda i: max(self._get_stock_size_(list_stock[i])),
            reverse=True
        )

        # Tương tự với sản phẩm
        self.prods_indices.sort(
            key=lambda i: list_prods[i]["size"].max(),
            reverse=True
        )

    def get_core_action(self, list_prods, list_stock):
        # Bỏ qua các stock đã sử dụng để tiết kiệm thời gian
        for i in self.stock_indices[self.stock_idx:]:
            stock = list_stock[i]

            stock_w, stock_h = self._get_stock_size_(stock)

            for j in self.prods_indices:
                prod = list_prods[j]
                quantity = prod["quantity"]

                if quantity <= 0:
                    continue

                prod_size = prod["size"]
                __w, __h = prod_size

                if (__w <= __h) != (stock_w <= stock_h):
                    prod_size = prod_size[::-1]

                prod_w, prod_h = prod_size

                if stock_w >= prod_w and stock_h >= prod_h:
                    pos = self.try_place(stock, stock_w, stock_h, prod_size)
                    if pos is not None:
                        break

                prod_size = prod_size[::-1]

                if stock_w >= prod_h and stock_h >= prod_w:
                    pos = self.try_place(stock, stock_w, stock_h, prod_size)
                    if pos is not None:
                        break

            if pos is not None:
                x, y = pos
                idx = self.stock_indices[self.stock_idx]
                return {
                    "stock_idx": idx,
                    "size": prod_size,
                    "position": (x, y)
                }

            self.stock_idx += 1
        #

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def rotate_prod(self, prod):
        prod["size"] = np.flip(prod["size"])

    def try_place(self, stock, stock_w, stock_h, prod_size):
        prod_w, prod_h = prod_size
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
                #
            #
        #
        return None

class ColumnGeneration(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = []
        self.init = False
        self.num_prods = None

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        demand = np.array([prod["quantity"] for prod in products])
        sizes = [prod["size"] for prod in products]
        num_prods = len(products)

        if not self.init or self.num_prods != num_prods:
            self.init_patterns(num_prods, sizes, stocks)
            self.init = True
            self.num_prods = num_prods

        while True:
            c = np.ones(len(self.patterns))
            A = np.array(self.patterns).T
            b = demand

            res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')

            if res.status != 0:
                break

            dual_prices = res.ineqlin.marginals if hasattr(res.ineqlin, 'marginals') else None

            if dual_prices is None:
                break

            new_pattern = self.solve_pricing_problem(dual_prices, sizes, stocks)

            if new_pattern is None or any(np.array_equal(new_pattern, p) for p in self.patterns):
                break

            self.patterns.append(new_pattern)

        best_pattern = self.select_best_pattern(self.patterns, demand)
        action = self.pattern_to_action(best_pattern, sizes, stocks)
        return action

    # khởi tạo các mẫu có thể, mỗi mẫu là một mảng nhị phân , cho biết có thể một sản phẩm có thể cắt ra từ stock hay không
    def init_patterns(self, num_prods, sizes, stocks):
        self.patterns = []
        for j in range(len(stocks)):
            stock_size = self._get_stock_size_(stocks[j])
            for i in range(num_prods):
                if stock_size[0] >= sizes[i][0] and stock_size[1] >= sizes[i][1]:
                    pattern = np.zeros(num_prods, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)
        #lọc ra các mẫu bị trùng lặp
        self.patterns = list({tuple(p): p for p in self.patterns}.values())

    #bài toán con tìm mẫu cắt có chi phí giảm dương nhất, cải thiện bài toán chính
    def solve_pricing_problem(self, dual_prices, sizes, stocks):
        best_pattern = None
        best_reduced_cost = -1

        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)

            if stock_w <= 0 or stock_h <= 0:
                continue

            #sử dụng dynamic programing
            n = len(sizes)
            dp = np.zeros((stock_h + 1, stock_w + 1))

            for i in range(n):
                w, h = sizes[i]
                if w > stock_w or h > stock_h or dual_prices[i] <= 0:
                    continue
                for x in range(stock_w, w - 1, -1):
                    for y in range(stock_h, h - 1, -1):
                        dp[y][x] = max(dp[y][x], dp[y - h][x - w] + dual_prices[i])

            #tạo mẫu cắt từ bảng
            pattern = np.zeros(n, dtype=int)
            w, h = stock_w, stock_h

            for i in range(n - 1, -1, -1):
                item_w, item_h = sizes[i]
                if dp[h][w] == dp[h - item_h][w - item_w] + dual_prices[i] and (h - item_h) >= 0 and (
                        w - item_w) >= 0 and dual_prices[i] > 0:
                    pattern[i] = 1
                    w -= item_w
                    h -= item_h

            reduced_cost = np.dot(pattern, dual_prices) - 1 #tính toán chi phí giảm
            if reduced_cost > best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_pattern = pattern

        #nếu chi phí giảm không quá nhỏ
        if best_reduced_cost > 1e-6:
            return best_pattern
        else:
            return None

    #chọn mẫu tốt nhất dựa trên độ bao phủ, đáp ứng nhiều sản phẩm nhất
    def select_best_pattern(self, patterns, demand):
        best_pattern = None
        best_coverage = -1
        best_cost = float('inf')

        for pattern in patterns:
            coverage = np.sum(np.minimum(pattern, demand)) #tổng số sản phẩm mà mẫu có thể cắt
            cost = np.sum(pattern)
            if coverage > best_coverage or (coverage == best_coverage and cost < best_cost):
                best_coverage = coverage
                best_cost = cost
                best_pattern = pattern

        return best_pattern

    #xác định sản phẩm và vị trí của nó trên stock
    def pattern_to_action(self, pattern, sizes, stocks):
        for i, count in enumerate(pattern):
            if count > 0:
                prod_size = sizes[i]
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self.bottom_left_place(stock, prod_size)
                        if position is not None:
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": position
                            }
        # nếu không tìm được vị trí hợp lệ
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }


    #tìm vị trí tốt nhất để đặt sản phẩm vào stock
    def bottom_left_place(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
