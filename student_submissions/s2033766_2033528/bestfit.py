import copy

from policy import Policy

__all__ = ["BestFit"]


class Rectangle:
    def __init__(self, width, height, quantity):
        self.width = width
        self.height = height
        self.quantity = quantity


class BestFit(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        list_prods = observation["products"]

        rectangles = [Rectangle(prod["size"][0], prod["size"][1], prod["quantity"]) for prod in list_prods if
                      prod["quantity"] > 0]
        # Sắp xếp các hình chữ nhật theo diện tích giảm dần
        rectangles.sort(key=lambda x: x.width * x.height, reverse=True)

        stock_areas = [(i, self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[0]) for i, stock in
                       enumerate(observation["stocks"])]
        stock_areas.sort(key=lambda x: x[1], reverse=True)
        stocks = observation["stocks"]
        # Pick a product that has quality > 0
        # Loop through all stocks
        for idx, stock_area in stock_areas:
            solution = self.wang_algorithm(idx, stocks[idx], stock_area, rectangles)
            if solution is None:
                continue
            return solution

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    @staticmethod
    def calculate_waste(stock, position, prod_size):
        x, y = position
        width, height = prod_size
        stock[x: x + width, y: y + height] = 0
        return (stock == -1).sum()

    def wang_algorithm(self, stock_idx, selected_stock, stock_area, rectangles):
        stock_w, stock_h = self._get_stock_size_(selected_stock)
        rejection_parameter = 1
        all_solution = []
        for rectangle in rectangles:
            rep_stock = copy.deepcopy(selected_stock)
            prod_w, prod_h = rectangle.width, rectangle.height
            prod_size = (prod_w, prod_h)
            is_placed = False
            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(selected_stock, (x, y), prod_size):
                            waste = self.calculate_waste(rep_stock, (x, y), prod_size)
                            if waste < rejection_parameter * stock_w * stock_h:
                                is_placed = True
                                all_solution.append((waste, [int(x), int(y)], prod_size))
                                break
                    if is_placed:
                        break
            if is_placed:
                continue
            if stock_w >= prod_h and stock_h >= prod_w:
                pos_x, pos_y = None, None
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(selected_stock, (x, y), prod_size[::-1]):
                            waste = self.calculate_waste(rep_stock, (x, y), prod_size)
                            if waste < rejection_parameter * stock_w * stock_h:
                                is_placed = True
                                all_solution.append((waste, [int(x), int(y)], prod_size))
                                break
                    if is_placed:
                        break

        if len(all_solution) == 0:
            return

        all_solution.sort(key=lambda x: x[0], reverse=True)
        best_solution = all_solution[0]
        (best_prod_w, best_prod_h) = best_solution[2]
        can_replace = False
        for x in range(stock_w - best_prod_w + 1):
            for y in range(stock_h - best_prod_h + 1):
                if self._can_place_(selected_stock, (x, y), best_solution[2]):
                    can_replace = True
        if not can_replace:
            return
        return {"stock_idx": stock_idx, "size": best_solution[2], "position": best_solution[1]}

