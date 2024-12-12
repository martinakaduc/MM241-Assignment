from policy import Policy


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.resetAll()
            self.id = 1
        elif policy_id == 2:
            self.resetAll()
            self.id = 2

    def get_action(self, observation, info):
        # Student code here
        if self.id == 1:
            if self.first_action:
                self.initialize(observation)

            if (len(self.best_placement) == 0 or self.current_step >= len(self.best_placement)):
                self.current_step = 0
                self.best_placement = []
                placements = []
                best_ratio = 0
                cnt = 0
                for i in range(self.stocks_len):
                    best_fit_array = []
                    stock_idx = self.sorted_stocks[i][0]
                    stock = self.stocks[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    list_prod = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
                    total_prod_area = 0
                    temp_stock = stock.copy()
                    for prod in list_prod:
                        quantity = prod["quantity"]
                        while quantity > 0:
                            prod_size = prod["size"]
                            prod_w, prod_h = prod_size
                            if stock_w < prod_w or stock_h < prod_h:
                                break
                        
                            best_fit = None
                            
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(temp_stock, (x, y), prod_size):
                                        best_fit = {"x": x, "y": y, "size": prod_size, "index": stock_idx}

                                        for i in range(prod_w):
                                            for j in range(prod_h):
                                                temp_stock[x+i][y+j] = stock_idx
                                        break
                                if best_fit:
                                    break
                            if best_fit:
                                total_prod_area += prod_w * prod_h
                                best_fit_array.append(best_fit)
                                quantity -= 1
                                continue
                            else:
                                new_size = prod_size[::-1]
                                for x in range(stock_w - prod_h + 1):
                                    for y in range(stock_h - prod_w + 1):
                                        if self._can_place_(temp_stock, (x, y), new_size):
                                            best_fit = {"x": x, "y": y, "size": new_size, "index": stock_idx}

                                            for i in range(prod_w):
                                                for j in range(prod_h):
                                                    temp_stock[x+j][y+i] = stock_idx
                                            break
                                    if best_fit:
                                        break
                                if best_fit:
                                    total_prod_area += prod_w * prod_h
                                    best_fit_array.append(best_fit)
                                    quantity -= 1
                                    continue
                                else:
                                    break
                    ratio = total_prod_area / (stock_w * stock_h)
                    if ratio > 0.9:
                        placements.append((best_fit_array, ratio))
                        cnt += 1
                        if cnt == 5:
                            break
                    if ratio > best_ratio:
                        best_ratio = ratio
                        placements.append((best_fit_array, ratio))
                self.best_placement = max(placements, key=lambda x: x[1])[0]
            if self.current_step < len(self.best_placement):
                best_fit = self.best_placement[self.current_step]
                self.current_step += 1
                return {"stock_idx": best_fit["index"], "size": best_fit["size"], "position": (best_fit["x"], best_fit["y"])}
        elif self.id == 2:
            if self.first_action:
                self.initialize(observation)

            for prod in self.sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    for i in range(self.stocks_len):
                        stock_idx = self.sorted_stocks[i][0]
                        stock = self.stocks[stock_idx]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        if stock_w < prod_w or stock_h < prod_h:
                            continue
                        
                        
                        best_fit = None
                        
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    best_fit = {"x": x, "y": y, "size": prod_size, "index": stock_idx}
                                    break
                            if best_fit:
                                break
                        if best_fit:
                            break
                        

                        if stock_w < prod_h or stock_h < prod_w:
                            continue

                        new_size = prod_size[::-1]
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), new_size):
                                    best_fit = {"x": x, "y": y, "size": new_size, "index": stock_idx}
                                    break
                            if best_fit:
                                break
                        if best_fit:
                            break
                        
                    if best_fit is None:
                        continue

                    if sum([prod["quantity"] for prod in observation["products"]]) == 1:
                        self.resetAll()

                    return {"stock_idx": best_fit["index"], "size": best_fit["size"], "position": (best_fit["x"], best_fit["y"])}
    # First algorithm Best fit decreasing (BFD)

    def resetAll(self):
        self.best_placement = []
        self.current_step = 0
        self.waste = []
        self.first_action = True
        self.sorted_products = []
        self.stocks = []
        self.stock_indices = []
        self.sorted_stocks = []
        self.stocks_len = 0
        self.total_area_stock = []

    def initialize(self, observation):
        self.first_action = False    
        # Sort products by size
        self.sorted_products = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        # Calculating total area of stocks
        stocks = observation["stocks"]
        self.stocks = stocks
        for i, stock in enumerate(self.stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            self.stock_indices.append(i)
            self.total_area_stock.append(stock_w * stock_h)
            self.waste.append(stock_w * stock_h)

        # Sort stocks by total area
        total_area_products = sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in observation["products"]])
        self.sorted_stocks = zip(self.stock_indices, self.total_area_stock)
        self.sorted_stocks = sorted(self.sorted_stocks, key=lambda x: abs(x[1]-total_area_products))
        self.stocks_len = len(self.sorted_stocks)
    # Student code here
    # You can add more functions if needed
