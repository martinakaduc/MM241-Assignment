from policy import Policy
#improve greedy using Next fit decreasing height (NFDH) and Floor-Ceiling (FC) heuristic

class Policy2311226_2311636_2312672_2313459(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.stock_index = 0
        self.numItem = 0
        self.prodIndex = 0
        self.sortedProd = None
        self.rotateIndex = 0
        self.sortedRotateProd = None
        if policy_id == 1:
            self.isFC = False
            self.currentStrip = 0
            self.nextStrip = -1
        elif policy_id == 2:
            self.greedySortProd = None
            self.isFC = True
            self.floorPhase = True
            self.floorLevel = 0
            self.ceilLevel = -1

    def get_action(self, observation, info):
        if self.isFC == 1:
            return self.FCheuristic(observation, info)
        else:
            return self.NFDHheuristic(observation, info)
    # Student code here
    #NFDH heuristic
    def NFDHheuristic(self, observation, info):
        self.resetEp(observation)
        prod_size = [0, 0]
        stock = observation["stocks"][self.stock_index]
        stock_w, stock_h = self._get_stock_size_(stock)
        # Pick a product that has quality > 0 and has max height
        while self.prodIndex < len(self.sortedProd):
            prod = self.sortedProd[self.prodIndex]
            if prod["quantity"] > 0:
                while self.rotateIndex < len(self.sortedRotateProd) and self.sortedRotateProd[self.rotateIndex]["quantity"] <= 0:
                    self.rotateIndex += 1
                isRotate = False
                if self.rotateIndex < len(self.sortedRotateProd):
                    rotatedProd = self.sortedRotateProd[self.rotateIndex]
                    if rotatedProd["size"][0] > prod["size"][1]:
                        prod = rotatedProd
                        isRotate = True
                prod_size = prod["size"][::-1] if isRotate else prod["size"]
                prod_w, prod_h = prod_size
                if stock_w < prod_w or stock_h < prod_h:
                    self.updateIndex(isRotate)
                    continue
                if prod_h > stock_h - self.currentStrip:
                    self.updateIndex(isRotate)
                    continue
                for x in range(stock_w - prod_w + 1):
                    if self._can_place_(stock, (x, self.currentStrip), prod_size):
                        if self.nextStrip <= self.currentStrip:
                            self.nextStrip = self.currentStrip + prod_h
                        self.numItem -= 1
                        return {"stock_idx": self.stock_index, "size": prod_size, "position": (x, self.currentStrip)}
                self.updateIndex(isRotate)
                continue
            self.prodIndex += 1
        #if ceil level or floor level reach stock height perform greedy fill and move to next stock
        #phase changer
        if self.nextStrip >= stock_h - 1:
            self.moveToNextStock()
            return self.NFDHheuristic(observation, info)
        self.rotateIndex = 0
        self.prodIndex = 0
        if self.nextStrip <= self.currentStrip:
            self.nextStrip = stock_h - 1
        self.currentStrip = self.nextStrip
        return self.NFDHheuristic(observation, info)
    #FC heuristic
    def FCheuristic(self, observation, info):
        self.resetEp(observation)
        prod_size = [0, 0]
        stock = observation["stocks"][self.stock_index]
        stock_w, stock_h = self._get_stock_size_(stock)
        # Pick a product that has quality > 0 and has max height
        while self.prodIndex < len(self.sortedProd):
            prod = self.sortedProd[self.prodIndex]
            if prod["quantity"] > 0:
                while self.rotateIndex < len(self.sortedRotateProd) and self.sortedRotateProd[self.rotateIndex]["quantity"] <= 0:
                    self.rotateIndex += 1
                isRotate = False
                if self.rotateIndex < len(self.sortedRotateProd):
                    rotatedProd = self.sortedRotateProd[self.rotateIndex]
                    if rotatedProd["size"][0] > prod["size"][1]:
                        prod = rotatedProd
                        isRotate = True
                prod_size = prod["size"][::-1] if isRotate else prod["size"]
                prod_w, prod_h = prod_size
                if stock_w < prod_w or stock_h < prod_h:
                    self.updateIndex(isRotate)
                    continue
                #floor phase
                if self.floorPhase:
                    if prod_h > stock_h - self.floorLevel:
                        self.updateIndex(isRotate)
                        continue
                    for x in range(stock_w - prod_w + 1):
                        if self._can_place_(stock, (x, self.floorLevel), prod_size):
                            if self.ceilLevel < self.floorLevel:
                                self.ceilLevel = self.floorLevel + prod_h - 1
                            self.numItem -= 1
                            return {"stock_idx": self.stock_index, "size": prod_size, "position": (x, self.floorLevel)}
                #ceiling phase
                else:
                    if prod_h > self.ceilLevel - self.floorLevel + 1:
                        self.updateIndex(isRotate)
                        continue
                    for x in range(stock_w - 1, prod_w - 1, -1):
                        if self._can_place_(stock, (x - prod_w + 1, self.ceilLevel - prod_h + 1), prod_size):
                            self.numItem -= 1
                            return {"stock_idx": self.stock_index, "size": prod_size, "position": (x - prod_w + 1, self.ceilLevel - prod_h + 1)}
                self.updateIndex(isRotate)
                continue
            self.prodIndex += 1
        #if ceil level or floor level reach stock height perform greedy fill and move to next stock
        if self.ceilLevel == stock_h - 1 or self.floorLevel == stock_h - 1:
            for prod in self.greedySortProd:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                self.numItem -= 1
                                return {"stock_idx": self.stock_index, "size": prod_size, "position": (x, y)}
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                self.numItem -= 1
                                return {"stock_idx": self.stock_index, "size": prod_size, "position": (x, y)}
            self.moveToNextStock()
            return self.FCheuristic(observation, info)
        #phase changer
        if self.floorPhase:
            # from floor phase to ceil phase, if floor phase did not place any product 
            # set ceil level to stock height than perform a greedy fill on that row
            self.floorPhase = False
            self.rotateIndex = 0
            self.prodIndex = 0
            if self.ceilLevel < self.floorLevel:
                self.ceilLevel = stock_h - 1
        else:
            self.rotateIndex = 0
            self.prodIndex = 0
            self.floorPhase = True
            self.floorLevel = self.ceilLevel + 1
        return self.FCheuristic(observation, info)
    # helper function 
    # reset variable when move to next EP
    def resetEp(self, observation):
        if self.numItem == 0: 
            list_prods = observation["products"]
            self.numItem = sum(prod["quantity"] for prod in list_prods)
            self.prodIndex = 0
            self.rotateIndex = 0
            self.stock_index = 0
            self.sortedProd = sorted(list_prods, key=lambda prod : prod["size"][1], reverse= True)
            self.sortedRotateProd = sorted(list_prods, key=lambda prod : prod["size"][0], reverse= True)
            if self.isFC == True:
                self.floorPhase = True
                self.floorLevel = 0
                self.ceilLevel = -1
                self.greedySortProd = sorted(self.sortedProd, key=lambda prod : prod["size"][1]*prod["size"][0], reverse= True)
            else:
                self.currentStrip = 0
                self.nextStrip = -1
    # reset variable when move to next stock
    def moveToNextStock(self):
        self.stock_index += 1
        self.prodIndex = 0
        self.rotateIndex = 0
        if self.isFC == True:
            self.floorPhase = True
            self.ceilLevel = -1
            self.floorLevel = 0
        else:
            self.currentStrip = 0
            self.nextStrip = -1
    #stock index update
    def updateIndex(self, isRotate):
        if isRotate == True:
            self.rotateIndex += 1
        else:
            self.prodIndex += 1
    # You can add more functions if needed
