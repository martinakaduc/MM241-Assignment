import numpy as np
from policy import Policy

class Policy2252880_2252442_2213973_2252608(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 2:
            # Initialize for later use
            self.crn_stock = None       # Hold the stock that we currently working on
            self.crn_stock_idx = -1     # The current stock index
            self.crn_strip_begin = 0    # The begin of the current strip in the current stock
            self.crn_strip_end = None   # The end of the current strip in the current stock
            self.crn_prod_idx = 0       # Use to track smaller product id for complexity decreasing
        elif policy_id == 1:
            self.new_stock_flag = True          # New stock -> True
            self.crn_stock_idx = -1             # Current stock id
            self.crn_list_prods_start = 0       # Use to filter all prod that have quantity = 0 at the beginning of list_prods
            self.j = 0                          # Pattern index
            self.m = 0                          # Number of in list_prods after filtered
            self.p = None                       # Pattern matrix, contain all the generated cut pattern
            self.a = None                       # Main pattern matrix along length    
            self.b = None                       # Main pattern matrix along width
            self.A = None                       # Addition pattern matrix along length, hold the product filled in the lost area after main pattern
            self.B = None                       # Addition pattern matrix along width, hold the product filled in the lost area after main pattern
            self.place_bp_idx = 0               # Index of the current product to be place in bp
            self.place_bP_idx = 0               # Index of the current product to be place in bP
            self.bp = None                      # Best main pattern
            self.bP = None                      # Best additional pattern
            self.strip_y = [0]                  # Hold the start location (pos_y) of a strip, number of strip is number of different product in main pattern 
            self.strip_y_it = -1                # Iterator for strip_y
            self.new_type = True                # Check if its a new type of product in pattern
            self.map_bp = None                  # Hold what strip for a product in self.bp to be put in
            self.map_bP = None                  # Hold what strip for a product in self.bP to be put in

    def get_action(self, observation, info):
        if self.policy_id == 2:
            # Re-initialize everything if the env is reset
            if info["filled_ratio"] == 0:
                self.__init__(2)
            # Sort the product list in observation in non-increasing width
            self._sort_product_by_width_(observation)
            list_prods = observation["products"]
            list_stocks = observation["stocks"]
            
            # Update crn_stock to the updated stock as stock will be changed after each action
            for i, stock in enumerate(list_stocks):
                if self.crn_stock_idx == -1:
                    break
                if i < self.crn_stock_idx:
                    continue
                self.crn_stock = stock
                break

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            # Pick a product that has quantity > 0
            for i_prod, prod in enumerate(list_prods):
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    pos_x, pos_y = None, None
                    # Init strip if there is currently no strip in stock
                    if self.crn_strip_end is None:
                        pos_x, pos_y = 0, 0
                        success = self._new_strip_(list_stocks, prod, pos_x, prod_w)
                        # No more strip can be make
                        if success == False:
                            break

                    # Cutting process
                    # Break after cutted a piece
                    while 1:
                        # Place a piece in strip
                        pos_x, pos_y = self._place_in_strip_(prod_size)
                        if pos_x is None and pos_y is None:     # Can't place
                            temp_prod_size = None
                            # Try to place smaller product into the strip until no more product can be place
                            temp_prod_size, pos_x, pos_y = self._place_smaller_in_strip_(observation)
                            if pos_x is None and pos_y is None: # No more product can be place
                                self.crn_prod_idx = i_prod
                                success = self._new_strip_(list_stocks, prod, self.crn_strip_end, prod_w) # Make new strip
                                # No more strip can be make
                                if success == False:
                                    break
                            else:
                                prod_size = temp_prod_size  # Update the prod_size (currently the big product)
                                break                       # to the small product to return the info of the small product
                        else:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = self.crn_stock_idx
                        break
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        elif self.policy_id == 1:
            # Re-initialize everything if the env is reset
            if info["filled_ratio"] == 0:
                self.__init__(1)
            
            # Step 1
            # Sort the product list in observation in non-increasing height
            list_prods = self._sort_product_by_height_(observation)
            
            # Filter the product list, remove all product with quantity = 0 at the beginning of the list
            list_prods = list_prods[self.crn_list_prods_start:]

            # Reset everthing when going to new stock
            if self.new_stock_flag:
                self.j = 0
                
                # Update self.crn_list_prods_start and re-filter list_prods
                list_prods = self._update_list_prods_(list_prods)
                self.m = len(list_prods)
                
                # Go from previous stock to the current stock
                self.crn_stock_idx = self.crn_stock_idx + 1
                
                # Re-create all pattern matrix with ncol = 0, nrow = number of product in list_prods and all value = 0
                self.p = [[0] * 1 for _ in range(self.m)]
                self.a = [[0] * 1 for _ in range(self.m)]
                self.b = [[0] * 1 for _ in range(self.m)]
                self.A = [[0] * 1 for _ in range(self.m)]
                self.B = [[0] * 1 for _ in range(self.m)]
                
                # Reset best pattern, no pattern for the current stock yet
                self.bp = None
                self.bP = None
                self.place_bp_idx = 0
                self.place_bP_idx = 0
                
                # No strip in current stock yet
                self.strip_y = [0]
                self.strip_y_it = -1
                self.new_type = True

                self.map_bp = [[-2] * 1 for _ in range(self.m)]
                self.map_bP = [[-2] * 1 for _ in range(self.m)]

            list_stocks = observation["stocks"]
            
            # Filter list stock, remove finished stock from list
            list_stocks = list_stocks[self.crn_stock_idx:]
            
            # Placed all product, return default value to end
            if not list_prods:
                prod_size = [0, 0]
                stock_idx = -1
                pos_x, pos_y = 0, 0
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            for i_stock, stock in enumerate(list_stocks):
                W, L = self._get_stock_size_(stock)
                m = self.m
                
                # New stock so generate pattern for the stock
                if self.new_stock_flag:
                    # Off flag
                    self.new_stock_flag = False
                    
                    # Step 2: generate pattern matrix a, b, init p
                    self._init_pattern_(L, W, list_prods, 0, m)

                    # Step 6: generate pattern matrix A, B, complete p
                    while self._fulfill_pattern_(L, W, list_prods) == 0:
                        pass

                    # After generated all pattern, find the best pattern to cut
                    self.bp, self.bP = self._find_best_pattern_(self.p, list_prods, L*W)
                    
                    # Fill in strip information (start and end)
                    self._make_strip_(list_prods)
                
                prod_size = [0, 0]
                stock_idx = self.crn_stock_idx
                pos_x, pos_y = 0, 0

                # Place process
                prod_size, pos_x, pos_y = self._pattern_place_(stock, list_prods)
                
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _sort_product_by_height_(self, observation):
        """Sort the product list in non-increasing order height
        In case width is equal, sort in non-increasing order width
        Updated directly into observation, no return value"""

        # Key method
        def get_size(product): 
            w, h = product["size"] 
            return h, w
        
        list_prods = observation["products"]
        return sorted(list_prods, key = get_size, reverse=True)

    def _update_list_prods_(self, list_prods):
        shift_count = 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                break
            # If product quantity at the current index is 0, shift crn_list_prod_start by 1. 
            self.crn_list_prods_start += 1
            shift_count += 1
        # Return the re-filtered list
        return list_prods[shift_count:]

    def _init_pattern_(self, L, W, list_prods, start, end):
        """Initiate the pattern with guillotine cut based on product, 
           creating strip. First product place on the left most of the stock define the strip"""
        
        j = self.j

        # Pre-compute sizes for quick access
        product_sizes = [(prod["size"][0], prod["size"][1]) for prod in list_prods]

        # Step 2: Pre-compute sum_l for range(0, start)
        sum_l = sum(product_sizes[z][1] * self.a[z][j] for z in range(start))

        # Step 3 & 4: Calculate number of each product type can be place along length/width 
        for z in range(start, end):
            w, l = product_sizes[z]
            # Store the number of the product along the length to a
            self.a[z][j] = (L - sum_l) // l

            # Update total filled length
            sum_l += l * self.a[z][j]

            # Store number of product of the same type as in "a" along width to b
            self.b[z][j] = W // w if self.a[z][j] > 0 else 0

            # Note: a[z][j] * b[z][j] = total number of product can be place in the area

        # Fill in mapping detail for bp
        for z in range(0, self.m):
            if self.a[z][j] > 0:
                self.map_bp[z][j] = z

        # Step 5: Store total number of prod can be place in the area into p
        column_p = [self.a[z][j] * self.b[z][j] for z in range(self.m)]
        for z in range(self.m):
            self.p[z][j] = column_p[z]

    def _fulfill_pattern_(self, L, W, list_prods):
        """Filling in the left space after initializing the pattern, 
           setting up and generate new pattern afterward"""
        
        j = self.j
        m = self.m

        # Pre-compute values to avoid redundant calculations
        product_sizes = [(prod["size"][0], prod["size"][1]) for prod in list_prods]

        # Width of the lost area on the right side of the placed product
        width_lost = [W - (self.b[i][j] * w) for i, (w, _) in enumerate(product_sizes)]  # Pre-computed k values
        
        # Step 6 (ii): Try to fill in the lost area
        for i, (w, l) in enumerate(product_sizes):
            k = width_lost[i]

            # Fill in mapping detail for bp, with -1 mean the current product i make the strip
            # And there is potentially lost space so there might be product in bP fill in that space
            if self.a[i][j] > 0:
                self.map_bP[i][j] = -1

            # Find the product to fill in the left space
            for z, (wz, lz) in enumerate(product_sizes):
                if z == i:
                    continue
                
                # If the length and width of lost area (self.a[i][j] * l * k) 
                # is bigger than the size of current considering prod
                if self.a[i][j] * l >= lz and k >= wz:
                    # Store the number of prod filled in the lost area along the length into A
                    self.A[z][j] = (self.a[i][j] * l) // lz

                    # Store the number of prod filled in the lost area along the width into B
                    self.B[z][j] = k // wz

                    # Update p, completing 1 pattern
                    self.p[z][j] += self.A[z][j] * self.B[z][j]

                    self.map_bP[i][j] = z

                # Fill in only 1 type of product to the lost area, after found that product and filled in, break
                if self.A[z][j] > 0:
                    break

        # Limit pattern generation
        if len(self.p[0]) >= 150000:
            return True

        # Step 7: Setting up for next pattern
        r = m - 1
        all_patern_generated = False
        while r >= 0: # Do step 8
            if self.a[r][j] > 0:
                # Check if all pattern generated, if yes, do Step 11
                if r == m - 1:
                    all_patern_generated = all(self.a[i][j] == 0 for i in range(0, m - 1))
                    if all_patern_generated:
                        break
                
                # Step 9: Start generating new pattern
                j += 1
                self.j = j

                # Creating new column for new pattern
                for mat in [self.a, self.b, self.p, self.A, self.B]:
                    for row in mat:
                        row.append(0)
                for mat in [self.map_bP, self.map_bp]:
                    for row in mat:
                        row.append(-2)

                # Generate according to condition
                if self.a[r][j - 1] >= self.b[r][j - 1]:
                    # For z = 1,2..., r - 1
                    for z in range(r):
                        self.a[z][j] = self.a[z][j - 1]
                        self.b[z][j] = self.b[z][j - 1]

                    # For z = r
                    self.a[r][j] = self.a[r][j - 1] - 1
                    self.b[r][j] = W // product_sizes[r][0] if self.a[r][j] > 0 else 0

                    # For z = r + 1,..., m
                    self._init_pattern_(L, W, list_prods, r + 1, m)
                else:
                    # Same
                    for z in range(r):
                        self.a[z][j] = self.a[z][j - 1]
                        self.b[z][j] = self.b[z][j - 1]

                    self.a[r][j] = self.a[r][j - 1]
                    self.b[r][j] = self.b[r][j - 1] - 1
                    self._init_pattern_(L, W, list_prods, r + 1, m)
                break
            
            # Step 10
            r -= 1

        return all_patern_generated

    def _find_best_pattern_(self, p, list_prods, stock_area):
        """Find the best pattern (pattern with minimum lost area)"""

        # Precompute product areas to avoid redundant calculations
        product_areas = [prod["size"][0] * prod["size"][1] for prod in list_prods]
        
        min_lost = float('inf')
        best_pattern_index = 0

        for j, pcol in enumerate(zip(*p)):
            # Adjust the cut pattern to match the product quantity and calc the filled area
            filled_area = 0
            for i, prow in enumerate(pcol):
                quantity = list_prods[i]["quantity"]

                # If the cut pattern required more product than the product quantity, 
                # set the value in the pattern to product quantity
                adjusted_value = min(prow, quantity)
                self.p[i][j] = adjusted_value


                filled_area += product_areas[i] * adjusted_value

            # Calc the lost area
            total_lost = stock_area - filled_area

            # Get the pattern with the min lost's index
            if total_lost < min_lost:
                min_lost = total_lost
                best_pattern_index = j

        bp = np.zeros(self.m)
        bP = np.zeros(self.m)

        # Fill in bp and bP the info of the best pattern
        self.best_pattern_idx = best_pattern_index
        p = [row[best_pattern_index] for row in self.p]
        p = np.array(p)
        for i, value in enumerate(p):
            if value > 0:
                org_bp_value = self.a[i][best_pattern_index] * self.b[i][best_pattern_index]
                if value > org_bp_value:
                    bp[i] = org_bp_value
                    bP[i] = value - org_bp_value
                else:
                    bp[i] = value
                
        # Return the best pattern
        return bp, bP

    def _make_strip_(self, list_prods):
        """Store strip information into self.strip_y"""

        self.strip_y_it = 0
        for i, prod in enumerate(list_prods):
            if self.a[i][self.best_pattern_idx] > 0:
                w, l = prod["size"]
                self.strip_y.append(self.strip_y[self.strip_y_it] + l * self.a[i][self.best_pattern_idx])
                self.strip_y_it += 1
        self.strip_y_it = -1

        bm = [row[self.best_pattern_idx] for row in self.map_bp]
        bM = [row[self.best_pattern_idx] for row in self.map_bP]

        self.bm = []
        self.bM = []
        
        # Filtering mapping information to match with strip_y
        for i in bm:
            if i != -2:
                self.bm.append(i)
        for i in bM:
            if i != -2:
                self.bM.append(i)
        self.bm = np.array(self.bm)
        self.bM = np.array(self.bM)

    def _pattern_place_(self, stock, list_prods):
        # Check if bp is empty, if empty, place by bP
        if np.all(self.bp == 0):
            self.place_bp_idx = len(list_prods)
        
        # Place bp
        # If place_bp_idx = length of list_prods, that's mean all product in pattern bp has been placed
        if self.place_bp_idx != len(list_prods):
            return self._place_by_bp_(stock, list_prods)
        
        # After placed all product in bp, place bP
        else:
            self.strip_y_it = 0
            return self._place_by_bP_(stock, list_prods)

    def _place_by_bp_(self, stock, list_prods):
        W, L = self._get_stock_size_(stock)
        for i, prod in enumerate(list_prods):
            # Get to the current prod index in bp
            if i != self.place_bp_idx:
                continue

            # If placed all the current product type, move on to the next type
            if self.bp[i] == 0:
                self.new_type = True
                self.place_bp_idx = self.place_bp_idx + 1
                continue
            
            prod_size = prod["size"]
            w, l = prod_size
            pos_x, pos_y = None, None

            # If the current product is the first new type product to be place, create a strip according to it
            if self.new_type:
                # Place product in the correct strip
                for id, sid in enumerate(self.bm):
                    if sid != i:
                        continue
                    
                    # Start and end pos of the correct strip
                    start_y = self.strip_y[id]
                    end_y = self.strip_y[id + 1] - l + 1
                    for y in range(start_y, end_y):
                        # New product type has to be placed in the left-most of the stock
                        x = 0
                        if self._can_place_(stock, (x, y), prod_size):
                            pos_x, pos_y = x, y
                            
                            # Store the strip end position (y)
                            if self.new_type:
                                self.new_type = False
                                self.strip_y_it = id        # Point to the current strip start point
                            break
                        else:
                            continue
            else:
                # Fill in the rest of the same product type that init the strip into the space
                start_y = self.strip_y[self.strip_y_it]
                end_y = self.strip_y[self.strip_y_it + 1] - l + 1

                for x in range(W - w + 1):
                    for y in range(start_y, end_y):
                        if self._can_place_(stock, (x, y), prod_size):
                            pos_x, pos_y = x, y
                            break
                    if pos_x is not None and pos_y is not None:
                        break

            # Place 1 product so decrease the number of the current product type by 1
            self.bp[i] = self.bp[i] - 1

            # Get to the next product to be place
            for j, prod in enumerate(list_prods):
                if j != self.place_bp_idx:
                    continue
                if self.bp[j] == 0:
                    self.new_type = True
                    self.place_bp_idx = self.place_bp_idx + 1
                    continue
                else:
                    break

            # If there is no more product to be place in bp
            if self.place_bp_idx == len(self.bp):
                # And if bP is empty(No product to be place in bP)
                if np.all(self.bP == 0):
                    # Finished placing product in the current stock, get to new stock
                    self.new_stock_flag = True

            # Return placed product information
            return prod_size, pos_x, pos_y
            
    def _place_by_bP_(self, stock, list_prods):
        W, L = self._get_stock_size_(stock)
        prod_size = [0, 0]
        pos_x, pos_y = 0, 0

        for i, prod in enumerate(list_prods):
            # Get to the current prod index in bp
            if i != self.place_bP_idx:
                continue

            # If placed all the current product type, move on to the next type
            if self.bP[i] == 0:
                self.new_type = True
                self.place_bP_idx = self.place_bP_idx + 1
                continue
            
            prod_size = prod["size"]
            w, l = prod_size
            pos_x, pos_y = None, None

            # Place product in the correct strip
            for id, sid in enumerate(self.bM):
                if sid != i:
                    continue
                # Start and end pos of the correct strip
                start_y = self.strip_y[id]
                end_y = self.strip_y[id + 1] - l + 1

                for y in range(start_y, end_y):
                    for x in range(W - w + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            pos_x, pos_y = x, y
                            break
                    if pos_x is not None and pos_y is not None:
                        break

            # Place 1 product so decrease the number of the current product type by 1
            self.bP[i] = self.bP[i] - 1

            # Get to the next product to be place
            for j, prod in enumerate(list_prods):
                if j != self.place_bP_idx:
                    continue
                if self.bP[j] == 0:
                    self.new_type = True
                    self.place_bP_idx = self.place_bP_idx + 1
                    continue
                else:
                    break

            # If there is no more product to be place in bp
            if self.place_bP_idx == len(list_prods):
                # Finished placing product in the current stock, get to new stock
                self.new_stock_flag = True

            # Return placed product information
            return prod_size, pos_x, pos_y
        
        return prod_size, pos_x, pos_y

    def _sort_product_by_width_(self, observation):
        """Sort the product list in non-increasing order width
           In case width is equal, sort in non-increasing order height
           Updated directly into observation, no return value"""

        # Key method
        def get_size(product): 
            w, h = product["size"] 
            return w, h
        
        list_prods = observation["products"]
        sorted_list_prods = sorted(list_prods, key = get_size, reverse=True)
        observation["products"] = sorted_list_prods

    def _find_best_stock_(self, list_stocks, prod):
        """Find the smallest stock in list_stocks that fit all or most of the demanded prod
           Return the index of the best stock and the best stock
           Return stock = None and id = -1 if cant find"""

        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        prod_quan = prod["quantity"]

        # Contain the info of the best empty stock that fit
        best_stock = None
        best_stock_idx = -1
        best_stock_fit_prod = -1
        best_stock_area = float('inf')

        for i, stock in enumerate(list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Continue to another stock if the prod cant fit in the stock
            if stock_w < prod_w or stock_h < prod_h:
                continue
            
            # Get number of product can fit horizontally and vertically
            crn_stock_fit_w = int(stock_w / prod_w)
            crn_stock_fit_h = int(stock_h / prod_h)

            # Total number of product can fit in the stock
            crn_stock_fit_prod = crn_stock_fit_w * crn_stock_fit_h

            # We only need to fit all the demanded, no need for higher
            if crn_stock_fit_prod > prod_quan:
                crn_stock_fit_prod = prod_quan

            crn_stock_area = stock_h * stock_w

            # Check if the current stock is the best stock
            if (best_stock_fit_prod < crn_stock_fit_prod or
                (best_stock_fit_prod == crn_stock_fit_prod and crn_stock_area < best_stock_area)):
                # If yes, check if it's a complete empty stock
                if self._is_empty_stock_(stock):
                    pass        # If yes, continue
                else:
                    continue    # If not, find another one

                # Check if there is space to place
                pos_x, pos_y = None, None
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            pos_x, pos_y = x, y
                            break
                    if pos_x is not None and pos_y is not None:
                        break
                
                if pos_x is not None and pos_y is not None:
                    # Check complete, update stock
                    best_stock = stock
                    best_stock_idx = i
                    best_stock_fit_prod = crn_stock_fit_prod
                    best_stock_area = crn_stock_area

        return best_stock_idx, best_stock
    
    def _is_empty_stock_(self, stock):
        """Check if the stock is empty, 
           Return True if it's empty, False otherwise"""
        return np.any(stock >= 0) == False

    def _new_strip_(self, list_stocks, prod, pos_x, prod_w):
        """This method make a new strip
           In case no more strip can be make in the current stock
           Find another best stock and make a strip in that stock
           Return True if new strip created, False otherwise"""
        
        # Update the stock that we will working on and its id if there is None
        if self.crn_stock_idx == -1 and self.crn_stock is None:
            self.crn_stock_idx, self.crn_stock = self._find_best_stock_(list_stocks, prod)
            # Check if stock found
            if self.crn_stock_idx == -1 and self.crn_stock is None:
                return False

        stock_w, stock_h = self._get_stock_size_(self.crn_stock)
        if pos_x + prod_w > stock_w:
            # Out stock limit, get new stock
            self.crn_stock_idx, self.crn_stock = self._find_best_stock_(list_stocks, prod)
            # Check if stock found
            if self.crn_stock_idx == -1 and self.crn_stock is None:
                return False
            pos_x = 0
        
        stock_w, stock_h = self._get_stock_size_(self.crn_stock)

        self.crn_strip_begin = pos_x
        self.crn_strip_end = pos_x + prod_w
        return True

    def _place_in_strip_(self, prod_size):
        """Place the product into the current strip in the current stock
           Return the x, y position of the placed product
           Return pos_x = pos_y = None if the product can't be place"""

        stock_w, stock_h = self._get_stock_size_(self.crn_stock)
        prod_w, prod_h = prod_size
        pos_x, pos_y = None, None

        # Find a position to place in the strip, the range is limited in the strip, not the stock
        for x in range(self.crn_strip_begin, self.crn_strip_end - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(self.crn_stock, (x, y), prod_size):
                    pos_x, pos_y = x, y
                    break
            if pos_x is not None and pos_y is not None:
                break
        return pos_x, pos_y
    
    def _place_smaller_in_strip_(self, observation):
        """Find a smaller product than the current product to place 
           into the lefted space in strip before creating a new strip
           Return the size of the smaller product and it's position
           Return None position in case no more product can be place"""

        list_prods = observation["products"]
        prod_size = [0, 0]
        pos_x, pos_y = None, None

        # Loop through the product list until a product is placed
        for i_prod, prod in enumerate(list_prods):
            if prod["quantity"] > 0:
                # self.crn_prod_idx is used for tracking the current smaller product
                # to decrease complexity
                if i_prod < self.crn_prod_idx:
                    continue
                else:
                    self.crn_prod_idx = i_prod

                prod_size = prod["size"]

                pos_x, pos_y = self._place_in_strip_(prod_size)
                if pos_x is not None and pos_y is not None:
                    break

        return prod_size, pos_x, pos_y
                
