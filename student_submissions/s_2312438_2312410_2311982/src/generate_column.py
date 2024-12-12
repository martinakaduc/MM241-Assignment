from policy import Policy
import numpy as np

class GenerateColumn(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = {}          # Track patterns for each stock
        self.used_patterns = set()  # Track used patterns
        self.current_stock = 0      # Track current stock
        self.used_positions = {}    # Track used positions per stock
        self.new_game = True        # Track new game
    
    def reset(self):
        self.patterns.clear()
        self.used_patterns.clear()
        self.current_stock = 0
        self.used_positions.clear()

    def _generate_patterns(self, products, stock_size):
        """Generate cutting patterns for a stock based on available products."""
        
        # Sort products by area (largest to smallest)
        sorted_products = sorted(enumerate(products), key=lambda x: -x[1]['size'][0] * x[1]['size'][1])
        patterns = []

        # Try to create patterns for each product
        for i, prod in sorted_products:
            if prod['quantity'] > 0:
                stock_w, stock_h = stock_size
                width, height = prod['size']

                # Skip if product is too large for the stock
                if width > stock_w or height > stock_h:
                    continue

                # Initialize pattern and remaining space
                pattern = [0] * len(products)
                remaining_w, remaining_h = stock_w, stock_h
                max_pieces_w = remaining_w // width
                max_pieces_h = remaining_h // height
                pieces = min(prod['quantity'], max_pieces_w * max_pieces_h)

                if pieces > 0:
                    pattern[i] = pieces

                    # Fill remaining space with other products
                    for j, other_prod in sorted_products:
                        if j != i and other_prod['quantity'] > 0:
                            w2, h2 = other_prod['size']
                            if w2 <= remaining_w and h2 <= remaining_h:
                                max_other = min(
                                    other_prod['quantity'],
                                    (remaining_w // w2) * (remaining_h // h2)
                                )
                                if max_other > 0:
                                    pattern[j] = max_other
                                    remaining_w -= w2 * max_other
                                    remaining_h -= h2 * max_other

                    patterns.append(pattern)
                    
        return patterns

    def get_action(self, observation, info):
        """Get the action to take based on the current observation."""
        # If it's a new game, reset the environment and start fresh
        if self.new_game:
            self.reset()  # Reset the environment or any necessary parameters
            self.new_game = False  # Set new_game to False to avoid resetting again

        stocks = observation["stocks"]  # List of stocks (probably locations or bins)
        products = observation["products"]  # List of products to be placed in stocks
        remaining_products = np.sum([p["quantity"] for p in products])  # Count how many products are remaining

        # If only one product is left, mark the game as new for the next iteration
        if remaining_products == 1:
            self.new_game = True

        # If no products are left, return a no-op action (do nothing)
        if not any(p["quantity"] > 0 for p in products):
            return {
                "stock_idx": -1,  # No stock
                "size": np.array([0, 0]),  # No product to place
                "position": np.array([0, 0])  # No position to place
            }

        # Sort stocks by the area of each stock (width * height)
        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True  # Sort in descending order of size
        )
        
        # Sort products by size (largest area first)
        products = sorted(
            products,
            key=lambda x: -x['size'][0] * x['size'][1]
        )

        # Loop through stocks starting from the current stock index
        for idx, _ in sorted_stocks[self.current_stock:]:
            # Initialize tracking for current stock if not already initialized
            if idx not in self.used_positions:
                self.used_positions[idx] = set()

            stock_size = self._get_stock_size_(stocks[idx])  # Get the size of the current stock
            
            # Generate or update patterns if none exists
            if not self.patterns:
                self.patterns = self._generate_patterns(products, stock_size)

            # Try to place the products using existing patterns
            for pattern_idx, pattern in enumerate(self.patterns):
                if pattern_idx not in self.used_patterns:  # Only try unused patterns
                    for prod_idx, pieces in enumerate(pattern):
                        if pieces > 0 and products[prod_idx]["quantity"] > 0:
                            # Find valid positions to place the current product in the stock
                            positions = self._find_feasible_positions(
                                stocks[idx],
                                products[prod_idx]["size"]
                            )
                            
                            # Loop through feasible positions
                            for pos in positions:
                                pos_key = (*pos, *products[prod_idx]["size"])
                                if pos_key not in self.used_positions[idx]:  # Check if the position is already used
                                    self.used_positions[idx].add(pos_key)  # Mark this position as used
                                    return {
                                        "stock_idx": idx,  # Return the stock index where the product will go
                                        "size": np.array(products[prod_idx]["size"]),  # Return the size of the product
                                        "position": np.array(pos)  # Return the position where the product is placed
                                    }

            # If the current stock is full, try the next stock
            self.current_stock += 1
            if self.current_stock >= len(stocks):
                # If all stocks are exhausted, generate new patterns for the remaining products
                self.current_stock = 0
                self.patterns = self._generate_patterns(products, stock_size)
                self.used_patterns.clear()  # Reset used patterns

        # If no valid placement found, return a no-op action (do nothing)
        return {
            "stock_idx": -1,  # No stock to place product
            "size": np.array([0, 0]),  # No product to place
            "position": np.array([0, 0])  # No position to place
        }
    
    def _find_feasible_positions(self, stock, product_size):
        """Find all feasible positions where a product can be placed on a stock."""
        
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product_size
        positions = []

        # Iterate through possible positions on the stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                # Check if the product can be placed at the current position
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    positions.append((x, y))

        return positions