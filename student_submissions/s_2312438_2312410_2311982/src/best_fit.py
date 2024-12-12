from policy import Policy
import numpy as np

class BestFit(Policy):
    def __init__(self):
        self.used_stocks = {}  # Track utilization per stock
        self.filled_spaces = {}  # Track filled spaces per stock
        self.min_waste_threshold = 0.3  # Minimum waste threshold
        self.max_height_ratio = 0.9  # Max height ratio to allow new stock

    def _get_skyline(self, stock_idx, stock):
        """
        Get the skyline profile for a given stock.

        The skyline represents the heights of stacked items in filled spaces, tracking the highest point at each horizontal position. This method computes the skyline by iterating over filled spaces and updating the heights based on the stock's dimensions.

        Args:
            stock_idx (int): The index of the stock.
            stock (object): The stock object containing its size and attributes.

        Returns:
            list: The height of the skyline at each horizontal position.
        """

        # Ensure the filled spaces for the current stock index are initialized
        if stock_idx not in self.filled_spaces:
            self.filled_spaces[stock_idx] = []
            
        # Get the dimensions (width and height) of the current stock
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Initialize the skyline array with zeros for each horizontal position
        skyline = [0] * stock_w
        
        # Iterate over the filled spaces for the current stock index and update the skyline
        for x, y, w, h in self.filled_spaces[stock_idx]:
            for i in range(x, x + w):
                # Update the maximum height at each position covered by the filled stock
                skyline[i] = max(skyline[i], y + h)
                
        # Return the final skyline profile
        return skyline


    def _find_best_position(self, stock_idx, stock, product_size):
        """
        Find the optimal position for placing a product using the skyline strategy.

        This method iterates through possible product orientations and positions 
        along the skyline to find the one that minimizes space waste, ensuring 
        the product fits within the stock's dimensions.

        Args:
            stock_idx (int): Index of the stock to place the product in.
            stock (object): The stock object.
            product_size (tuple): Dimensions (width, height) of the product.

        Returns:
            tuple: Best position (x, y), orientation (width, height), and waste score.
        """
        # Retrieve the stock's width and height
        stock_w, stock_h = self._get_stock_size_(stock)

        # Unpack the product's width and height from the given size tuple
        prod_w, prod_h = product_size

        # Get the current skyline profile for the stock
        skyline = self._get_skyline(stock_idx, stock)

        # Initialize variables to track the best position and the associated waste
        best_pos = None
        min_waste = float('inf')
        best_orientation = None

        # Try both possible orientations for the product (if applicable, you could extend this)
        for w, h in [(prod_w, prod_h)]:
            
            # Skip the product if it doesn't fit within the stock's dimensions
            if w > stock_w or h > stock_h:
                continue
            
            # Iterate through possible horizontal positions (x) within the stock's width
            for x in range(stock_w - w + 1):
                # Find the maximum height at the given position over the width of the product
                y = max(skyline[x:x + w])
                
                # Skip if the product cannot fit vertically (overflows the stock height)
                if y + h > stock_h:
                    continue
                
                # Check if the product can be placed at the current (x, y) position
                if self._can_place_(stock, (x, y), (w, h)):
                    # Calculate the waste score for placing the product at this position
                    waste = self._calculate_waste_score(
                        stock, x, y, w, h,
                        self.used_stocks.get(stock_idx, 0)
                    )
                    
                    # Update the best position if the current placement results in less waste
                    if waste < min_waste:
                        min_waste = waste
                        best_pos = (x, y)
                        best_orientation = (w, h)

        # Return the best position, orientation, and the associated waste score
        return best_pos, best_orientation, min_waste
    
    def find_edge_position(self, stock):
        """
        Find the edge position of the stock.

        This method identifies the first empty position (represented by -1) in the stock grid 
        along the x (horizontal) and y (vertical) axes, where the product can be placed.

        Args:
            stock (2D array): The stock grid where placement is considered, with empty spaces marked by -1.

        Returns:
            tuple: The (x, y) coordinates of the first empty position in the stock grid.
        """
        # Get the width and height of the stock
        stock_w, stock_h = self._get_stock_size_(stock)

        # Initialize the coordinates to (0, 0), which represents the start of the grid
        x, y = 0, 0

        # Iterate through the columns of the stock grid to find the first empty position along the x-axis
        for i in range(stock_w):
            # Check if there is an empty space in the current column (if any value is -1)
            if np.any(stock[i] == -1):
                x = i  # Set the x-coordinate to the column where the first empty space is found
                break  # Exit the loop once the first empty position is found

        # Iterate through the rows of the stock grid to find the first empty position along the y-axis
        for j in range(stock_h):
            # Check if there is an empty space in the current row (if any value is -1)
            if np.any(stock[:, j] == -1):
                y = j  # Set the y-coordinate to the row where the first empty space is found
                break  # Exit the loop once the first empty position is found

        # Return the (x, y) coordinates of the first empty position in the stock grid
        return x, y

    def _calculate_waste_score(self, stock, x, y, w, h, utilization):
        """
        Calculate the waste score based on multiple factors such as area utilization, 
        height usage, edge alignment, and current stock utilization.

        Args:
            stock (object): The stock where the product is being placed.
            x (int): The x-coordinate of the placement.
            y (int): The y-coordinate of the placement.
            w (int): The width of the product.
            h (int): The height of the product.
            utilization (float): Current stock utilization.

        Returns:
            float: The calculated waste score.
        """
        # Get the stock dimensions
        stock_w, stock_h = self._get_stock_size_(stock)

        # Find the first empty edge position in the stock grid
        x_edge, y_edge = self.find_edge_position(stock)

        # Calculate area utilization ratio
        area_ratio = (w * h) / (stock_w * stock_h)

        # Calculate height utilization and penalty if it exceeds the maximum allowed height ratio
        height_ratio = (y + h) / stock_h
        height_penalty = 0.4 * height_ratio if height_ratio > self.max_height_ratio else 0

        # Calculate edge alignment bonus (0.2 if aligned with edge, penalties for certain conditions)
        edge_bonus = 0
        if x == x_edge or y == y_edge:
            edge_bonus = 0.2
        if x == x_edge and y > y_edge:
            edge_bonus -= 0.1
        if y == y_edge and x > x_edge:
            edge_bonus -= 0.1

        # Calculate current stock utilization bonus
        util_bonus = 1 * utilization if utilization > 0 else 0

        # Return the final waste score considering all factors
        return (1 - area_ratio) + height_penalty - edge_bonus - util_bonus

    def get_action(self, observation, info):
        """
        Determine the best action to take based on the current observation of stocks and products.

        The function selects a product and places it on an appropriate stock, 
        minimizing the waste by evaluating multiple factors like product size, stock size, 
        and current stock utilization. It returns the optimal stock index, size, and position 
        for the selected product placement.

        Args:
            observation (dict): The current state of stocks and products.
            info (dict): Additional information (not used in this method).

        Returns:
            dict: The selected action with "stock_idx", "size", and "position".
        """
        
        # Extract available stocks and products
        stocks = observation["stocks"]
        products = [p for p in observation["products"] if p["quantity"] > 0]
        
        # If no products are available, return a no-op action
        if not products:
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        # Sort products by area (largest first), then by dimensions (longest and shortest sides)
        products.sort(key=lambda x: (
            -x['size'][0] * x['size'][1],  # Area
            -max(x['size']),  # Longest side
            -min(x['size'])   # Shortest side
        ))

        best_action = None
        min_global_waste = float('inf')

        # Select the first available product
        current_product = next(p for p in products if p["quantity"] > 0)

        # Sort stocks by area (largest first)
        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )

        # Try placing the product in each stock
        for stock_idx, stock in sorted_stocks:
            for current_product in products:
                stock_w, stock_h = self._get_stock_size_(stock)

                # Skip stocks that are too small for the current product
                if stock_w < min(current_product['size']) or stock_h < min(current_product['size']):
                    continue

                # Find the best position to place the product in the stock
                pos, size, waste = self._find_best_position(
                    stock_idx, stock, current_product['size']
                )

                # Update best action if current placement has lower waste
                if pos and waste < min_global_waste:
                    min_global_waste = waste
                    best_action = {
                        "stock_idx": stock_idx,
                        "size": np.array(size),
                        "position": np.array(pos)
                    }

                    # Early exit if the waste is below the threshold
                    if waste < self.min_waste_threshold:
                        break

            # If a valid action is found, update the stock usage and filled spaces
            if best_action:
                stock_idx = best_action["stock_idx"]
                x, y = best_action["position"]
                w, h = best_action["size"]

                # Update the filled spaces for the stock
                if stock_idx not in self.filled_spaces:
                    self.filled_spaces[stock_idx] = []
                self.filled_spaces[stock_idx].append((x, y, w, h))

                # Update the used stock utilization
                stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])
                self.used_stocks[stock_idx] = self.used_stocks.get(stock_idx, 0) + \
                                            (w * h) / (stock_w * stock_h)

                return best_action

        # Return a no-op action if no valid placement was found
        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }


    def _can_place_product(self, stock, x, y, w, h):
        """Check if a product can be placed at a given position on the stock."""
        
        # Ensure the product fits within the stock boundaries
        if x + w > stock.shape[0] or y + h > stock.shape[1]:
            return False
        
        # Check if the space is empty (represented by -1)
        return np.all(stock[x:x + w, y:y + h] == -1)