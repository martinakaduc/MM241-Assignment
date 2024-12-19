import os
import numpy as np


def compute_filled_ratio(stocks):
    stocks_array = np.array(stocks)
    usable_area = np.sum(stocks_array == -1, axis=(1, 2))
    filled_area = np.sum(stocks_array > -1, axis=(1, 2))
    filled_ratios = np.zeros(len(stocks))
    filled_ratios[usable_area == 0] = 1.0
    unfilled_mask = usable_area > 0
    filled_ratios[unfilled_mask] = filled_area[unfilled_mask] / usable_area[unfilled_mask]

    return filled_ratios


def calculate_remaining_products(observation):
    return sum(product["quantity"] for product in observation["products"])


def get_number_of_used_stocks(stocks):
    stocks_array = np.array(stocks)
    filled_area = np.sum(stocks_array > -1, axis=(1, 2))
    return sum(filled_area > 0)


def compute_wasted_ratio(stocks):
    stocks_array = np.array(stocks)
    unused_area = np.sum(stocks_array == -1, axis=(1, 2))
    filled_area = np.sum(stocks_array > -1, axis=(1, 2))
    used_idx = filled_area > 0
    wasted_ratios = 1.0 - filled_area[used_idx] / (filled_area[used_idx] + unused_area[used_idx])
    return np.average(wasted_ratios)


class MLModel:
    # Directory to save the model
    model_dir = "student_submissions/saved_models"
    model_file = os.path.join(model_dir, "q_table.npy")
    metatdata_file = os.path.join(model_dir, "metadata.npy")

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    def save_model(self, policy):
        # Save Q-table
        np.save(self.model_file, policy.q_table)
        # Save metadata
        metadata = {
            "epsilon": policy.epsilon,
            "epsilon_decay": policy.epsilon_decay,
            "epsilon_min": policy.epsilon_min,
            "learning_rate": policy.learning_rate,
            "discount_factor": policy.discount_factor,
        }
        np.save(self.metatdata_file, metadata)
        print("Model saved successfully.")

    def load_model(self, policy):
        if os.path.exists(self.model_file) and os.path.exists(self.metatdata_file):
            # Load Q-table
            policy.q_table = np.load(self.model_file)
            # Load metadata
            metadata = np.load(self.metatdata_file, allow_pickle=True).item()
            policy.epsilon = metadata["epsilon"]
            policy.epsilon_decay = metadata["epsilon_decay"]
            policy.epsilon_min = metadata["epsilon_min"]
            policy.learning_rate = metadata["learning_rate"]
            policy.discount_factor = metadata["discount_factor"]
            print("Model loaded successfully.")
        else:
            print("No saved model found. Training from scratch.")
