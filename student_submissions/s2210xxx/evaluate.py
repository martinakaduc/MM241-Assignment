import gym_cutting_stock
import gymnasium as gym
import numpy as np  # Sử dụng NumPy để đo thời gian
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 10


def get_product_info(observation):
    """
    Calculate and return the number of products, their total area
    """
    pro_count = 0
    pro_area = 0
    for pro in observation["products"]:
        pro_count += pro["quantity"]
        pro_area += pro["size"][0] * pro["size"][1] * pro["quantity"]
    return pro_count, pro_area

def get_stock_size(stock):
    """
    Check if the stock is used or not
    If used, return its dimension
    """
    stock_h = 0
    stock_w = 0
    empty = 0
    if np.any(stock >= 0):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        empty = np.sum(stock == -1)
    return stock_w, stock_h, empty    
    
def get_stock_info(observation):
    """
    Calculate and return the number of used stocks, their total area
    """
    sto_count = 0
    sto_area = 0
    sto_full = 0
    for sto in observation["stocks"]:
        sto_w, sto_h, empty = get_stock_size(sto)
        if sto_h > 0 and sto_w > 0:
            sto_count += 1
            sto_area += sto_h * sto_w
        
        if empty == 0:
            sto_full += 1
    return sto_count, sto_area, sto_full
    

def test_policy(policy, ep):
    """
    Run the policy, measure execution time, and calculate its parameters in each episode.
    """
    observation, info = env.reset(seed=ep+9)
    # Get information about products
    pro_count, pro_area = get_product_info(observation)
    start_time = np.datetime64('now', 'ms')  # Start timing using NumPy
    while True:
        action = policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Calculate elapsed time for this episode
    elapsed_time = (np.datetime64('now', 'ms') - start_time) / np.timedelta64(1000, 'ms')
    # Get information about used stocks
    sto_count, sto_area, sto_full = get_stock_info(observation)
    file.write(f"\tN. of products:\t {pro_count}\n")
    file.write(f"\tTotal area of products:\t {pro_area}\n")
    file.write(f"\tN. of stocks:\t {sto_count}\n")
    file.write(f"\tTotal area of stocks:\t {sto_area}\n")
    file.write(f"\tReward:\t {reward}\n")
    file.write(f"\tTime:\t {elapsed_time:.2f}s\n")
    
    print(f"Running ep {ep}: {elapsed_time}, reward = {reward}\n")
    

if __name__ == "__main__":
    with open("evaluateResult.txt", "a") as file:  # Mở file để ghi kết quả
        ep = 0
        while ep < NUM_EPISODES:
            file.write(f"Episode {ep+1}-----------------------------\n")
            
            # Test GreedyPolicy
            file.write(f"----Greedy Policy----------------\n")
            greedy_policy = GreedyPolicy()
            test_policy(greedy_policy, ep)

            # Test RandomPolicy
            file.write(f"----Random Policy----------------\n")
            random_policy = RandomPolicy()
            test_policy(random_policy, ep)

            # Test Policy2313109
            file.write(f"----Dynamic Programming Policy---\n")
            policy2313109 = Policy2210xxx(policy_id=2)
            test_policy(policy2313109, ep)

            # Test Policy2211960
            file.write(f"----First-Fit Decreasing Policy--\n")
            policy2211960 = Policy2210xxx(policy_id=1)
            test_policy(policy2211960, ep)
            ep += 1
    
    env.close()
