from gymnasium.envs.registration import register

register(
    id="gym_cutting_stock/CuttingStock-v0",
    entry_point="gym_cutting_stock.envs:CuttingStockEnv",
)
