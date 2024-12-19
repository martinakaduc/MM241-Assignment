from policy import Policy
from . import QLearningPolicy, BestFit

        
class Policy2033766_2033528(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = BestFit()
        if policy_id == 2:
            self.policy = QLearningPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
