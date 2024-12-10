from policy import Policy
from student_submissions.s2310737_2311628_2311659_2312130_2313200 import algor

class Policy2310737_2311628_2311659_2312130_2313200(Policy):
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2]

        self.policy = algor.WorstFit() if policy_id == 1 else algor.ColumnGeneration()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
