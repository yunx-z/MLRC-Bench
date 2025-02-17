import numpy as np

from meltingpot.utils.policies.policy import Policy

class RandomPolicy(Policy):
    """ 
        Policy class for Meltingpot competition 
        About Populations:
            We will make multiple instances of this class for every focal agent
            If you want to sample different agents for every population/episode
            add the required required randomization in the "initial_state" function
    """
    def __init__(self, policy_id):
        # You can implement any init operations here or in setup()
        seed = 42
        self.rng = np.random.RandomState(seed)
        self.substrate_name = None
    
    def initial_state(self):
        """ Called at the beginning of every episode """
        state = None
        return state

    def step(self, timestep, prev_state):
        """ Returns random actions according to spec """
        action = self.rng.randint(7)
        state = None
        return action, None
    
    def close(self):
        """ Required by base class """
        pass