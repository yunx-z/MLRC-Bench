## How to write your own policies?

We recommend that you place the code for all your policies in the `my_policies` directory (though it is not mandatory). All your submissions should contain an Policy class, subclassed from `meltingpot.utils.policies.policy.Policy`. We have added random agent examples in [`random_policy.py`](random_policy.py)

**Add your per substrate Policy class name in** [`user_config.py`](user_config.py) in the `SubmissionPolicies` dictionary

## The Policy Class

The Policy class you define will be used as the common entrypoint to create your policies (per substrate). This class ensures a common interface for evaluations while allowing you the flexibility to implement as many different agents as you like. Each policy object will be re-initialized for every episode.

**If you would like a population of polciies per substrate. Please add the required randomization in the `__init__` function of the class, such that initilization picks up different agents for each episode**

The Agent class should contain the following functions:

1. `initial_state` - This is called at the start of each episode.
2. `step` - For the provided observation, return the policy's predicted actions

See [`random_policy.py`](random_policy.py) for an example.
