from my_policies.random_policy import RandomPolicy

# Specify different policy classes for different substrates here
# The class names should be a Class that takes the policy_id from the meltingpot scenario
SubmissionPolicies = {
    "allelopathic_harvest__open": RandomPolicy,
    "clean_up": RandomPolicy,
    "prisoners_dilemma_in_the_matrix__arena": RandomPolicy,
    "territory__rooms": RandomPolicy,
}

# This should be a mapping of roles in a substrate to a list of policy_ids
# If all the entire population works for any role, this dictionary can be left unchanged
SubmissionRoles = {
    "allelopathic_harvest__open":  
        {
            'player_who_likes_red': [f'agent_{i}' for i in range(16)],
            'player_who_likes_green': [f'agent_{i}' for i in range(16)],
        },
    "clean_up":
        {
            'default': [f'agent_{i}' for i in range(7)],
        },
    "prisoners_dilemma_in_the_matrix__arena":
        {
            'default': [f'agent_{i}' for i in range(8)],
        },
    "territory__rooms":
        {
            'default': [f'agent_{i}' for i in range(9)],
        },
}