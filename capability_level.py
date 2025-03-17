
def get_capability_level(task, model):
    """
    read `idea_evals.json` and `test_idea_evals.json` from 8 runs under logs/{task}/{model}
    compute L1-L6 scores
    return a list of scores like [1,3,4,3,3,5,6,1]
    """
