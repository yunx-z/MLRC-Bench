# import tensorflow as tf
# Need to run this before import meltingpot for TF1 background bots
# No need to do this during evaluation, the background bots will run in separate container
# tf.compat.v1.disable_eager_execution() 

import contextlib
from collections.abc import Iterator, Mapping
import pandas as pd
import os

from meltingpot.utils.evaluation import evaluation
from meltingpot.utils.policies.policy import Policy
from meltingpot.configs import scenarios as scenario_configs
from meltingpot import substrate

from my_policies.user_config import SubmissionPolicies, SubmissionRoles

@contextlib.contextmanager
def build_population(policy_class, policy_ids) -> Iterator[Mapping[str, Policy]]:
    """Builds a population from the specified saved models.

    Yields:
        A mapping from name to policy.
    """
    with contextlib.ExitStack() as stack:
        yield {
            p_id: stack.enter_context(policy_class(p_id))
            for p_id in policy_ids
        }

def get_substrate_name(scenario_name):
    if scenario_name in scenario_configs.SCENARIO_CONFIGS:
        return scenario_configs.SCENARIO_CONFIGS[scenario_name].substrate
    elif scenario_name in substrate.SUBSTRATES:
        return scenario_name
    else:
        raise NotImplementedError(f"No scenario named {scenario_name}")

def evaluate_meltingpot_config(scenario_name, num_episodes):
    substrate_name = get_substrate_name(scenario_name)
    submission_policy_class =  SubmissionPolicies[substrate_name]
    roles = substrate.get_config(substrate_name).default_player_roles
    policy_ids = [f"agent_{i}" for i in range(len(roles))]
    # unique_roles = set(roles)
    # names_by_role = {role: policy_ids for role in unique_roles}
    names_by_role = SubmissionRoles[substrate_name]
    with build_population(submission_policy_class, policy_ids) as population:
        results = evaluation.evaluate_population(
            population=population,
            names_by_role=names_by_role,
            scenario=scenario_name,
            num_episodes=num_episodes)
    return results

def normalize_scores(scores, minmax_scores):
    max_score, min_score = minmax_scores['max_score'], minmax_scores['min_score']
    norm_scores = (scores - min_score) / ( max_score - min_score)
    return norm_scores

def print_scores(scenario_name, results, minmax_scores):
    print(f"Results for {scenario_name=}")
    scores = results['focal_per_capita_return'].values
    norm_scores = normalize_scores(scores, minmax_scores[scenario_name])
    mean_score = norm_scores.mean()
    print(f"{scores=} \n {mean_score=:0.3f} \n {norm_scores=}")

if __name__ == '__main__':

    from mp2results import get_normalization_scores

    minmax_scores = get_normalization_scores()

    num_episodes = 1

    eval_configs = {
        # 'territory__rooms': num_episodes, 
        'territory__rooms_4': num_episodes,
        'territory__rooms_0': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena': num_episodes,
        'prisoners_dilemma_in_the_matrix__arena_0': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena_1': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena_2': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena_3': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena_4': num_episodes,
        # 'prisoners_dilemma_in_the_matrix__arena_5': num_episodes,
        # 'clean_up': num_episodes,
        'clean_up_2': num_episodes,
        # 'allelopathic_harvest__open': num_episodes,
        'allelopathic_harvest__open_0': num_episodes,
    }
    for scenario_name, num_ep in eval_configs.items():
        results = evaluate_meltingpot_config(scenario_name, num_ep)
        os.makedirs('local_eval_results/', exist_ok=True)
        results.to_csv(f'local_eval_results/{scenario_name}.csv', index=False)
        print_scores(scenario_name, results, minmax_scores)

    for scenario_name, num_ep in eval_configs.items():
        results = pd.read_csv(f'local_eval_results/{scenario_name}.csv')
        print_scores(scenario_name, results, minmax_scores)
    
