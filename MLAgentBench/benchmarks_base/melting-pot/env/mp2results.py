import pandas as pd
import numpy as np
import os

EXCLUDE_LIST = ['clean_up_0', 'clean_up_1']

def get_normalization_scores():

    if not os.path.exists('meltingpot-results-2.3.0.feather'):
        print("Please download the mp2.0 results first")
        print("> wget https://storage.googleapis.com/dm-meltingpot/meltingpot-results-2.3.0.feather")
        import sys; sys.exit(0)

    mp2res = pd.read_feather('meltingpot-results-2.3.0.feather') # Might require installing pyarrow==12.0.1

    substrate_names = [
        'clean_up',
        'territory__rooms',
        'prisoners_dilemma_in_the_matrix__arena',
        'allelopathic_harvest__open'
    ]

    res_df = mp2res[mp2res['substrate'].isin(substrate_names)]

    minmax_scores = {}
    for scenario in res_df['scenario'].unique():
        if scenario in EXCLUDE_LIST:
            continue
        algo_results = res_df[res_df['scenario'] == scenario].groupby('mapla')['focal_per_capita_return'].agg(np.mean)
        max_score = algo_results.max()
        min_score = algo_results.min()
        minmax_scores[scenario] = dict(
            max_score=max_score,
            min_score=min_score,
        )
    return minmax_scores

if __name__ == '__main__':
    minmax_scores = get_normalization_scores()
    for scenario, scores in minmax_scores.items():
        print(( 
                f"{scenario:45}" 
                f"{'Max result':15}" 
                f"{scores['max_score']:0.3f}\t"
                f"{'Min result':15}"
                f"{scores['min_score']:0.3f}"
        ))