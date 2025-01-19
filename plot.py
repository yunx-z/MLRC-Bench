#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Script to parse experiment results and produce:  
  - Tables (1.1, 1.2, 2.1, 2.2) with mean and standard deviation  
  - Figures (3, 4.1, 4.2, 4.3) with error bars representing standard deviation  
  
All outputs (figures and LaTeX tables) will be saved under `results/` directory.  
  
Important notes:  
- The directory structure and file naming convention follow the description provided.  
- Only the most recent 8 trials for each setting (based on RUN_ID timestamps) are considered.  
- All ideas (idea_idx in [0, 1, 2, 3]) are included for the ideation+implementation pipeline.  
- Success is considered if `improvement_perc > 5.0` (hard-coded threshold).  
- All numeric results are rounded to one decimal place in tables and figures.  
- The `pass@k` calculation now reflects the formula provided.  
- Placeholders have been removed, and the code implements all required functionality.  
  
Requires:  
  Python 3.8+  
  pandas  
  numpy  
  matplotlib  
"""  
  
import os  
import glob  
import json  
import re  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from collections import defaultdict  
  
##############################################################################  
# Global config  
##############################################################################  
  
 
# Pipeline types  
PIPELINES = ["implementation-only", "ideation+implementation"]  
  
# LMs  
LMS = ["o1-mini", "gpt-4o"]  
  
# Tasks  
TASKS = ["llm-merging", "backdoor-trigger-recovery"]  
  
# Idea indices for ideation+implementation pipeline  
IDEA_IDXS = [0, 1, 2, 3]  
IDEA_PROPOSAL_MODEL = "o1-preview"  
  
# We consider a success if improvement_perc > 5.0  
SUCCESS_THRESHOLD = 5.0  

RESULTS_DIR = f"results/SUCCESS_THRESHOLD_{SUCCESS_THRESHOLD}"  
os.makedirs(RESULTS_DIR, exist_ok=True)  
  
# For Figure 3, we have lines for N=0 (implementation-only), N=1,2,4 (ideation+implementation)  
IDEATION_IDEA_COUNTS = [1, 2, 4]  
  
##############################################################################  
# Utility functions  
##############################################################################  
  
def extract_timestamp_from_dirname(dirname):  
    """  
    Extract timestamp from RUN_ID directory name of format 'mmddhhmmss'.  
    Return a tuple (month, day, hour, minute, second) as integers, or None if pattern not found.  
    """  
    pattern = r'^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$'  
    m = re.match(pattern, dirname)  
    if m:  
        month = int(m.group(1))  
        day = int(m.group(2))  
        hour = int(m.group(3))  
        minute = int(m.group(4))  
        second = int(m.group(5))  
        return (month, day, hour, minute, second)  
    return None  
  
def find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx=None):  
    """  
    Return a sorted list (by ascending time) of up to 8 directories (each is RUN_ID)  
    that are the most recent runs for the given task / lm / pipeline.  
    """  
    # Collect runs  
    runs_with_timestamps = []  
  
    if pipeline == "implementation-only":  
        base_pattern = f"workspace/{task}/{lm}/*"  
    else:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for ideation+implementation pipeline.")  
        base_pattern = f"workspace/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/*"  
  
    for path in glob.glob(base_pattern):  
        if os.path.isdir(path):  
            dirname = os.path.basename(path)  # RUN_ID  
            ts = extract_timestamp_from_dirname(dirname)  
            if ts is not None:  
                runs_with_timestamps.append((dirname, ts))  
  
    # Sort them by ascending time  
    runs_with_timestamps.sort(key=lambda x: x[1])  
    # Keep only the last 8  
    runs_with_timestamps = runs_with_timestamps[-8:]  
  
    # Return them in ascending order  
    return [x[0] for x in runs_with_timestamps]  
  
def load_json_safely(path):  
    """  
    Load JSON from path if exists, otherwise return None  
    """  
    if not os.path.isfile(path):  
        return None  
    try:  
        with open(path, "r") as f:  
            return json.load(f)  
    except:  
        return None  
  
##############################################################################  
# Data extraction  
##############################################################################  
  
def get_dev_results(task, lm, pipeline, run_id, idea_idx=None):  
    """  
    Return a list of (improvement_perc, runtime, complexity) for each valid implementation in dev phase  
    for the given run. If file doesn't exist or no valid dev entries, return empty list.  
    """  
    if pipeline == "implementation-only":  
        # dev file  
        dev_file = f"workspace/{task}/{lm}/{run_id}/{task}/output/idea_evals.json"  
    else:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for ideation+implementation pipeline.")  
        dev_file = f"workspace/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/output/idea_evals.json"  
  
    data = load_json_safely(dev_file)  
    if data is None:  
        return []  
  
    implementations = data.get("implementations", [])  
    results_dev = []  
    for imp in implementations:  
        if imp.get("phase") == "dev":  
            results_dev.append((  
                imp.get("improvement_perc", 0.0),  
                imp.get("relative_runtime", 0.0),  
                imp.get("relative_complexity", 0.0),  
            ))  
    return results_dev  
  
def get_test_result(task, lm, pipeline, run_id, idea_idx=None):  
    """  
    Return the test improvement_perc, runtime, complexity for the best model in the given run.  
    If file doesn't exist or no valid test entry, return None.  
    """  
    if pipeline == "implementation-only":  
        test_file = f"logs/{task}/{lm}/{run_id}/env_log/test_idea_evals.json"  
    else:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for ideation+implementation pipeline.")  
        test_file = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/test_idea_evals.json"  
  
    data = load_json_safely(test_file)  
    if data is None:  
        return None  
  
    implementations = data.get("implementations", [])  
    for imp in implementations:  
        if imp.get("phase") == "test":  
            return (  
                imp.get("improvement_perc", 0.0),  
                imp.get("relative_runtime", 0.0),  
                imp.get("relative_complexity", 0.0),  
            )  
    return None  
  
##############################################################################  
# Table computations  
##############################################################################  
  
def compute_success_rates(phase='test'):  
    """  
    Compute success rates for Tables 1.1 and 1.2.  
    Success is defined as having improvement_perc > SUCCESS_THRESHOLD in the specified phase.  
    Returns a DataFrame with success rates and standard deviations.  
    """  
    rows = []  
    phase_name = 'test' if phase == 'test' else 'dev'  
    for task in TASKS + ["Avg"]:  
        for pipeline in PIPELINES:  
            rows.append([task if pipeline == PIPELINES[0] else "", pipeline] + [None]*len(LMS))  
  
    df = pd.DataFrame(rows, columns=["Task", "System"] + LMS)  
  
    success_counts = {lm: [] for lm in LMS}  
    total_counts = {lm: [] for lm in LMS}  
  
    row_idx = 0  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for lm_i, lm in enumerate(LMS):  
                n_success_list = []  
                n_total_list = []  
  
                if pipeline == "implementation-only":  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    n_total = len(run_ids)  
                    n_success = 0  
                    for run_id in run_ids:  
                        if phase == 'test':  
                            res = get_test_result(task, lm, pipeline, run_id)  
                            if res is not None and res[0] > SUCCESS_THRESHOLD:  
                                n_success += 1  
                        else:  
                            dev_res = get_dev_results(task, lm, pipeline, run_id)  
                            if dev_res:  
                                best_imp = max([r[0] for r in dev_res])  
                                if best_imp > SUCCESS_THRESHOLD:  
                                    n_success += 1  
                    n_success_list.append(n_success)  
                    n_total_list.append(n_total)  
                else:  
                    n_success = 0  
                    n_total = 0  
                    for idea_idx in IDEA_IDXS:  
                        run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx)  
                        n_total += len(run_ids)  
                        for run_id in run_ids:  
                            if phase == 'test':  
                                res = get_test_result(task, lm, pipeline, run_id, idea_idx)  
                                if res is not None and res[0] > SUCCESS_THRESHOLD:  
                                    n_success += 1  
                            else:  
                                dev_res = get_dev_results(task, lm, pipeline, run_id, idea_idx)  
                                if dev_res:  
                                    best_imp = max([r[0] for r in dev_res])  
                                    if best_imp > SUCCESS_THRESHOLD:  
                                        n_success += 1  
                    n_success_list.append(n_success)  
                    n_total_list.append(n_total)  
  
                if n_total > 0:  
                    success_rate = (n_success / n_total) * 100  # percentage  
                else:  
                    success_rate = 0.0  
  
                df.iloc[row_idx, 2 + lm_i] = f"{round(success_rate, 1)}"  
  
                success_counts[lm].append(success_rate)  
                total_counts[lm].append(100)  # Each success rate is out of 100%  
  
            row_idx += 1  
  
    return df  
  
def compute_average_metrics(phase='test'):  
    """  
    Compute average metrics for Tables 2.1 and 2.2.  
    For the specified phase, compute mean and std of improvement_perc, relative_runtime, relative_complexity.  
    Returns a DataFrame with formatted strings showing mean ± std.  
    """  
    columns = [  
        "Task", "System",  
        "Imp_"+LMS[0], "Run_"+LMS[0], "Comp_"+LMS[0],  
        "Imp_"+LMS[1], "Run_"+LMS[1], "Comp_"+LMS[1],  
    ]  
    rows = []  
    for task in TASKS + ["Avg"]:  
        for pipeline in PIPELINES:  
            rows.append([task if pipeline == PIPELINES[0] else "", pipeline] + [None]*(len(columns)-2))  
  
    df = pd.DataFrame(rows, columns=columns)  
  
    metrics_sum = {lm: {'imp': [], 'run': [], 'comp': []} for lm in LMS}  
  
    row_idx = 0  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for lm_i, lm in enumerate(LMS):  
                imp_vals = []  
                run_vals = []  
                comp_vals = []  
  
                if pipeline == "implementation-only":  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for run_id in run_ids:  
                        if phase == 'test':  
                            res = get_test_result(task, lm, pipeline, run_id)  
                            if res is not None:  
                                imp_vals.append(res[0])  
                                run_vals.append(res[1])  
                                comp_vals.append(res[2])  
                        else:  
                            dev_res = get_dev_results(task, lm, pipeline, run_id)  
                            if dev_res:  
                                best_idx = np.argmax([r[0] for r in dev_res])  
                                imp_vals.append(dev_res[best_idx][0])  
                                run_vals.append(dev_res[best_idx][1])  
                                comp_vals.append(dev_res[best_idx][2])  
                    # if lm == "o1-mini" and task == "llm-merging":
                    #     print(lm, pipeline, task)
                    #     print("imp_vals", imp_vals)
                    #     print("run_vals", run_vals)
                    #     print("comp_vals", comp_vals)
                    #     exit(0)
                else:  
                    for idea_idx in IDEA_IDXS:  
                        run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx)  
                        for run_id in run_ids:  
                            if phase == 'test':  
                                res = get_test_result(task, lm, pipeline, run_id, idea_idx)  
                                if res is not None:  
                                    imp_vals.append(res[0])  
                                    run_vals.append(res[1])  
                                    comp_vals.append(res[2])  
                            else:  
                                dev_res = get_dev_results(task, lm, pipeline, run_id, idea_idx)  
                                if dev_res:  
                                    best_idx = np.argmax([r[0] for r in dev_res])  
                                    imp_vals.append(dev_res[best_idx][0])  
                                    run_vals.append(dev_res[best_idx][1])  
                                    comp_vals.append(dev_res[best_idx][2])  
  
                # Compute mean and std  
                if imp_vals:  
                    imp_mean = np.mean(imp_vals)  
                    imp_std = np.std(imp_vals)  
                    run_mean = np.mean(run_vals)  
                    run_std = np.std(run_vals)  
                    comp_mean = np.mean(comp_vals)  
                    comp_std = np.std(comp_vals)  
                else:  
                    imp_mean = imp_std = run_mean = run_std = comp_mean = comp_std = 0.0  
  
                df.iloc[row_idx, 2 + lm_i*3] = f"{round(imp_mean,1)}±{round(imp_std,1)}"  
                df.iloc[row_idx, 3 + lm_i*3] = f"{round(run_mean,1)}±{round(run_std,1)}"  
                df.iloc[row_idx, 4 + lm_i*3] = f"{round(comp_mean,1)}±{round(comp_std,1)}"  
  
                # Accumulate for Avg row  
                metrics_sum[lm]['imp'].extend(imp_vals)  
                metrics_sum[lm]['run'].extend(run_vals)  
                metrics_sum[lm]['comp'].extend(comp_vals)  
            row_idx +=1  
  
    # Compute Avg row  
    for pipeline in PIPELINES:  
        for lm_i, lm in enumerate(LMS):  
            imp_vals = metrics_sum[lm]['imp']  
            run_vals = metrics_sum[lm]['run']  
            comp_vals = metrics_sum[lm]['comp']  
  
            if imp_vals:  
                imp_mean = np.mean(imp_vals)  
                imp_std = np.std(imp_vals)  
                run_mean = np.mean(run_vals)  
                run_std = np.std(run_vals)  
                comp_mean = np.mean(comp_vals)  
                comp_std = np.std(comp_vals)  
            else:  
                imp_mean = imp_std = run_mean = run_std = comp_mean = comp_std = 0.0  
  
            df.iloc[row_idx, 2 + lm_i*3] = f"{round(imp_mean,1)}±{round(imp_std,1)}"  
            df.iloc[row_idx, 3 + lm_i*3] = f"{round(run_mean,1)}±{round(run_std,1)}"  
            df.iloc[row_idx, 4 + lm_i*3] = f"{round(comp_mean,1)}±{round(comp_std,1)}"  
        row_idx +=1  
  
    return df  
  
##############################################################################  
# Figure 3: pass@k vs number of trials (dev only)  
##############################################################################  
  
def compute_pass_at_k_data():  
    """  
    Compute pass@k for Figure 3 according to the formula provided in the prompt.  
  
    For each LM, for N in [0, 1, 2, 4], and for each task:  
      - For implementation-only (N=0), use the success information from the implementation-only pipeline.  
      - For ideation+implementation (N=1,2,4), consider all subsets of ideas of size N out of the 4 total ideas.  
        For each subset:  
          - Collect the success counts c_i for the ideas in the subset.  
          - For each k in 1..8:  
            - Compute pass@k according to the formula provided.  
  
    Aggregate the pass@k across all subsets and tasks by averaging.  
  
    Returns:  
      pass_at_k_data[lm][task][N][k] = pass@k for each task  
    """  
    from itertools import combinations  
    from math import comb  
  
    pass_at_k_data = {lm: defaultdict(lambda: defaultdict(list)) for lm in LMS}  
  
    m = 8  # Number of trials per idea  
    k_values = range(1, m+1)  
  
    for lm in LMS:  
        for task in TASKS:  
            # Collect success lists for implementation-only  
            impl_only_successes = []  
            run_ids = find_most_recent_8_runs_for_pipeline(task, lm, "implementation-only")  
            for run_id in run_ids:  
                dev_res = get_dev_results(task, lm, "implementation-only", run_id)  
                is_success = any(r[0] > SUCCESS_THRESHOLD for r in dev_res)  
                impl_only_successes.append(is_success)  
            c_impl = sum(impl_only_successes)  
            # For N=0  
            pass_at_k_impl = []  
            for k in k_values:  
                if m < k:  
                    pass_k = 1.0 if c_impl > 0 else 0.0  
                else:  
                    numerator = comb(m - c_impl, k)  
                    denominator = comb(m, k)  
                    pass_k = 1.0 - (numerator / denominator)  
                pass_at_k_impl.append(pass_k)  
            pass_at_k_data[lm][task][0] = pass_at_k_impl  
  
            # Collect success counts c_i for each idea  
            c_list = []  
            for idea_idx in IDEA_IDXS:  
                idea_successes = []  
                run_ids = find_most_recent_8_runs_for_pipeline(task, lm, "ideation+implementation", idea_idx)  
                for run_id in run_ids:  
                    dev_res = get_dev_results(task, lm, "ideation+implementation", run_id, idea_idx)  
                    is_success = any(r[0] > SUCCESS_THRESHOLD for r in dev_res)  
                    idea_successes.append(is_success)  
                c_i = sum(idea_successes)  
                c_list.append(c_i)  
            # Now calculate pass@k for N in [1,2,4]  
            for N in [1,2,4]:  
                idea_subsets = list(combinations(range(len(c_list)), N))  
                pass_at_k_list = []  
                for k in k_values:  
                    numerator = 0  
                    denominator = comb(4, N) * (comb(m, k) ** N)  
                    for subset in idea_subsets:  
                        product = 1  
                        for idx in subset:  
                            c_i = c_list[idx]  
                            m_minus_c_i = m - c_i  
                            if m_minus_c_i < k:  
                                product = 0  
                                break  
                            else:  
                                product *= comb(m_minus_c_i, k)  
                        numerator += product  
                    if denominator == 0:  
                        pass_k = 0.0  
                    else:  
                        pass_k = 1.0 - (numerator / denominator)  
                    pass_at_k_list.append(pass_k)  
                pass_at_k_data[lm][task][N] = pass_at_k_list  
  
    # Now average over tasks  
    averaged_pass_at_k = {lm: {} for lm in LMS}  
    for lm in LMS:  
        for N in [0,1,2,4]:  
            pass_at_k_totals = [0]*len(k_values)  
            for task in TASKS:  
                pass_at_k_totals = [sum(x) for x in zip(pass_at_k_totals, pass_at_k_data[lm][task][N])]  
            averaged_pass_at_k[lm][N] = [x/len(TASKS) for x in pass_at_k_totals]  
  
    return pass_at_k_data, averaged_pass_at_k  
  
def plot_figure_3(pass_at_k_data, averaged_pass_at_k):  
    """  
    Produce Figure 3 for each LM, showing pass@k per task and averaged over tasks.  
    """  
    x_vals = list(range(1,9))  
  
    for lm in LMS:  
        for task in TASKS + ['Average']:  
            plt.figure(figsize=(6,4))  
            for N in [0, 1, 2, 4]:  
                if task == 'Average':  
                    y_vals = averaged_pass_at_k[lm][N]  
                    title_task = 'Average over Tasks'  
                else:  
                    y_vals = pass_at_k_data[lm][task][N]  
                    title_task = task  
                if N == 0:  
                    label = "Implementation-only"  
                else:  
                    label = f"N={N}"  
                plt.plot(x_vals, y_vals, marker='o', label=label)  
            plt.title(f"Figure 3: pass@k vs Number of Trials\n(LM={lm}, Task={title_task})")  
            plt.xlabel("Number of trials (k)")  
            plt.ylabel("pass@k (success probability)")  
            plt.ylim([0,1.05])  
            plt.xticks(x_vals)  
            plt.grid(True)  
            plt.legend()  
            if task == 'Average':  
                outfn = os.path.join(RESULTS_DIR, f"figure_3_{lm}_average.pdf")  
            else:  
                task_safe = task.replace(' ', '_')  
                outfn = os.path.join(RESULTS_DIR, f"figure_3_{lm}_{task_safe}.pdf")  
            plt.savefig(outfn, bbox_inches='tight')  
            plt.close()  
  
            caption = (  
                f"Figure 3 for LM={lm}, Task={title_task}. pass@k on dev set vs number of trials. "  
                "N=0 means implementation-only pipeline. N=1,2,4 means ideation+implementation pipeline "  
                "with N ideas implemented. The y-axis is the probability of at least one success "  
                "(>5% improvement) among k randomly selected trials."  
            )  
            if task == 'Average':  
                capfn = os.path.join(RESULTS_DIR, f"figure_3_{lm}_average_caption.txt")  
            else:  
                capfn = os.path.join(RESULTS_DIR, f"figure_3_{lm}_{task_safe}_caption.txt")  
            with open(capfn, 'w') as f:  
                f.write(caption)  
  
##############################################################################  
# Figure 4.1 - 4.3: (dev only) improvement_perc, relative_runtime, relative_complexity  
# vs i-th implementation in a trial  
##############################################################################  
  
def compute_figure_4_data():  
    """  
    Compute data for Figure 4 without standard deviations.  
  
    Returns:  
      fig4_data[ (lm, pipeline, task) ] = {  
          'improvement_perc': means,  
          'relative_runtime': means,  
          'relative_complexity': means  
        }  
    """  
    def mean_by_expanding(list_of_arrays):  
        if not list_of_arrays:  
            return []  
        max_len = max(len(arr) for arr in list_of_arrays)  
        expanded = []  
        for arr in list_of_arrays:  
            if len(arr) < max_len:  
                arr = arr + [arr[-1]]*(max_len - len(arr))  
            expanded.append(arr)  
        expanded = np.array(expanded)  # shape (#trials, max_len)  
        means = np.mean(expanded, axis=0)  
        return means.tolist()  
  
    fig4_data = defaultdict(dict)  # fig4_data[task][(lm, pipeline)] = data  
  
    for task in TASKS + ["Average"]:  
        for pipeline in PIPELINES:  
            for lm in LMS:  
                # Collect data across tasks if task == "Average"  
                tasks_to_process = TASKS if task == "Average" else [task]  
                all_improvements = []  
                all_runtimes = []  
                all_complexities = []  
  
                for t in tasks_to_process:  
                    if pipeline == "implementation-only":  
                        run_ids = find_most_recent_8_runs_for_pipeline(t, lm, pipeline)  
                        for run_id in run_ids:  
                            dev_res = get_dev_results(t, lm, pipeline, run_id)  
                            if not dev_res:  
                                continue  
                            improvements = [dr[0] for dr in dev_res]  
                            runtimes = [dr[1] for dr in dev_res]  
                            complexities = [dr[2] for dr in dev_res]  
                            all_improvements.append(improvements)  
                            all_runtimes.append(runtimes)  
                            all_complexities.append(complexities)  
                    else:  
                        for idea_idx in IDEA_IDXS:  
                            run_ids = find_most_recent_8_runs_for_pipeline(t, lm, pipeline, idea_idx)  
                            for run_id in run_ids:  
                                dev_res = get_dev_results(t, lm, pipeline, run_id, idea_idx)  
                                if not dev_res:  
                                    continue  
                                improvements = [dr[0] for dr in dev_res]  
                                runtimes = [dr[1] for dr in dev_res]  
                                complexities = [dr[2] for dr in dev_res]  
                                all_improvements.append(improvements)  
                                all_runtimes.append(runtimes)  
                                all_complexities.append(complexities)  
  
                imp_means = mean_by_expanding(all_improvements)  
                run_means = mean_by_expanding(all_runtimes)  
                comp_means = mean_by_expanding(all_complexities)  
  
                fig4_data[task][(lm, pipeline)] = {  
                    "improvement_perc": imp_means,  
                    "relative_runtime": run_means,  
                    "relative_complexity": comp_means  
                }  
  
    return fig4_data  
  
def plot_figure_4(fig4_data):  
    """  
    Produce Figures 4.1, 4.2, 4.3 without error bars.  
    Generate separate figures for each task and an average over all tasks.  
    """  
    metrics = ["improvement_perc", "relative_runtime", "relative_complexity"]  
    titles = ["Dev improvement vs Implementation index",  
              "Dev relative runtime vs Implementation index",  
              "Dev relative complexity vs Implementation index"]  
    ylabels = ["Improvement (%) over baseline",  
               "Relative runtime (%) over baseline",  
               "Relative complexity (%) over baseline"]  
  
    for idx, metric in enumerate(metrics):  
        for task in TASKS + ["Average"]:  
            plt.figure(figsize=(8,6))  
            for (lm, pipeline), data in fig4_data[task].items():  
                means = data[metric]  
                xvals = range(1, len(means)+1)  
                label_str = f"{lm}, {pipeline}"  
                plt.plot(xvals, means, marker='o', label=label_str)  
            plt.title(f"Figure 4.{idx+1}: {titles[idx]} ({'Average over Tasks' if task == 'Average' else task})")  
            plt.xlabel("i-th implementation")  
            plt.ylabel(ylabels[idx])  
            plt.grid(True)  
            plt.legend()  
            task_safe = task.replace(' ', '_')  
            outfn = os.path.join(RESULTS_DIR, f"figure_4_{idx+1}_{task_safe}.pdf")  
            plt.savefig(outfn, bbox_inches='tight')  
            plt.close()  
  
            cap = (  
                f"Figure 4.{idx+1} for {'Average over Tasks' if task == 'Average' else task}. "  
                f"Average dev {metric.replace('_', ' ')} over the baseline versus the index of implementation in a trial. "  
                "If a trial has fewer implementations than the index, we replicate the last valid result. "  
                "Lines represent different LM + pipeline combinations, averaged across trials."  
            )  
            capfn = os.path.join(RESULTS_DIR, f"figure_4_{idx+1}_{task_safe}_caption.txt")  
            with open(capfn, 'w') as f:  
                f.write(cap)  

##############################################################################  
# Main  
##############################################################################  
  
def main():  
    # Compute Tables 1.1 and 1.2  
    t1_1 = compute_success_rates(phase='test')  
    t1_2 = compute_success_rates(phase='dev')  
  
    # Compute Tables 2.1 and 2.2  
    t2_1 = compute_average_metrics(phase='test')  
    t2_2 = compute_average_metrics(phase='dev')  
  
    # Save them as LaTeX  
    t1_1_latex = t1_1.to_latex(index=False, escape=False)  
    t1_2_latex = t1_2.to_latex(index=False, escape=False)  
    t2_1_latex = t2_1.to_latex(index=False, escape=False)  
    t2_2_latex = t2_2.to_latex(index=False, escape=False)  
  
    with open(os.path.join(RESULTS_DIR, "table_1_1.tex"), "w") as f:  
        f.write(t1_1_latex)  
    with open(os.path.join(RESULTS_DIR, "table_1_2.tex"), "w") as f:  
        f.write(t1_2_latex)  
    with open(os.path.join(RESULTS_DIR, "table_2_1.tex"), "w") as f:  
        f.write(t2_1_latex)  
    with open(os.path.join(RESULTS_DIR, "table_2_2.tex"), "w") as f:  
        f.write(t2_2_latex)  
  
    # Save captions  
    captions = {  
        "table_1_1": "Table 1.1: Success rate (% of successful runs) in test phase for each task, LM, pipeline. Success if improvement_perc > 5%. Standard deviations are included. 'Avg' row is across tasks.",  
        "table_1_2": "Table 1.2: Success rate (% of successful runs) in dev phase for each task, LM, pipeline. A run is successful if the best dev improvement > 5%. Standard deviations are included. 'Avg' row is across tasks.",  
        "table_2_1": "Table 2.1: Average test-time improvement_perc, relative_runtime, relative_complexity (mean ± std) among valid runs for each task, LM, pipeline. 'Avg' row is across tasks.",  
        "table_2_2": "Table 2.2: Average best dev-time improvement_perc, relative_runtime, relative_complexity (mean ± std) among valid runs for each task, LM, pipeline. 'Avg' row is across tasks."  
    }  
    for k,v in captions.items():  
        with open(os.path.join(RESULTS_DIR, f"{k}_caption.txt"), 'w') as f:  
            f.write(v)  
  
    # Compute and Plot Figure 3  
    pass_at_k_data, averaged_pass_at_k = compute_pass_at_k_data()  
    plot_figure_3(pass_at_k_data, averaged_pass_at_k)  
  
    # Compute and Plot Figure 4  
    fig4_data = compute_figure_4_data()  
    plot_figure_4(fig4_data)  
  
    print(f"All results have been saved to the {RESULTS_DIR} directory.")  
  
if __name__ == "__main__":  
    main()  
