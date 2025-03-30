#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
Script to parse experiment results and produce:  
  - Tables 1.x and 2.x with mean and std  
  - Figures (3, 4.1, 4.2, 4.3)  
  - Scatter plot: avg. API cost vs avg. success  
  - Correlation heatmaps: subjective vs. objective metrics  
  - A new pipeline "human+single-agent" is included in tables and figures  
  - Radar charts for table 2.1 & 2.2 (kept for reference),  
    plus a new radar chart per task that:   
      0) does not show an "Avg" chart,  
      1) uses absolute performance (performance), runtime (runtime), and complexity (method_complexity),  
      2) rescales subjective metrics from [1..5] to [20..100],  
      3) normalizes the three objective metrics to [0..100] across pipeline-LM + baseline,  
      4) adds a baseline line (dashed style) from MLAgentBench/benchmarks_base_exp/{TASK}/env/output/idea_evals.json.  
  
Outputs (figures, LaTeX tables, captions, cost report) are saved to `results/`.  
All numeric results are rounded to one decimal place for tables, but the radar charts use a scaling as described.  
  
Requires:  
  Python 3.8+  
  pandas  
  numpy  
  matplotlib  
  seaborn  
"""  
  
import os  
import glob  
import json  
import re  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker
import seaborn as sns  
from collections import defaultdict  
from math import comb  
from itertools import combinations  
import matplotlib.colors as mcolors
import colorsys

  
from MLAgentBench.constants import *
##############################################################################  
# Global config  
##############################################################################  
  
# Pipeline types  
SINGLE_AGENT = "MLAB"  
MULTI_AGENT = "CoI-Agent Idea + MLAB"  
HUMAN_SINGLE_AGENT = "Human Idea + MLAB"  
PIPELINES = [SINGLE_AGENT, MULTI_AGENT, HUMAN_SINGLE_AGENT]  
  
# LMs  
LMS = ["claude-3-5-sonnet-v2", "gemini-exp-1206", "llama3-1-405b-instruct", "o3-mini", "gpt-4o"]  
colors = ['#0173b2', '#029e73', '#cc78bc', '#ca9161', '#ece133', '#56b4e9']
LM_COLORS = {lm : c for lm, c in zip(LMS, colors)}
# Tasks 
task_name_mapping = {
        "llm-merging" : "llm-merging",
        "backdoor-trigger" : "backdoor-trigger-recovery",
        "temporal-action-loc" : "perception_temporal_action_loc",
        "machine-unlearning" : "machine_unlearning",
        "meta-learning" : "meta-learning",
        "product-rec" : "product-recommendation",
        "erasing-watermark" : "erasing_invisible_watermarks",
        "weather-forecast" : "weather_forcast",
        }
TASKS = list(task_name_mapping.keys())
for k in TASKS:
    v = task_name_mapping[k]
    task_name_mapping[v] = v

  
# Idea indices  
IDEA_IDXS = [0, 1, 2, 3]  
IDEA_PROPOSAL_MODEL = "o1-preview" 
adaptive_threshold = 0.05
  
# For human+single-agent  
HUMAN_IDEA_IDX = "rag"  
HUMAN_IDEA_PROPOSAL_MODEL = "human"  
  
  
 
# For figure line styles  
PIPELINE_LINESTYLES = {  
    SINGLE_AGENT: "solid",  
    MULTI_AGENT: "dashed",  
    HUMAN_SINGLE_AGENT: "dotted",  
}  
  
HUMAN_PERFORMANCE = {
    # only for test
    "llm-merging": {"performance" : 0.83}, 
    "backdoor-trigger": {"performance" : 67.5732}, 
    "temporal-action-loc": {"performance" : 0.4859}, 
    "machine-unlearning": {"performance" : 0.0984971060},
    "meta-learning": {"performance" : 0.699},
    "product-rec": {"performance" : 0.41208},
    "erasing-watermark": {"performance" : -0.04363443074502975},
    "weather-forecast": {"performance" : 0.1014596},
} 
all_task_improvement_perc = []
for task in HUMAN_PERFORMANCE:
    human_perf = HUMAN_PERFORMANCE[task]["performance"] 
    base_perf = ALL_BASE_PERFORMANCE[task_name_mapping[task]]["test"]
    task_improvement_perc = 100 * (human_perf - base_perf) / abs(base_perf) 
    HUMAN_PERFORMANCE[task]["improvement_perc"] = task_improvement_perc 
    all_task_improvement_perc.append(task_improvement_perc)

HUMAN_PERFORMANCE["Average"] = {"improvement_perc" : sum(all_task_improvement_perc) / len(all_task_improvement_perc)}

# Consider success if improvement_perc > 5.0  
# TASK_THRESHOLD[task] = 5.0  
# Task adaptive threshold
TASK_THRESHOLD = {task : HUMAN_PERFORMANCE[task]["improvement_perc"]*adaptive_threshold for task in TASKS}
print("HUMAN_PERFORMANCE", HUMAN_PERFORMANCE)
print(f"task success threshold: {adaptive_threshold} of human improvement", TASK_THRESHOLD)

# Results directory  
RESULTS_DIR = f"results/adaptive_threshold_{adaptive_threshold}/IDEA_PROPOSAL_MODEL_{IDEA_PROPOSAL_MODEL}"  
os.makedirs(RESULTS_DIR, exist_ok=True) 
 
##############################################################################  
# Utility functions  
##############################################################################  
  
def extract_timestamp_from_dirname(_dirname):  
    # Remove `_PID` if present
    dirname = _dirname.split('_')[0]  # Keep only the timestamp part

    pattern = r'^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$'  
    m = re.match(pattern, dirname)  
    if m:  
        ts = tuple(int(x) for x in m.groups())  
        if ts[0] >= 10: # we only use experiments conducted from January
            return None
        else:
            return ts
    return None  
  
def load_json_safely(path):  
    if not os.path.isfile(path):  
        return None  
    try:  
        with open(path, "r") as f:  
            return json.load(f)  
    except:  
        return None  
  
def find_most_recent_8_runs_for_pipeline(_task, lm, pipeline, idea_idx=None):  
    """  
    Collect run dirs from workspace and logs, unify, then keep the last 8 by ascending time.  
    """  

    log_runs = []  
    task = task_name_mapping[_task] 
    if pipeline == SINGLE_AGENT:  
        base_pattern_logs = f"logs/{task}/{lm}/*"  
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for multi-agent pipeline.")  
        base_pattern_logs = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/*"  
    elif pipeline == HUMAN_SINGLE_AGENT:  
        base_pattern_logs = f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/*"  
    else:  
        base_pattern_logs = ""  
  
    if base_pattern_logs:  
        for path in glob.glob(base_pattern_logs):  
            if os.path.isdir(path):  
                dirname = os.path.basename(path)  
                ts = extract_timestamp_from_dirname(dirname)  
                if ts is not None:  
                    log_runs.append((dirname, ts))  
  
    items = list(log_runs)  
    items.sort(key=lambda x: x[1])  # ascending  
    items = items[-8:]  
    return [x[0] for x in items]  
  
##############################################################################  
# Dev/Test result helpers  
##############################################################################  
  
def get_dev_results(_task, lm, pipeline, run_id, idea_idx=None):  
    task = task_name_mapping[_task]
    if pipeline == SINGLE_AGENT:  
        dev_file =f"logs/{task}/{lm}/{run_id}/env_log/idea_evals.json" 
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for multi-agent pipeline.")  
        dev_file = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/idea_evals.json"  
    else:  
        dev_file = f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/idea_evals.json"  
  
    data = load_json_safely(dev_file)  
    if not data:  
        return []  
    out = []  
    BASE_RUNTIME = ALL_BASE_RUNTIME[task]["dev"] 
    BASE_PERFORMANCE = ALL_BASE_PERFORMANCE[task]["dev"]

    for imp in data.get("implementations", []):  
        if imp.get("phase") == "dev" and imp["performance"] is not None: # performance should not be None  
            out.append(  
                (  
                    100 * (imp["performance"] - BASE_PERFORMANCE) / abs(BASE_PERFORMANCE), # updated with newest estimation
                    100 * (imp["runtime"] - BASE_RUNTIME) / BASE_RUNTIME,
                    imp["relative_complexity"],
                )  
            )  
    return out  
  
def get_test_result(_task, lm, pipeline, run_id, idea_idx=None):  
    task = task_name_mapping[_task]
    if pipeline == SINGLE_AGENT:  
        test_file = f"logs/{task}/{lm}/{run_id}/env_log/test_idea_evals.json"  
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for multi-agent pipeline.")  
        test_file = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/test_idea_evals.json"  
    else:  
        test_file = f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/test_idea_evals.json"  
  
    data = load_json_safely(test_file)  
    if not data:  
        return None  
    BASE_RUNTIME = ALL_BASE_RUNTIME[task]["test"] 
    BASE_PERFORMANCE = ALL_BASE_PERFORMANCE[task]["test"]

    for imp in data.get("implementations", []):  
        if imp.get("phase") == "test" and imp["performance"] is not None : # performance should not be None 
            # if task == "machine_unlearning":
            #     # substitute with best dev's runtime
            #     dev_results = get_dev_results(_task, lm, pipeline, run_id, idea_idx)
            #     dev_results.sort(key=lambda x: x[0])
            #     best_dev_result = dev_results[-1]
            #     best_dev_runtime = best_dev_result[1] 

            ret = (  
                100 * (imp["performance"] - BASE_PERFORMANCE) / abs(BASE_PERFORMANCE), # updated with newest estimation
                0 if task == "machine_unlearning" else 100 * (imp["runtime"] - BASE_RUNTIME) / BASE_RUNTIME,
                imp.get("relative_complexity", 0.0),  
            )  
            return ret
    return None
  
def load_api_cost(_task, lm, pipeline, run_id, idea_idx=None):  
    task = task_name_mapping[_task]
    if pipeline == SINGLE_AGENT:  
        cost_file = f"logs/{task}/{lm}/{run_id}/env_log/api_cost.json"  
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for multi-agent pipeline.")  
        cost_file = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/api_cost.json"  
    else:  
        cost_file = f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/api_cost.json"  
  
    data = load_json_safely(cost_file)  
    if not data:  
        return 0.0  
    return float(data.get("total_cost", 0.0))  
  
##############################################################################  
# Success rate calculations (Tables 1.1 / 1.2)  
##############################################################################  
  
def compute_success_rates_data(phase='test'):  
    """  
    Return dict[(task,pipeline,lm)] -> (mean_sr, std_sr)  
    where mean_sr, std_sr in range [0,100].  
    """  
    result = {}  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for lm in LMS:  
                success_list = []  
                if pipeline == SINGLE_AGENT:  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for rid in run_ids:  
                        if phase=='test':  
                            r = get_test_result(task, lm, pipeline, rid)  
                            success_list.append(  
                                1 if (r and r[0]>TASK_THRESHOLD[task]) else 0  
                            )  
                        else: # dev  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_imp = max(x[0] for x in dev_res)  
                                success_list.append(1 if (best_imp>TASK_THRESHOLD[task]) else 0)  
                            else:  
                                success_list.append(0)  
                elif pipeline == MULTI_AGENT:  
                    # accumulate across all ideas  
                    agg = []  
                    for idea_idx in IDEA_IDXS:  
                        run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx)  
                        for rid in run_ids:  
                            if phase=='test':  
                                r = get_test_result(task, lm, pipeline, rid, idea_idx)  
                                agg.append(  
                                    1 if (r and r[0]>TASK_THRESHOLD[task]) else 0  
                                )  
                            else:  
                                dev_res = get_dev_results(task, lm, pipeline, rid, idea_idx)  
                                if dev_res:  
                                    best_imp = max(x[0] for x in dev_res)  
                                    agg.append(1 if (best_imp>TASK_THRESHOLD[task]) else 0)  
                                else:  
                                    agg.append(0)  
                    success_list = agg  
                else: # human+single-agent  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for rid in run_ids:  
                        if phase=='test':  
                            r = get_test_result(task, lm, pipeline, rid)  
                            success_list.append(  
                                1 if (r and r[0]>TASK_THRESHOLD[task]) else 0  
                            )  
                        else:  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_imp = max(x[0] for x in dev_res)  
                                success_list.append(1 if (best_imp>TASK_THRESHOLD[task]) else 0)  
                            else:  
                                success_list.append(0)  
  
                if success_list:  
                    mean_sr = np.mean(success_list)*100  
                    std_sr  = np.std(success_list, ddof=1)*100 if len(success_list) >= 3 else 0.0
                else:  
                    mean_sr, std_sr = 0.0, 0.0  
                result[(task,pipeline,lm)] = (mean_sr, std_sr)  
    return result  

def convert_table_1(df):
    """Converts the output of construct_table_1 to the desired transposed format."""

    tasks = df['Task'].unique()
    systems = df['System'].unique()
    lms = LMS

    new_rows = []
    for task in tasks:
        if task == "Avg": # handle the avg rows
            continue
        new_row = [task]
        for system in systems:
            if system == "Avg":
                continue
            for lm in lms:
                if system != SINGLE_AGENT and lm != "gpt-4o":
                    continue

                value = df[(df['Task'] == task) & (df['System'] == system)][lm].values
                if len(value) > 0:
                    new_row.append(value[0])
                else:
                    print(task, system, lm, "no value")
                    new_row.append("")
        new_rows.append(new_row)
    
    #Handle the avg rows
    new_row = ["Avg"]
    for system in systems:
        if system == "Avg":
            continue
        for lm in lms:
            if system != SINGLE_AGENT and lm != "gpt-4o":
                continue

            value = df[(df['Task'] == "Avg") & (df['System'] == system)][lm].values
            if len(value) > 0:
                new_row.append(value[0])
            else:
                new_row.append("")
    new_rows.append(new_row)

    new_columns = ["Task"]
    for system in systems:
        if system == "Avg":
            continue
        for lm in lms:
            if system != SINGLE_AGENT and lm != "gpt-4o":
                continue
            safe_system = system.replace("\n", "\\\\")
            new_columns.append(f"\makecell{{{safe_system}\\\\{lm}}}")

    new_df = pd.DataFrame(new_rows, columns=new_columns)
    return new_df
  
def construct_table_1(success_data, phase='test'):  
    rows = []  
    pipeline_lm_task_values = defaultdict(list)  
  
    for task in TASKS:  
        for i, pipeline in enumerate(PIPELINES):  
            row_task = task 
            row = [row_task, pipeline]  
            for lm in LMS:  
                mean_sr, std_sr = success_data.get((task,pipeline,lm),(0.0,0.0))  
                row.append(f"{round(mean_sr,1)}")  
                pipeline_lm_task_values[(pipeline,lm)].append(mean_sr)  
            rows.append(row)  
  
    # "Avg" row for each pipeline  
    for pipeline in PIPELINES:  
        row = ["Avg", pipeline]  
        for lm in LMS:  
            vals = pipeline_lm_task_values[(pipeline,lm)]  
            if vals:  
                avg_sr = np.mean(vals)  
            else:  
                avg_sr = 0.0  
            row.append(f"{round(avg_sr,1)}")  
        rows.append(row)  
  
    cols = ["Task","System"]+LMS  
    df = pd.DataFrame(rows, columns=cols)  
    return convert_table_1(df)  
  
##############################################################################  
# Table 2.x average metrics (still uses improvement & relative values)  
##############################################################################  
  
def compute_test_llm_eval_metrics(_task, lm, pipeline, run_id, idea_idx=None):  
    """  
    Return the "with_code" portion of llm_eval if any.  
    """  
    task = task_name_mapping[_task]
    if pipeline == SINGLE_AGENT:  
        test_file = f"logs/{task}/{lm}/{run_id}/env_log/test_idea_evals.json"  
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            return {}  
        test_file = f"logs/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/test_idea_evals.json"  
    else:  
        test_file = f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{run_id}/env_log/test_idea_evals.json"  
  
    data = load_json_safely(test_file)  
    if not data:  
        return {}  
    for imp in data.get("implementations", []):  
        if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None  
            llm_eval = imp.get("llm_eval", {})  
            return llm_eval.get("with_code", {})  
    return {}  
  
def compute_average_metrics_data(phase='test', include_llm_eval=False):  
    """  
    returns dict[(task,pipeline,lm)] -> {  
      'imp_mean', 'imp_std',  
      'run_mean','run_std',  
      'comp_mean','comp_std',  
      'clarity_mean','clarity_std', ...  
    }  
    """  
    result = {}  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for lm in LMS:  
                imp_vals, run_vals, comp_vals = [], [], []  
                clarity_vals, validity_vals = [], []  
                rigor_vals, innov_vals, gener_vals = [],[],[]  
                if pipeline==SINGLE_AGENT:  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for rid in run_ids:  
                        if phase=='test':  
                            rr = get_test_result(task, lm, pipeline, rid)  
                            if rr:  
                                imp_vals.append(rr[0])  
                                run_vals.append(rr[1])  
                                comp_vals.append(rr[2])  
                                if include_llm_eval:  
                                    wc = compute_test_llm_eval_metrics(task, lm, pipeline, rid)  
                                    for f, store in [  
                                        ("Clarity", clarity_vals),  
                                        ("Validity", validity_vals),  
                                        ("Rigorousness", rigor_vals),  
                                        ("Innovativeness", innov_vals),  
                                        ("Generalizability", gener_vals),  
                                    ]:  
                                        rating = wc.get(f, {}).get("Rating", None)  
                                        if rating is not None:  
                                            store.append(rating)  
                        else: # dev  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_idx = np.argmax([x[0] for x in dev_res])  
                                best = dev_res[best_idx]  
                                imp_vals.append(best[0])  
                                run_vals.append(best[1])  
                                comp_vals.append(best[2])  
                elif pipeline==MULTI_AGENT:  
                    for idea_idx in IDEA_IDXS:  
                        run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx)  
                        for rid in run_ids:  
                            if phase=='test':  
                                rr = get_test_result(task, lm, pipeline, rid, idea_idx)  
                                if rr:  
                                    imp_vals.append(rr[0])  
                                    run_vals.append(rr[1])  
                                    comp_vals.append(rr[2])  
                                    if include_llm_eval:  
                                        wc = compute_test_llm_eval_metrics(task, lm, pipeline, rid, idea_idx)  
                                        for f, store in [  
                                            ("Clarity", clarity_vals),  
                                            ("Validity", validity_vals),  
                                            ("Rigorousness", rigor_vals),  
                                            ("Innovativeness", innov_vals),  
                                            ("Generalizability", gener_vals),  
                                        ]:  
                                            rating = wc.get(f, {}).get("Rating", None)  
                                            if rating is not None:  
                                                store.append(rating)  
                            else:  
                                dev_res = get_dev_results(task, lm, pipeline, rid, idea_idx)  
                                if dev_res:  
                                    best_idx = np.argmax([x[0] for x in dev_res])  
                                    best = dev_res[best_idx]  
                                    imp_vals.append(best[0])  
                                    run_vals.append(best[1])  
                                    comp_vals.append(best[2])  
                else:  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for rid in run_ids:  
                        if phase=='test':  
                            rr = get_test_result(task, lm, pipeline, rid)  
                            if rr:  
                                imp_vals.append(rr[0])  
                                run_vals.append(rr[1])  
                                comp_vals.append(rr[2])  
                                if include_llm_eval:  
                                    wc = compute_test_llm_eval_metrics(task, lm, pipeline, rid)  
                                    for f, store in [  
                                        ("Clarity", clarity_vals),  
                                        ("Validity", validity_vals),  
                                        ("Rigorousness", rigor_vals),  
                                        ("Innovativeness", innov_vals),  
                                        ("Generalizability", gener_vals),  
                                    ]:  
                                        rating = wc.get(f, {}).get("Rating", None)  
                                        if rating is not None:  
                                            store.append(rating)  
                        else:  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_idx = np.argmax([x[0] for x in dev_res])  
                                best = dev_res[best_idx]  
                                imp_vals.append(best[0])  
                                run_vals.append(best[1])  
                                comp_vals.append(best[2])  
  
                def mean_std(arr):  
                    if not arr:  
                        return (0.0, 0.0)  
                    if len(arr) < 3:
                        return (np.mean(arr), 0.0)
                    return (np.mean(arr), np.std(arr, ddof=1))  
  
                imp_m, imp_s = mean_std(imp_vals)  
                run_m, run_s = mean_std(run_vals)  
                comp_m,comp_s= mean_std(comp_vals)  
                c_m,c_s      = mean_std(clarity_vals)  
                v_m,v_s      = mean_std(validity_vals)  
                r_m,r_s      = mean_std(rigor_vals)  
                i_m,i_s      = mean_std(innov_vals)  
                g_m,g_s      = mean_std(gener_vals)  
  
                result[(task,pipeline,lm)] = {  
                    'imp_mean':imp_m,'imp_std':imp_s,  
                    'run_mean':run_m,'run_std':run_s,  
                    'comp_mean':comp_m,'comp_std':comp_s,  
                    'clarity_mean':c_m,'clarity_std':c_s,  
                    'validity_mean':v_m,'validity_std':v_s,  
                    'rigorous_mean':r_m,'rigorous_std':r_s,  
                    'innov_mean':i_m,'innov_std':i_s,  
                    'gener_mean':g_m,'gener_std':g_s,  
                }  
    return result  

def convert_table_2(df):
    """Converts the output of build_table_2 to the desired transposed format."""

    tasks = df['Task'].unique()
    metrics = df['Metric'].unique()
    systems = df['System'].unique()
    lms = LMS  # Use the global LMS

    new_rows = []
    for task in tasks:
        for metric in metrics:
            new_row = [task, metric]
            for system in systems:
                for lm in lms:
                    if system != SINGLE_AGENT and lm != "gpt-4o":
                        continue
                    value = df[(df['Task'] == task) & (df['Metric'] == metric) & (df['System'] == system)][lm].values
                    if len(value) > 0:
                        new_row.append(value[0])
                    else:
                        assert 0
                        new_row.append("")  # Handle cases where data is missing
            new_rows.append(new_row)

    # Construct the columns for the new DataFrame
    new_columns = ["Task", "Metric"]
    for system in systems:
        for lm in lms:
            if system != SINGLE_AGENT and lm != "gpt-4o":
                continue

            safe_system = system.replace("\n", "\\\\")
            new_columns.append(f"\makecell{{{safe_system}\\\\{lm}}}")

    new_df = pd.DataFrame(new_rows, columns=new_columns)
    return new_df
 
def build_table_2(average_data, phase='test', include_llm_eval=False):  
    """  
    Builds a longer-style table with columns:  
      Task | System | Metric | ...one column per LM...  
    so that each metric (e.g. Imp, Run, Comp) appears in its own row.  
    """  
    # We'll define which metrics to include:  
    base_metrics = [  
        ("Imp",  "imp_mean",  "imp_std"),  
        ("Run",  "run_mean",  "run_std"),  
        ("Comp", "comp_mean", "comp_std"),  
    ]  
    llm_eval_metrics = [  
        ("Clarity",        "clarity_mean",   "clarity_std"),  
        ("Validity",       "validity_mean",  "validity_std"),  
        ("Rigorousness",   "rigorous_mean",  "rigorous_std"),  
        ("Innovativeness", "innov_mean",     "innov_std"),  
        ("Generalizability","gener_mean",    "gener_std"),  
    ]  
    # If including LLM eval metrics, add them:  
    metrics = base_metrics[:]  
    if include_llm_eval and phase == 'test':  
        metrics += llm_eval_metrics  
  
    # Build the columns: "Task", "System", "Metric" plus each LM as a column  
    columns = ["Task", "System", "Metric"] + LMS  
  
    rows = []  
    # Go through each task & pipeline & gather the metrics as separate rows  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for (metric_label, mean_key, std_key) in metrics:  
                # Prepare the row up to the LM values  
                row = [task, pipeline, metric_label]  
                # For each LM, pick up its mean±std for this metric  
                for lm in LMS:  
                    d = average_data.get((task, pipeline, lm), {})  
                    m = d.get(mean_key, 0.0)  
                    s = d.get(std_key, 0.0)  
                    row.append(f"{round(m,1)}±{round(s,1)}")  
                rows.append(row)  
  
    df = pd.DataFrame(rows, columns=columns)  
    return convert_table_2(df) 
  
##############################################################################  
# Figure 3: pass@k  
##############################################################################  

# change to test TODO
def get_test_improvement_success(task, lm, pipeline, run_id, threshold=5.0, idea_idx=None):  
    """  
    Return True if test results exist with improvement>threshold.  
    """  
    test_res = []  
    if pipeline==SINGLE_AGENT:  
        test_res = get_test_result(task, lm, pipeline, run_id)  
    elif pipeline==MULTI_AGENT and idea_idx is not None:  
        test_res = get_test_result(task, lm, pipeline, run_id, idea_idx)  
    else:  
        test_res = get_test_result(task, lm, pipeline, run_id)  
    return test_res and test_res[0]>threshold  
  
def compute_pass_at_k_data():  
    pass_at_k_data = {lm: defaultdict(dict) for lm in LMS}  
    m=8 # TODO: avoid hardcode 8 
    kvals = range(1,m+1)  

    def get_pass_at_k(kvals, c_impl):
        # pass@k  
        arr_impl=[]  
        for k in kvals:  
            if k>m:  
                pass_k = 1.0 if c_impl>0 else 0.0  
            else:  
                denom = comb(m,k)  
                num = comb(m-c_impl,k)  
                pass_k = 1.0 - num/denom if denom>0 else 0.0  
            arr_impl.append(pass_k)   
        return arr_impl

    for lm in LMS:  
        for task in TASKS:  
            # single-agent  
            for N, pipeline in [(0, SINGLE_AGENT), (-1, HUMAN_SINGLE_AGENT)]:
                runs_sa = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                sa_successes = 0
                for rid in runs_sa:
                    if get_test_improvement_success(task, lm, pipeline, rid, threshold=TASK_THRESHOLD[task]):  
                        sa_successes+=1  
                c_impl = sa_successes  
                pass_at_k_data[lm][task][N] = get_pass_at_k(kvals, c_impl) 

            # multi-agent
            # total_success_over_all_ideas = 0
            # for idea_idx in IDEA_IDXS:  
            #     runs_ = find_most_recent_8_runs_for_pipeline(task, lm, MULTI_AGENT, idea_idx)  
            #     for rid in runs_:  
            #         if get_test_improvement_success(task, lm, MULTI_AGENT, rid, threshold=TASK_THRESHOLD[task], idea_idx=idea_idx):  
            #             total_success_over_all_ideas+=1  
            # pass_at_k_data[lm][task][1] = get_pass_at_k(range(1,m*len(IDEA_IDXS)+1), total_success_over_all_ideas)
 
            # multi-agent  
            c_list=[]  
            for idea_idx in IDEA_IDXS:  
                runs_ = find_most_recent_8_runs_for_pipeline(task, lm, MULTI_AGENT, idea_idx)  
                c_ = 0  
                for rid in runs_:  
                    if get_test_improvement_success(task, lm, MULTI_AGENT, rid, threshold=TASK_THRESHOLD[task], idea_idx=idea_idx):  
                        c_+=1  
                c_list.append(c_)  
            for N in [1,2,4]:  
                arrN=[]  
                subsets = list(combinations(range(len(c_list)),N))  
                for k in kvals:  
                    numerator=0  
                    denominator = comb(4,N)*(comb(m,k)**N)  
                    for sub in subsets:  
                        product=1  
                        for idx in sub:  
                            c_i = c_list[idx]  
                            if m-c_i<k:  
                                product=0  
                                break  
                            else:  
                                product*=comb(m-c_i,k)  
                        numerator+=product  
                    pass_k=1.0-(numerator/denominator) if denominator>0 else 0.0  
                    arrN.append(pass_k)  
                pass_at_k_data[lm][task][N] = arrN  
  
    # average  
    averaged = {lm:{} for lm in LMS}  
    for lm in LMS:  
        for N in [-1,0,1,2,4]:  
            sums=[0.0]*len(pass_at_k_data[lm][TASKS[0]][N]) 
            for task in TASKS:  
                y_ = pass_at_k_data[lm][task][N] 
                sums=[a+b for a,b in zip(sums,y_)]  
            avg=[x/len(TASKS) for x in sums]  
            averaged[lm][N]=avg  
    return pass_at_k_data, averaged  
  
def plot_figure_3(pass_at_k_data, averaged_pass_at_k):  
    plt.rcParams['font.size'] = 10
    for lm in LMS:  
        if lm != "gpt-4o":
            continue
        for task in TASKS+["Average"]:  
            plt.figure(figsize=(6,4))  
            for N in [0,-1,1,2,4]:  
                if task=="Average":  
                    y=averaged_pass_at_k[lm][N]  
                    t_="Average Over All Tasks"  
                else:  
                    y=pass_at_k_data[lm][task][N]  
                    t_=task  
                if N==0:  
                    label=f"{SINGLE_AGENT}"  
                elif N==-1:
                    label=f"{HUMAN_SINGLE_AGENT}"
                else:  
                    label=f"{MULTI_AGENT}\n# Ideas = {N}"  
                xvals=range(1,1+len(y))  
                plt.plot(xvals, y, marker='o', label=label)  
            plt.title(f"{lm}, {t_}")  
            plt.xlabel("Number of Trials (k)")  
            plt.ylabel("pass@k")  
            # plt.ylim([0,1.05])  
            plt.xticks(list(xvals))  
            plt.grid(True)  
            plt.legend()  
            if task=="Average":  
                outfn = os.path.join(RESULTS_DIR,f"figure_3_{lm}_average.pdf")  
            else:  
                outfn = os.path.join(RESULTS_DIR,f"figure_3_{lm}_{task.replace(' ','_')}.pdf")  
            plt.savefig(outfn,bbox_inches='tight')  
            plt.close()  
            caption=(  
                f"Figure 3 for LM={lm}, Task={t_}. pass@k on dev set vs # of trials. "  
                "N=0 => single-agent, N=1,2,4 => multi-agent. Probability of at least one success (>5% improvement)."  
            )  
            if task=="Average":  
                capfn = os.path.join(RESULTS_DIR,f"figure_3_{lm}_average_caption.txt")  
            else:  
                capfn = os.path.join(RESULTS_DIR,f"figure_3_{lm}_{task.replace(' ','_')}_caption.txt")  
            # with open(capfn,"w") as f:  
            #     f.write(caption)  
  
##############################################################################  
# Figure 4: dev improvement vs i-th implementation  
##############################################################################  
  
def compute_figure_4_data():  
    def expand_and_avg(list_of_arrays):  
        if not list_of_arrays:  
            return []  
        max_len = max(len(a) for a in list_of_arrays)  
        expanded=[]  
        for arr in list_of_arrays:  
            if len(arr)<max_len:  
                arr=arr+[arr[-1]]*(max_len-len(arr))  
            expanded.append(arr)  
        return np.mean(expanded,axis=0).tolist()  
  
    fig4_data=defaultdict(dict)  
    for task in TASKS+["Average"]:  
        for lm in LMS:  
            for pipeline in PIPELINES:  
                all_imps, all_runs, all_comps=[],[],[]  
                tasks_ = TASKS if task=="Average" else [task]  
                for t_ in tasks_:  
                    if pipeline==SINGLE_AGENT:  
                        rids = find_most_recent_8_runs_for_pipeline(t_, lm, pipeline)  
                        for rid in rids:  
                            dev_res = get_dev_results(t_, lm, pipeline, rid)  
                            if dev_res:  
                                im=[x[0] for x in dev_res]  
                                ru=[x[1] for x in dev_res]  
                                co=[x[2] for x in dev_res]  
                                all_imps.append(im)  
                                all_runs.append(ru)  
                                all_comps.append(co)  
                    elif pipeline==MULTI_AGENT:  
                        for idx in IDEA_IDXS:  
                            rids = find_most_recent_8_runs_for_pipeline(t_, lm, pipeline, idx)  
                            for rid in rids:  
                                dev_res = get_dev_results(t_, lm, pipeline, rid, idx)  
                                if dev_res:  
                                    im=[x[0] for x in dev_res]  
                                    ru=[x[1] for x in dev_res]  
                                    co=[x[2] for x in dev_res]  
                                    all_imps.append(im)  
                                    all_runs.append(ru)  
                                    all_comps.append(co)  
                    else:  
                        rids = find_most_recent_8_runs_for_pipeline(t_, lm, pipeline)  
                        for rid in rids:  
                            dev_res = get_dev_results(t_, lm, pipeline, rid)  
                            if dev_res:  
                                im=[x[0] for x in dev_res]  
                                ru=[x[1] for x in dev_res]  
                                co=[x[2] for x in dev_res]  
                                all_imps.append(im)  
                                all_runs.append(ru)  
                                all_comps.append(co)  
                imp_means=expand_and_avg(all_imps)  
                run_means=expand_and_avg(all_runs)  
                comp_means=expand_and_avg(all_comps)  
                fig4_data[task][(lm,pipeline)] = {  
                    "improvement_perc": imp_means,  
                    "relative_runtime": run_means,  
                    "relative_complexity": comp_means,  
                }  
    return fig4_data  

def plot_figure_4(fig4_data):
    plt.rcParams['font.size'] = 18
    metrics = ["improvement_perc", "relative_runtime", "relative_complexity"]
    titles = ["Performance Improvement (%, \u2191)", "Increased Runtime (%, \u2193)", "Increased Lines of Code (%, \u2193)"]
    nums = [1, 2, 3]

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    for task in TASKS + ["Average"]:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))  # Create figure and subplots, adjust figsize as needed
        fig.suptitle(f"{task}" if task != "Average" else "Average over all tasks", fontsize=20, y=1.0) # Lower suptitle closer to plots

        handles, labels_for_legend = [], [] # To collect handles and labels for the centralized legend

        for i, met in enumerate(metrics):
            ax = axes[i] # Current subplot axis
            metric_handles = [] # Handles for this metric subplot
            metric_labels = [] # Labels for this metric subplot

            for lm in LMS:
                for pipeline in PIPELINES:
                    if pipeline != SINGLE_AGENT:
                        continue
                    arr = fig4_data[task][(lm, pipeline)][met]
                    if not arr:
                        continue
                    xvals = range(1, len(arr) + 1)
                    style_ = PIPELINE_LINESTYLES.get(pipeline, 'solid')
                    color_ = LM_COLORS.get(lm, 'black')
                    lab = f"{lm}"
                    line, = ax.plot(xvals, arr, marker='o', linestyle=style_, color=color_, label=lab) # Get line object

                    metric_handles.append(line) # Collect handles for legend
                    metric_labels.append(lab) # Collect labels for legend


            ax.set_title(titles[i]) # Set subplot title from titles list
            ax.set_xlabel("i-th implementation in a trial")
            # ax.set_ylabel(titles[i]) # Removed y-axis label in subplots
            ax.grid(True)
            ax.yaxis.label.set_visible(False) # alternative way to hide y label

            # Set x-axis ticks to integers only
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


        # Centralized legend below the subplots
        # Get unique handles and labels to avoid duplicates in legend
        unique_labels = []
        unique_handles = []
        label_set = set()
        for h, l in zip(metric_handles, metric_labels):
            if l not in label_set:
                unique_handles.append(h)
                unique_labels.append(l)
                label_set.add(l)


        fig.legend(unique_handles, unique_labels, loc='lower center', ncol=len(unique_labels), bbox_to_anchor=(0.5, -0.05)) # Adjust bbox_to_anchor and ncol for best position

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for suptitle and legend
        # Manually adjust subplot params to move subplots up if necessary
        plt.subplots_adjust(top=0.88) # Adjust top to move subplots down relative to suptitle

        outfn = os.path.join(RESULTS_DIR, f"figure_4_all_{task.replace(' ', '_')}.pdf") # Save single figure
        fig.savefig(outfn, bbox_inches='tight')
        plt.close(fig) # Close the figure

        cap = (
            f"Figure 4 for {task}. "
            f"Plots improvement_perc, relative_runtime, and relative_complexity (from left to right) vs implementation index. "
            f"Different colors represent different LMs."
        )
        capfn = os.path.join(RESULTS_DIR, f"figure_4_all_{task.replace(' ', '_')}_caption.txt")
        # with open(capfn, "w") as f:
        #     f.write(cap)
 
 
##############################################################################  
# Scatter: cost vs success  
##############################################################################  

def load_idea_cost(_task,lm):
    task = task_name_mapping[_task]
    idea_costs = []
    for i in IDEA_IDXS:
        coi_idea_file = f"../CoI-Agent/results/{task}/{lm}/{i}/result.json"
        with open(coi_idea_file, 'r') as reader:
            items = json.load(reader)
        idea_costs.append(items["api_cost"])
    return sum(idea_costs) / len(idea_costs)

def compute_api_cost_and_success_for_scatter():  
    data_points=[]  
    for pipeline in PIPELINES:  
        for lm in LMS:  
            if lm == "gemini-exp-1206":
                continue
            costs=[]  
            successes=[]  
            for task in TASKS:  
                if pipeline==MULTI_AGENT:  
                    if lm != "gpt-4o":
                        continue
                    for idx in IDEA_IDXS:  
                        rids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idx)  
                        for rid in rids:  
                            c_=load_api_cost(task,lm,pipeline,rid,idx)  
                            idea_cost=load_idea_cost(task,IDEA_PROPOSAL_MODEL)
                            costs.append(c_+idea_cost)  
                            res=get_test_result(task,lm,pipeline,rid,idx)  
                            s=1 if (res and res[0]>TASK_THRESHOLD[task]) else 0  
                            successes.append(s)  
                elif pipeline==SINGLE_AGENT:  
                    rids = find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        c_=load_api_cost(task,lm,pipeline,rid)  
                        costs.append(c_)  
                        res=get_test_result(task,lm,pipeline,rid)  
                        s=1 if(res and res[0]>TASK_THRESHOLD[task]) else 0  
                        successes.append(s)  
                else:  
                    if lm != "gpt-4o":
                        continue
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        c_=load_api_cost(task,lm,pipeline,rid)  
                        costs.append(c_)  
                        res=get_test_result(task,lm,pipeline,rid)  
                        s=1 if(res and res[0]>TASK_THRESHOLD[task]) else 0  
                        successes.append(s)  
            # print(lm)
            # print(costs)
            # print(successes)
            if costs and successes:
                avc = np.mean(costs) 
                avs = np.mean(successes)*100 
                data_points.append({  
                    "lm":lm,  
                    "pipeline":pipeline,  
                    "avg_cost":avc,  
                    "avg_sr":avs,  
                })  
    return data_points  
  
def compute_pareto_front(data_points):  
    sdata=sorted(data_points,key=lambda d:d["avg_cost"])  
    front=[]  
    max_sr=-1  
    for p in sdata:  
        if p["avg_sr"]>max_sr:  
            front.append(p)  
            max_sr=p["avg_sr"]  
    return front  
  
def plot_scatter_api_cost_vs_success():  
    plt.rcParams['font.size'] = 10
    from adjustText import adjust_text
    data=compute_api_cost_and_success_for_scatter()  
    shape_map={SINGLE_AGENT:'o',MULTI_AGENT:'s',HUMAN_SINGLE_AGENT:'^'}  
    plt.figure(figsize=(7,5))  
    # x_offset = {
    #         "claude-3-5-sonnet-v2": -1,
    #         "llama3-1-405b-instruct": 1,
    #         "o1-mini": -0.5, 
    #         "gpt-4o" : 0,
    #         }
    # y_offset = {
    #         "claude-3-5-sonnet-v2": 0.5,
    #         "llama3-1-405b-instruct": -2,
    #         "o1-mini": 1, 
    #         "gpt-4o" : 0,
    #         }

    texts = []
    for dp in data:  
        x=dp["avg_cost"]  
        y=dp["avg_sr"]  
        lm=dp["lm"]  
        pipeline=dp["pipeline"]  
        sh=shape_map.get(pipeline,'o')  
        col=LM_COLORS.get(lm,'black')  
        plt.scatter(x,y,marker=sh,color=col,s=100,edgecolors='black')  
        safe_pipeline = pipeline.replace('w/ ', 'w/\n')
        if lm == "claude-3-5-sonnet-v2":
            text = plt.text(x-0.5,y-0.5,f"{lm}\n({safe_pipeline})", ha='left', multialignment='left')  
        else:
            text = plt.text(x,y,f"{lm}\n({safe_pipeline})", ha='left', multialignment='left')  
        texts.append(text)

    pf=compute_pareto_front(data)  
    pf_x=[p["avg_cost"] for p in pf]  
    pf_y=[p["avg_sr"] for p in pf]  
    plt.plot(pf_x,pf_y,color='black',linestyle='--',label='Pareto Front')  

    # Enhanced adjust_text parameters
    adjust_text(texts, autoalign='y',
                expand_points=(1.3, 1.3),  # Further expand points
                # arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
                force_text=(0.2, 0.5), force_points=(0.3, 0.7),
                force_pull=0.9, # new parameter
                avoid_points=False, #new parameter
                )

    plt.xlabel("Average API Cost Per Trial (USD)")  
    plt.ylabel("Average Success Rate (%)")  
    # plt.title("Scatter: API Cost vs Success Rate (Test)")  
    plt.grid(True)  
    plt.legend()  
    outfn=os.path.join(RESULTS_DIR,"scatter_api_cost_vs_success.pdf")  
    plt.savefig(outfn,bbox_inches='tight')  
    plt.close()  
    cap=(  
        "Scatter plot of average API cost (x-axis) vs average success rate (y-axis). "  
        "Shapes => pipeline, colors => LM, dashed green line => Pareto front."  
    )  
    # with open(os.path.join(RESULTS_DIR,"scatter_api_cost_vs_success_caption.txt"),"w") as f:  
    #     f.write(cap)  
  
##############################################################################  
# Correlation heatmaps  
##############################################################################  
  
def gather_test_correlation_data(tasks_to_do=TASKS):  
    records_with=[]  
    records_without=[]  
    subj_fields=["Clarity","Validity","Rigorousness","Innovativeness","Generalizability"]  
    obj_fields=["improvement_perc","relative_runtime","relative_complexity"]  

    # recalculate improvement_perc?
  
    def process_imp(imp):  
        BASE_RUNTIME = ALL_BASE_RUNTIME[imp["task_name"]][imp["phase"]] 
        BASE_PERFORMANCE = ALL_BASE_PERFORMANCE[imp["task_name"]][imp["phase"]]

        base={  
            "improvement_perc": 100 * (imp["performance"] - BASE_PERFORMANCE) / abs(BASE_PERFORMANCE),  
            # None values are automatically ignored when calculating correlation
            "relative_runtime": -100 * (imp["runtime"] - BASE_RUNTIME) / BASE_RUNTIME if BASE_RUNTIME else None,  # higher the better
            "relative_complexity":-imp.get("relative_complexity",0.0),  # higher the better
        }  
        llm_eval=imp.get("llm_eval",{})  
        wc=llm_eval.get("with_code",{})  
        if wc:  
            row_w=base.copy()  
            for sf in subj_fields:  
                r=wc.get(sf,{}).get("Rating",None)  
                if r is not None:  
                    row_w[sf]=r  
            records_with.append(row_w)  
        woc=llm_eval.get("without_code",{})  
        if woc:  
            row_wo=base.copy()  
            for sf in subj_fields:  
                r=woc.get(sf,{}).get("Rating",None)  
                if r is not None:  
                    row_wo[sf]=r  
            records_without.append(row_wo)  
  
    for a_task in tasks_to_do:  
        task = task_name_mapping[a_task]
        for pipeline in PIPELINES:  
            if pipeline==MULTI_AGENT:  
                for idx in IDEA_IDXS:  
                    for lm in LMS:  
                        rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline,idx)  
                        for rid in rids:  
                            test_file=f"logs/{task}--{idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{rid}/env_log/test_idea_evals.json"  
                            dat=load_json_safely(test_file)  
                            if dat and "implementations" in dat:  
                                for imp in dat["implementations"]:  
                                    if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None  
                                        process_imp(imp)  
            elif pipeline==SINGLE_AGENT:  
                for lm in LMS:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        test_file=f"logs/{task}/{lm}/{rid}/env_log/test_idea_evals.json"  
                        dat=load_json_safely(test_file)  
                        if dat and "implementations" in dat:  
                            for imp in dat["implementations"]:  
                                if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None
                                    process_imp(imp)  
            else:  
                for lm in LMS:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        test_file=f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{rid}/env_log/test_idea_evals.json"  
                        dat=load_json_safely(test_file)  
                        if dat and "implementations" in dat:  
                            for imp in dat["implementations"]:  
                                if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None
                                    process_imp(imp)  
  
    df_with=pd.DataFrame(records_with)  
    df_without=pd.DataFrame(records_without)  
    return df_with, df_without  
  
def plot_correlation_heatmaps():
    plt.rcParams['font.size'] = 10
    # for a_task in TASKS:
    dfw, dfwo = gather_test_correlation_data(tasks_to_do=TASKS)
    subj = ["Clarity", "Validity", "Rigorousness", "Innovativeness", "Generalizability"]
    obj = ["improvement_perc", "relative_runtime", "relative_complexity"]
    obj_label_map = {
        "improvement_perc": "Effectiveness",
        "relative_runtime": "Efficiency",
        "relative_complexity": "Simplicity"
    }

    for nm, df_ in [("with_code", dfw), ("without_code", dfwo)]:
        allf = subj + obj
        ex = [c for c in allf if c in df_.columns]
        if len(df_) < 2 or len(ex) < 2:
            continue
        corr_ = df_[ex].corr(method='spearman')
        rAvail = [f for f in subj if f in corr_.index]
        cAvail = [f for f in obj if f in corr_.columns]
        if not rAvail or not cAvail:
            continue
        subcorr = corr_.loc[rAvail, cAvail]
        plt.figure(figsize=(4, 3))
        # print(nm, subcorr)
        ax = sns.heatmap(
            subcorr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=[obj_label_map.get(x, x) for x in subcorr.columns], 
            cbar_kws={'label': 'Correlation'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Horizontal labels
        nm2title = {"with_code" : "w/ code", "without_code" : "w/o code"}
        plt.title(f"{nm2title[nm]}")
        outfn = os.path.join(RESULTS_DIR, f"correlation_heatmap_{nm}.pdf")
        plt.savefig(outfn, bbox_inches='tight')
        plt.close()
        cap = (
            f"Correlation heatmap ({nm}) between LLM-as-a-Judge subjective metrics (rows) "
            f"and ML objective metrics (cols) for valid test implementations."
        )
        # with open(os.path.join(RESULTS_DIR, f"correlation_heatmap_{nm}_caption.txt"), "w") as f:
        #     f.write(cap)

  
##############################################################################  
# (Old) Radar chart for table 2.1 & 2.2  
##############################################################################  
  
def plot_radar_chart_for_table_2(df_21, df_22):  
    """  
    Kept for reference from original instructions.  
    """  
  
    def radar_prep(df):  
        cats = df.columns[2:]  
        rows=[]  
        for i,row_ in df.iterrows():  
            label = f"{row_[0]}-{row_[1]}"  
            vals=[]  
            for c in cats:  
                s=str(row_[c])  
                if "±" in s:  
                    s2=s.split("±")[0]  
                    try:  
                        v=float(s2)  
                    except:  
                        v=0.0  
                else:  
                    try:  
                        v=float(s)  
                    except:  
                        v=0.0  
                vals.append(v)  
            rows.append((label,vals))  
        return cats, rows  
  
    def plot_radar_item(ax, cats, vals, label):  
        N=len(cats)  
        angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()  
        angles+=[angles[0]]  
        vals+=[vals[0]]  
        ax.plot(angles, vals, linewidth=1, linestyle='solid', label=label)  
        ax.fill(angles, vals, alpha=0.1)  
  
    # Table 2.1  
    if not df_21.empty:  
        cats, data_ = radar_prep(df_21)  
        N=len(cats)  
        angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist()  
        fig=plt.figure(figsize=(6,6))  
        ax=plt.subplot(111, polar=True)  
        ax.set_theta_offset(np.pi/2)  
        ax.set_theta_direction(-1)  
        ax.set_thetagrids([a*180/np.pi for a in angles],cats)  
        for (lab, vals) in data_:  
            plot_radar_item(ax, cats.tolist(), vals, lab)  
        plt.title("Radar Chart: Table 2.1 (Test + LLM metrics)")  
        plt.legend(bbox_to_anchor=(1.2,1.05))  
        outfn=os.path.join(RESULTS_DIR,"radar_table_2_1.pdf")  
        plt.savefig(outfn,bbox_inches='tight')  
        plt.close()  
        # with open(os.path.join(RESULTS_DIR,"radar_table_2_1_caption.txt"),"w") as f:  
        #     f.write("Radar chart for Table 2.1 (test) with objective+subjective metrics.\n")  
  
    # Table 2.2  
    if not df_22.empty:  
        cats, data_ = radar_prep(df_22)  
        N=len(cats)  
        angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()  
        fig=plt.figure(figsize=(6,6))  
        ax=plt.subplot(111, polar=True)  
        ax.set_theta_offset(np.pi/2)  
        ax.set_theta_direction(-1)  
        ax.set_thetagrids([a*180/np.pi for a in angles], cats)  
        for (lab, vals) in data_:  
            plot_radar_item(ax, cats.tolist(), vals, lab)  
        plt.title("Radar Chart: Table 2.2 (Dev)")  
        plt.legend(bbox_to_anchor=(1.2,1.05))  
        outfn=os.path.join(RESULTS_DIR,"radar_table_2_2.pdf")  
        plt.savefig(outfn,bbox_inches='tight')  
        plt.close()  
        # with open(os.path.join(RESULTS_DIR,"radar_table_2_2_caption.txt"),"w") as f:  
        #     f.write("Radar chart for Table 2.2 (dev) with objective metrics only.\n")  
  
##############################################################################  
# NEW Radar chart with absolute performance + baseline  
##############################################################################  
  
def load_baseline_data_for_task(_task):  
    """  
    Read from MLAgentBench/benchmarks_base_exp/{task}/env/output/idea_evals.json  
    Return (perf, runtime, complexity) as the baseline (test).  
    If there's multiple test items, average them. If none, return None.  
    """  
    task = task_name_mapping[_task]
    path = f"MLAgentBench/benchmarks_base_exp/{task}/env/output/idea_evals.json"  
    data = load_json_safely(path)  
    if not data:  
        return None  
    # gather all with phase=="test"  
    perf_vals = []  
    run_vals = []  
    comp_vals = []  
    clar_list=[]  
    val_list=[]  
    rig_list=[]  
    inn_list=[]  
    gen_list=[]  
    BASE_RUNTIME = ALL_BASE_RUNTIME[task]["test"] if task != "machine_unlearning" else 0 
    BASE_PERFORMANCE = ALL_BASE_PERFORMANCE[task]["test"]


    for imp in data.get("implementations", []):  
        if imp.get("phase")=="test":  
            perf_vals.append(imp.get("performance", 0.0))  
            run_vals.append(imp.get("runtime",0.0))  
            comp_vals.append(imp.get("method_complexity",0.0))  
            llm_eval=imp.get("llm_eval",{})  
            if llm_eval:
                wc=llm_eval.get("with_code",{})  
                for subf, arr in [  
                    ("Clarity", clar_list),  
                    ("Validity", val_list),  
                    ("Rigorousness", rig_list),  
                    ("Innovativeness", inn_list),  
                    ("Generalizability", gen_list),  
                ]:  
                    rating=wc.get(subf,{}).get("Rating",0)  
                    # scale rating 1..5 => 20..100  
                    arr.append(rating if rating>0 else 0)  

    if not perf_vals:  
        return None  
    return (  
        BASE_PERFORMANCE, # np.mean(perf_vals),  
        BASE_RUNTIME, # np.mean(run_vals),  
        comp_vals[0],  # assume LLOC of each test is same
        np.mean(clar_list),
        np.mean(val_list),
        np.mean(rig_list),
        np.mean(inn_list),
        np.mean(gen_list)
    )  

def get_best_dev_runtime(_task, lm, pipeline, run_id, idea_idx=None):
    dev_results = get_dev_results(_task, lm, pipeline, run_id, idea_idx)
    dev_results.sort(key=lambda x: x[0])
    best_dev_result = dev_results[-1]
    best_dev_runtime = best_dev_result[1] 
    return best_dev_runtime

def gather_test_absolute_data_for_task(_task):  
    """  
    For each pipeline, LM, gather the average (performance, runtime, method_complexity, clarity, validity, rigorousness, innov, gener).  
    We'll parse logs from up to 8 runs. Each run has test_idea_evals.json with a single 'test' item if valid.  
    We'll average them. Return a dict ( (lm, pipeline) -> (perf, run, comp, clar, val, rig, inn, gen) ).  
    Subjective scores are scaled from [1..5] to [20..100].  
    If no data, 0.  
    """  
    task = task_name_mapping[_task]
    out = {}  
    for pipeline in PIPELINES:  
        for lm in LMS:  
            perf_list=[]  
            run_list=[]  
            comp_list=[]  
            clar_list=[]  
            val_list=[]  
            rig_list=[]  
            inn_list=[]  
            gen_list=[]  
            if pipeline==MULTI_AGENT:  
                for idx in IDEA_IDXS:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline,idx)  
                    for rid in rids:  
                        fpath=f"logs/{task}--{idx}--{IDEA_PROPOSAL_MODEL}/{lm}/{rid}/env_log/test_idea_evals.json"  
                        d=load_json_safely(fpath)  
                        if d and "implementations" in d:  
                            for imp in d["implementations"]:  
                                if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None
                                    p=imp.get("performance",0.0)  
                                    r=get_best_dev_runtime(_task, lm, pipeline, rid, idx) if task == "machine_unlearning" else imp.get("runtime",0.0)  
                                    c=imp.get("method_complexity",0.0)  
                                    perf_list.append(p)  
                                    run_list.append(r)  
                                    comp_list.append(c)  
                                    llm_eval=imp.get("llm_eval",{})  
                                    wc=llm_eval.get("with_code",{})  
                                    for subf, arr in [  
                                        ("Clarity", clar_list),  
                                        ("Validity", val_list),  
                                        ("Rigorousness", rig_list),  
                                        ("Innovativeness", inn_list),  
                                        ("Generalizability", gen_list),  
                                    ]:  
                                        rating=wc.get(subf,{}).get("Rating",0)  
                                        # scale rating 1..5 => 20..100  
                                        arr.append(rating if rating>0 else 0)  
            elif pipeline==SINGLE_AGENT:  
                rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                for rid in rids:  
                    fpath=f"logs/{task}/{lm}/{rid}/env_log/test_idea_evals.json"  
                    d=load_json_safely(fpath)  
                    if d and "implementations" in d:  
                        for imp in d["implementations"]:  
                            if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None 
                                p=imp.get("performance",0.0)  
                                r=get_best_dev_runtime(_task, lm, pipeline, rid) if task == "machine_unlearning" else imp.get("runtime",0.0)  
                                c=imp.get("method_complexity",0.0)  
                                perf_list.append(p)  
                                run_list.append(r)  
                                comp_list.append(c)  
                                llm_eval=imp.get("llm_eval",{})  
                                wc=llm_eval.get("with_code",{})  
                                for subf, arr in [  
                                    ("Clarity", clar_list),  
                                    ("Validity", val_list),  
                                    ("Rigorousness", rig_list),  
                                    ("Innovativeness", inn_list),  
                                    ("Generalizability", gen_list),  
                                ]:  
                                    rating=wc.get(subf,{}).get("Rating",0)  
                                    arr.append(rating if rating>0 else 0)  
            else:  
                rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                for rid in rids:  
                    fpath=f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{rid}/env_log/test_idea_evals.json"  
                    d=load_json_safely(fpath)  
                    if d and "implementations" in d:  
                        for imp in d["implementations"]:  
                            if imp.get("phase")=="test" and imp["performance"] is not None: # performance should not be None
                                p=imp.get("performance",0.0)  
                                r=get_best_dev_runtime(_task, lm, pipeline, rid) if task == "machine_unlearning" else imp.get("runtime",0.0)  
                                c=imp.get("method_complexity",0.0)  
                                perf_list.append(p)  
                                run_list.append(r)  
                                comp_list.append(c)  
                                llm_eval=imp.get("llm_eval",{})  
                                wc=llm_eval.get("with_code",{})  
                                for subf, arr in [  
                                    ("Clarity", clar_list),  
                                    ("Validity", val_list),  
                                    ("Rigorousness", rig_list),  
                                    ("Innovativeness", inn_list),  
                                    ("Generalizability", gen_list),  
                                ]:  
                                    rating=wc.get(subf,{}).get("Rating",0)  
                                    arr.append(rating if rating>0 else 0)  
  
            if perf_list:  
                p_ = np.mean(perf_list)  
                r_ = np.mean(run_list)  
                c_ = np.mean(comp_list)  
                cl_= np.mean(clar_list) if clar_list else 0  
                va_= np.mean(val_list) if val_list else 0  
                ri_= np.mean(rig_list) if rig_list else 0  
                inn_=np.mean(inn_list) if inn_list else 0  
                ge_= np.mean(gen_list) if gen_list else 0  
                out[(lm, pipeline)] = (p_, r_, c_, cl_, va_, ri_, inn_, ge_)  
    return out  

# handle degenerate case  
def normf(val, vmin, vmax):  
    # normalize to 1-5
    if abs(vmax-vmin)<1e-9:  
        return 50.0  
    return 1+4*(val-vmin)/(vmax-vmin) 

def plot_radar_chart_for_each_task():  
    """  
    For each real task (skip 'Average'):  
      1) load baseline data (performance, runtime, complexity) => baseline line in dashed  
      2) gather test data from each pipeline-lm => performance, runtime, complexity, plus scaled subjective  
      3) compute min & max for performance, runtime, complexity across baseline+all pipeline-lm => scale to [0..100]  
      4) subjective metrics are already scaled to [0..100] (they're 1..5 => multiply by 20)  
      5) plot the 8-dim radar with [perf, runtime, comp, Clarity, Validity, Rigorousness, Innov, Gener].  
         baseline in dashed, others in pipeline style or a single style.   
    """  
    # Our spokes  
    spokes = ["Effectiveness","Efficiency","Simplicity","Clarity","Validity","Rigorousness","Innovativeness","Generalizability"]  
    for task in TASKS:  
        # load baseline  
        baseline = load_baseline_data_for_task(task)  
        if baseline:  
            base_perf, base_run, base_comp, bcl_, bva_, bri_, binn_, bge_ = baseline  
        else:  
            assert 0, "baseline test results cannot be None!"
  
        # baseline has no subjective => set them 0  
        # We'll add them to a dictionary key "baseline"  
        # We'll gather pipeline-lm  
        pipeline_lm_data = gather_test_absolute_data_for_task(task)  # (lm,pipeline) -> (perf, run, comp, clar, val, rig, inn, gen)  
  
        # 1) gather all performance, runtime, complexity in a list for normalization  
        perf_vals = [base_perf]  
        run_vals  = [-base_run] # higher the better 
        comp_vals = [-base_comp]  
        for k,v in pipeline_lm_data.items():  
            perf_vals.append(v[0])  
            run_vals.append(-v[1])  
            comp_vals.append(-v[2])  
        # print(task)
        # print("perf_vals", perf_vals)
        # print("run_vals", run_vals)
        # print("comp_vals", comp_vals)

        pmin, pmax = min(perf_vals), max(perf_vals)  
        rmin, rmax = min(run_vals), max(run_vals)  
        cmin, cmax = min(comp_vals), max(comp_vals)  
  
 
        # We'll store lines as (label, [8 values], style, color)  
        lines=[]  
  
        # pipeline-lm lines  
        for (lm,pipeline), vals in pipeline_lm_data.items():  
            if pipeline != SINGLE_AGENT:
                continue
            (p_, r_, c_, cl_, va_, ri_, inn_, ge_) = vals  
            # normalize p_,r_,c_  
            pnorm = normf(p_, pmin, pmax)  
            rnorm = normf(-r_, rmin, rmax)  
            cnorm = normf(-c_, cmin, cmax)  
            # print(lm,pipeline)
            # print(pnorm, rnorm, cnorm)
            # subjective already scaled to [0..100]  
            # rename rigor => "Rigor"  
            linevals=[pnorm, rnorm, cnorm, cl_, va_, ri_, inn_, ge_]  
            style_ = PIPELINE_LINESTYLES.get(pipeline, 'solid')  
            color_ = LM_COLORS.get(lm, 'black')  
            label_ = f"{lm}" 
            lines.append((label_, linevals, style_, color_))  

        # baseline line should appear on top 
        bperf_ = normf(base_perf, pmin, pmax)  
        brun_  = normf(-base_run , rmin, rmax)  
        bcomp_ = normf(-base_comp, cmin, cmax)  
        # print("baseline")
        # print(bperf_, brun_, bcomp_)
        bline = [bperf_, brun_, bcomp_, bcl_, bva_, bri_, binn_, bge_]  
        lines.append(("Baseline", bline, '--', 'black'))  
  
        # Now plot  
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        # Configure the polar plot
        ax.set_theta_offset(np.pi / 2)  # Rotate the starting angle to the top
        ax.set_theta_direction(-1)  # Reverse the direction to clockwise

        # Set up the spokes
        N = len(spokes)
        angles = list(np.linspace(0, 2 * np.pi, N, endpoint=False))
        ax.set_thetagrids([a * 180 / np.pi for a in angles], spokes)
        ax.tick_params(axis='x', pad=20)  # Add padding to the spoke labels to avoid overlap

        # Plot the lines
        for (lab, vals, ls, col) in lines:
            # Close the data loop for the radar chart
            angles_ = angles + [angles[0]]
            vals_ = vals + [vals[0]]
            
            # Plot the line and fill area
            ax.plot(angles_, vals_, linestyle=ls, color=col, linewidth=1.5, label=lab)
            # Create lighter color by adjusting lightness
            rgb = mcolors.to_rgb(col)  # Convert to RGB tuple
            h, l, s = colorsys.rgb_to_hls(*rgb)  # Convert to HLS
            
            # Lighten the color (adjust 1.5 to control lightness)
            light_l = min(1.0, l * 1.5)  # Increase lightness by 50%
            light_rgb = colorsys.hls_to_rgb(h, light_l, s)
            
            # Plot fill with lighter color
            ax.fill(angles_, vals_, color=light_rgb, alpha=0.3)  # Adjust alpha if needed
                    
        # Sort the legend alphabetically
        handles, labels = ax.get_legend_handles_labels()
        sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_legend)
        plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.2, 1.05))
        plt.title(f"{task}")  
        outfn = os.path.join(RESULTS_DIR, f"radar_{task.replace(' ','_')}.pdf")  
        plt.savefig(outfn, bbox_inches='tight')  
        plt.close()  
  
        cap = (  
            f"Radar chart for {task}, showing normalized performance, runtime, complexity, and scaled subjective metrics.\n"  
            "Baseline is dashed in black, pipeline-LM combos use line style + color by pipeline + LM.\n"  
            "Objective metrics are linearly normalized to [0..100]. Subjective (1..5 => [20..100]).\n"  
        )  
        # capfn = os.path.join(RESULTS_DIR, f"radar_per_task_{task.replace(' ','_')}_caption.txt")  
        # with open(capfn, 'w') as ff:  
        #     ff.write(cap)  

def plot_combined_radar_charts():
    plt.rcParams['font.size'] = 12
    spokes = ["Effectiveness", "Efficiency", "Simplicity", "Clarity", "Validity", "Rigorousness", "Innovativeness", "Generalizability"]
    tasks = [t for t in TASKS if t != 'Average']  # Exclude 'Average' task
    
    # Determine grid layout (example: 2 rows, 3 cols for 6 tasks)
    n_rows = 2
    n_cols = 4 # (len(tasks) + 1) // 2  # Adjust as needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*6), subplot_kw={'polar': True})
    axes = axes.flatten()  # Flatten to iterate easily

    legend_elements = {}
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        # --- Data Loading and Normalization (Same as Original) ---
        baseline = load_baseline_data_for_task(task)
        base_perf, base_run, base_comp, *base_subj = baseline
        pipeline_lm_data = gather_test_absolute_data_for_task(task)
        
        # Normalize metrics
        perf_vals = [base_perf] + [v[0] for v in pipeline_lm_data.values()]
        run_vals = [-base_run] + [-v[1] for v in pipeline_lm_data.values()]
        comp_vals = [-base_comp] + [-v[2] for v in pipeline_lm_data.values()]
        
        pmin, pmax = min(perf_vals), max(perf_vals)
        rmin, rmax = min(run_vals), max(run_vals)
        cmin, cmax = min(comp_vals), max(comp_vals)
        
        # --- Baseline Data ---
        bperf = normf(base_perf, pmin, pmax)
        brun = normf(-base_run, rmin, rmax)
        bcomp = normf(-base_comp, cmin, cmax)
        bline = [bperf, brun, bcomp] + base_subj  # Assume base_subj has 5 elements
        
        # --- Pipeline Data ---
        lines = [("Baseline", bline, '--', 'black')]
        for (lm, pipeline), vals in pipeline_lm_data.items():
            if pipeline != SINGLE_AGENT:
                continue
            p, r, c, *subj = vals
            pnorm = normf(p, pmin, pmax)
            rnorm = normf(-r, rmin, rmax)
            cnorm = normf(-c, cmin, cmax)
            linevals = [pnorm, rnorm, cnorm] + subj
            style = PIPELINE_LINESTYLES.get(pipeline, 'solid')
            color = LM_COLORS.get(lm, 'black')
            lines.append((lm, linevals, style, color))
        
        # --- Plotting on Subplot ---
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        angles = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False)
        ax.set_thetagrids(np.degrees(angles), spokes, fontsize=12)
        ax.tick_params(axis='x', pad=17, labelsize=12)  # Reduced padding
        ax.tick_params(axis='y', labelsize=12)  # Smaller radial labels

        # Plot each line
        for (lab, vals, ls, col) in lines:
            assert len(angles) == len(vals)
            closed_angles = np.append(angles, angles[0])
            closed_vals = np.append(vals, vals[0])
            line = ax.plot(closed_angles, closed_vals, ls=ls, color=col, linewidth=1.5, label=lab)
            # Light fill
            light_col = mcolors.to_rgba(col, alpha=0.1)
            ax.fill(closed_angles, closed_vals, color=light_col)
            if lab not in legend_elements:
                legend_elements[lab] = line[0]
        
        # Add legend to the right of each subplot
        ax.set_title(task, fontsize=18)  # Add title padding

    
    # Hide unused subplots
    # for j in range(len(tasks), len(axes)):
    
    #  axes[j].axis('off')
    # --- Create Unified Legend ---
    # Sort legend elements alphabetically
    sorted_labels = sorted(legend_elements.keys())
    sorted_handles = [legend_elements[l] for l in sorted_labels]
    
    # Create unified legend at bottom
    fig.legend(sorted_handles, sorted_labels,
              loc='lower center',
              ncol=3,
              bbox_to_anchor=(0.5, -0.00),
              frameon=False,
              fontsize=12)
 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make space for legend
    plt.savefig(os.path.join(RESULTS_DIR, "combined_radar_charts.pdf"), bbox_inches='tight')
    plt.close()

  
##############################################################################  
# API cost budget  
##############################################################################  
  
def generate_api_cost_report():  
    cost_data=defaultdict(float)  
    for task in TASKS:  
        for pipeline in PIPELINES:  
            for lm in LMS:  
                if pipeline==MULTI_AGENT:  
                    for idx in IDEA_IDXS:  
                        rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline,idx)  
                        for rid in rids:  
                            c_=load_api_cost(task,lm,pipeline,rid,idx)  
                            cost_data[(task,pipeline,lm)]+=c_  
                elif pipeline==SINGLE_AGENT:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        cost_data[(task,pipeline,lm)]+=load_api_cost(task,lm,pipeline,rid)  
                else:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        cost_data[(task,pipeline,lm)]+=load_api_cost(task,lm,pipeline,rid)  
    lines=[]  
    lines.append("API Cost Budget Report\n")  
    total=0.0  
    pipeline_lm_sum=defaultdict(float)  
    for task in TASKS:  
        lines.append(f"Task: {task}\n")  
        for pipeline in PIPELINES:  
            for lm in LMS:  
                c_=cost_data[(task,pipeline,lm)]  
                total+=c_  
                pipeline_lm_sum[(pipeline,lm)]+=c_  
                lines.append(f"  Pipeline={pipeline}, LM={lm}, cost={round(c_,2)}")  
        lines.append("")  
    lines.append("Summaries by Pipeline+LM (across tasks):")  
    for pipeline in PIPELINES:  
        for lm in LMS:  
            c_=pipeline_lm_sum[(pipeline,lm)]  
            lines.append(f"  Pipeline={pipeline}, LM={lm}, cost={round(c_,2)}")  
    lines.append(f"\nTotal Cost of All Experiments: {round(total,2)}")  
  
    # outfn=os.path.join(RESULTS_DIR,"api_cost_report.txt")  
    # with open(outfn,"w") as f:  
    #     f.write("\n".join(lines))  
  
##############################################################################  
# Main  
##############################################################################  
  
def main():  
    # Tables 1.1 / 1.2  
    test_succ_data = compute_success_rates_data(phase='test')  
    df_1_1 = construct_table_1(test_succ_data, phase='test')  
    # dev_succ_data = compute_success_rates_data(phase='dev')  
    # df_1_2 = construct_table_1(dev_succ_data, phase='dev')  
  
    t1_1_latex = df_1_1.to_latex(index=False, escape=False)  
    # t1_2_latex = df_1_2.to_latex(index=False, escape=False)  
    with open(os.path.join(RESULTS_DIR,"table_1_1.tex"),"w") as f:  
        f.write(t1_1_latex)  
    # with open(os.path.join(RESULTS_DIR,"table_1_2.tex"),"w") as f:  
    #     f.write(t1_2_latex)  
    # with open(os.path.join(RESULTS_DIR,"table_1_1_caption.txt"),"w") as f:  
    #     f.write(f"Table 1.1: Test success rate (improvement>{TASK_THRESHOLD[task]}%).\n")  
    # with open(os.path.join(RESULTS_DIR,"table_1_2_caption.txt"),"w") as f:  
    #     f.write(f"Table 1.2: Dev success rate (best dev improvement>{TASK_THRESHOLD[task]}%).\n")  
  
    # Tables 2.1 / 2.2  
    data_2_1 = compute_average_metrics_data(phase='test', include_llm_eval=True)  
    df_2_1   = build_table_2(data_2_1, phase='test', include_llm_eval=True)  
    data_2_2 = compute_average_metrics_data(phase='dev', include_llm_eval=False)  
    df_2_2   = build_table_2(data_2_2, phase='dev', include_llm_eval=False)  
  
    t2_1_latex = df_2_1.to_latex(index=False, escape=False)  
    t2_2_latex = df_2_2.to_latex(index=False, escape=False)  
    with open(os.path.join(RESULTS_DIR,"table_2_1.tex"),"w") as f:  
        f.write(t2_1_latex)  
    with open(os.path.join(RESULTS_DIR,"table_2_2.tex"),"w") as f:  
        f.write(t2_2_latex)  
    # with open(os.path.join(RESULTS_DIR,"table_2_1_caption.txt"),"w") as f:  
    #     f.write("Table 2.1: Test-time improvement_perc, relative_runtime, relative_complexity, plus LLM metrics.\n")  
    # with open(os.path.join(RESULTS_DIR,"table_2_2_caption.txt"),"w") as f:  
    #     f.write("Table 2.2: Dev-time best improvement_perc, relative_runtime, relative_complexity.\n")  
  
    # (Original) Radar charts for 2.1 & 2.2  
    # plot_radar_chart_for_table_2(df_2_1, df_2_2)  
  
    # (NEW) Radar chart per task (skip 'Average'), using absolute performance + baseline  
    # plot_radar_chart_for_each_task()  
    plot_combined_radar_charts()
  
    # Figure 3  
    pass_data, pass_avg = compute_pass_at_k_data()  
    plot_figure_3(pass_data, pass_avg)  
  
    # Figure 4  
    fig4_data=compute_figure_4_data()  
    plot_figure_4(fig4_data)  
  
    # Scatter  
    plot_scatter_api_cost_vs_success()  
  
    # Correlation  
    plot_correlation_heatmaps()  
  
    # Cost  
    generate_api_cost_report()  
  
    print(f"All results saved under {RESULTS_DIR}")  
  
if __name__=="__main__":  
    main()  
