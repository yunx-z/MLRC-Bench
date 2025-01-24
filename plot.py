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
import seaborn as sns  
from collections import defaultdict  
from math import comb  
from itertools import combinations  
  
##############################################################################  
# Global config  
##############################################################################  
  
# Pipeline types  
SINGLE_AGENT = "single-agent"  
MULTI_AGENT = "multi-agent"  
HUMAN_SINGLE_AGENT = "human+single-agent"  
PIPELINES = [SINGLE_AGENT, MULTI_AGENT, HUMAN_SINGLE_AGENT]  
  
# LMs  
LMS = ["o1-mini", "gpt-4o", "claude-3-5-sonnet-v2", "gemini-exp-1206", "llama3-1-405b-instruct"]  
colors = ['#a1c9f4', '#ffb482', '#8de5a1', '#ff9f9b', '#d0bbff', '#debb9b', '#fab0e4', '#cfcfcf', '#fffea3', '#b9f2f0']
LM_COLORS = {lm : c for lm, c in zip(LMS, colors)}
# Tasks  
TASKS = ["llm-merging", "backdoor-trigger-recovery"]  
  
# Idea indices  
IDEA_IDXS = [0, 1, 2, 3]  
IDEA_PROPOSAL_MODEL = "o1-preview"  
  
# For human+single-agent  
HUMAN_IDEA_IDX = "rag"  
HUMAN_IDEA_PROPOSAL_MODEL = "human"  
  
# Consider success if improvement_perc > 5.0  
SUCCESS_THRESHOLD = 5.0  
  
# Results directory  
RESULTS_DIR = f"results/SUCCESS_THRESHOLD_{SUCCESS_THRESHOLD}"  
os.makedirs(RESULTS_DIR, exist_ok=True)  
  
# For figure line styles  
PIPELINE_LINESTYLES = {  
    SINGLE_AGENT: "solid",  
    MULTI_AGENT: "dashed",  
    HUMAN_SINGLE_AGENT: "dotted",  
}  
  
  
##############################################################################  
# Utility functions  
##############################################################################  
  
def extract_timestamp_from_dirname(dirname):  
    pattern = r'^(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$'  
    m = re.match(pattern, dirname)  
    if m:  
        return tuple(int(x) for x in m.groups())  
    return None  
  
def load_json_safely(path):  
    if not os.path.isfile(path):  
        return None  
    try:  
        with open(path, "r") as f:  
            return json.load(f)  
    except:  
        return None  
  
def find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idea_idx=None):  
    """  
    Collect run dirs from workspace and logs, unify, then keep the last 8 by ascending time.  
    """  
    ws_runs = []  
    if pipeline == SINGLE_AGENT:  
        base_pattern_ws = f"workspace/{task}/{lm}/*"  
    elif pipeline == MULTI_AGENT:  
        if idea_idx is None:  
            raise ValueError("idea_idx must be specified for multi-agent pipeline.")  
        base_pattern_ws = f"workspace/{task}--{idea_idx}--{IDEA_PROPOSAL_MODEL}/{lm}/*"  
    elif pipeline == HUMAN_SINGLE_AGENT:  
        base_pattern_ws = f"workspace/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/*"  
    else:  
        base_pattern_ws = ""  
  
    if base_pattern_ws:  
        for path in glob.glob(base_pattern_ws):  
            if os.path.isdir(path):  
                dirname = os.path.basename(path)  
                ts = extract_timestamp_from_dirname(dirname)  
                if ts is not None:  
                    ws_runs.append((dirname, ts))  
  
    log_runs = []  
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
  
    combined = {}  
    for rn, ts in ws_runs:  
        combined[rn] = ts  
    for rn, ts in log_runs:  
        if rn not in combined or combined[rn] < ts:  
            combined[rn] = ts  
  
    items = list(combined.items())  
    items.sort(key=lambda x: x[1])  # ascending  
    items = items[-8:]  
    return [x[0] for x in items]  
  
##############################################################################  
# Dev/Test result helpers  
##############################################################################  
  
def get_dev_results(task, lm, pipeline, run_id, idea_idx=None):  
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
    for imp in data.get("implementations", []):  
        if imp.get("phase") == "dev":  
            out.append(  
                (  
                    imp.get("improvement_perc", 0.0),  
                    imp.get("relative_runtime", 0.0),  
                    imp.get("relative_complexity", 0.0),  
                )  
            )  
    return out  
  
def get_test_result(task, lm, pipeline, run_id, idea_idx=None):  
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
    for imp in data.get("implementations", []):  
        if imp.get("phase") == "test":  
            return (  
                imp.get("improvement_perc", 0.0),  
                imp.get("relative_runtime", 0.0),  
                imp.get("relative_complexity", 0.0),  
            )  
    return None  
  
def load_api_cost(task, lm, pipeline, run_id, idea_idx=None):  
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
                                1 if (r and r[0]>SUCCESS_THRESHOLD) else 0  
                            )  
                        else: # dev  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_imp = max(x[0] for x in dev_res)  
                                success_list.append(1 if (best_imp>SUCCESS_THRESHOLD) else 0)  
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
                                    1 if (r and r[0]>SUCCESS_THRESHOLD) else 0  
                                )  
                            else:  
                                dev_res = get_dev_results(task, lm, pipeline, rid, idea_idx)  
                                if dev_res:  
                                    best_imp = max(x[0] for x in dev_res)  
                                    agg.append(1 if (best_imp>SUCCESS_THRESHOLD) else 0)  
                                else:  
                                    agg.append(0)  
                    success_list = agg  
                else: # human+single-agent  
                    run_ids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline)  
                    for rid in run_ids:  
                        if phase=='test':  
                            r = get_test_result(task, lm, pipeline, rid)  
                            success_list.append(  
                                1 if (r and r[0]>SUCCESS_THRESHOLD) else 0  
                            )  
                        else:  
                            dev_res = get_dev_results(task, lm, pipeline, rid)  
                            if dev_res:  
                                best_imp = max(x[0] for x in dev_res)  
                                success_list.append(1 if (best_imp>SUCCESS_THRESHOLD) else 0)  
                            else:  
                                success_list.append(0)  
  
                if success_list:  
                    mean_sr = np.mean(success_list)*100  
                    std_sr  = np.std(success_list, ddof=1)*100  
                else:  
                    mean_sr, std_sr = 0.0, 0.0  
                result[(task,pipeline,lm)] = (mean_sr, std_sr)  
    return result  
  
def construct_table_1(success_data, phase='test'):  
    rows = []  
    pipeline_lm_task_values = defaultdict(list)  
  
    for task in TASKS:  
        for i, pipeline in enumerate(PIPELINES):  
            row_task = task if i==0 else ""  
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
    return df  
  
##############################################################################  
# Table 2.x average metrics (still uses improvement & relative values)  
##############################################################################  
  
def compute_test_llm_eval_metrics(task, lm, pipeline, run_id, idea_idx=None):  
    """  
    Return the "with_code" portion of llm_eval if any.  
    """  
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
        if imp.get("phase")=="test":  
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
  
def build_table_2(average_data, phase='test', include_llm_eval=False):  
    n_extra = 5 if (include_llm_eval and phase=='test') else 0  
    cols = ["Task","System"]  
    for lm in LMS:  
        cols.append(f"Imp_{lm}")  
        cols.append(f"Run_{lm}")  
        cols.append(f"Comp_{lm}")  
        if include_llm_eval and phase=='test':  
            cols.append(f"Clar_{lm}")  
            cols.append(f"Val_{lm}")  
            cols.append(f"Rig_{lm}")  
            cols.append(f"Inn_{lm}")  
            cols.append(f"Gen_{lm}")  
  
    rows = []  
    pipeline_lm_arrays = defaultdict(lambda: [[] for _ in range(len(cols)-2)])  
  
    for task in TASKS:  
        for i,pipeline in enumerate(PIPELINES):  
            row_task = task if i==0 else ""  
            row = [row_task, pipeline]  
            for lm_i, lm in enumerate(LMS):  
                d = average_data.get((task,pipeline,lm),{})  
                imp_m, imp_s = d.get('imp_mean',0.0), d.get('imp_std',0.0)  
                run_m, run_s = d.get('run_mean',0.0), d.get('run_std',0.0)  
                comp_m,comp_s= d.get('comp_mean',0.0), d.get('comp_std',0.0)  
                row.append(f"{round(imp_m,1)}±{round(imp_s,1)}")  
                row.append(f"{round(run_m,1)}±{round(run_s,1)}")  
                row.append(f"{round(comp_m,1)}±{round(comp_s,1)}")  
  
                base = 2 + lm_i*(3+n_extra)  
                pipeline_lm_arrays[(pipeline,lm)][base-2].append(imp_m)  
                pipeline_lm_arrays[(pipeline,lm)][base-1].append(run_m)  
                pipeline_lm_arrays[(pipeline,lm)][base].append(comp_m)  
  
                if include_llm_eval and phase=='test':  
                    c_m = d.get('clarity_mean',0.0); c_s = d.get('clarity_std',0.0)  
                    v_m = d.get('validity_mean',0.0); v_s = d.get('validity_std',0.0)  
                    r_m = d.get('rigorous_mean',0.0); r_s = d.get('rigorous_std',0.0)  
                    i_m = d.get('innov_mean',0.0); i_s = d.get('innov_std',0.0)  
                    g_m = d.get('gener_mean',0.0); g_s = d.get('gener_std',0.0)  
                    row.append(f"{round(c_m,1)}±{round(c_s,1)}")  
                    row.append(f"{round(v_m,1)}±{round(v_s,1)}")  
                    row.append(f"{round(r_m,1)}±{round(r_s,1)}")  
                    row.append(f"{round(i_m,1)}±{round(i_s,1)}")  
                    row.append(f"{round(g_m,1)}±{round(g_s,1)}")  
                    pipeline_lm_arrays[(pipeline,lm)][base+1].append(c_m)  
                    pipeline_lm_arrays[(pipeline,lm)][base+2].append(v_m)  
                    pipeline_lm_arrays[(pipeline,lm)][base+3].append(r_m)  
                    pipeline_lm_arrays[(pipeline,lm)][base+4].append(i_m)  
                    pipeline_lm_arrays[(pipeline,lm)][base+5].append(g_m)  
  
            rows.append(row)  
  
    for pipeline in PIPELINES:  
        row = ["Avg", pipeline]  
        for lm_i, lm in enumerate(LMS):  
            arrs = pipeline_lm_arrays[(pipeline,lm)]  
            def mval(a):  
                return round(np.mean(a),1) if a else 0.0  
            base = lm_i*(3+n_extra)  
            i_avg = mval(arrs[base+0])  
            r_avg = mval(arrs[base+1])  
            c_avg = mval(arrs[base+2])  
            row.append(f"{i_avg}")  
            row.append(f"{r_avg}")  
            row.append(f"{c_avg}")  
            if include_llm_eval and phase=='test':  
                clar_avg = mval(arrs[base+3])  
                val_avg  = mval(arrs[base+4])  
                rig_avg  = mval(arrs[base+5])  
                inn_avg  = mval(arrs[base+6])  
                gen_avg  = mval(arrs[base+7])  
                row.append(f"{clar_avg}")  
                row.append(f"{val_avg}")  
                row.append(f"{rig_avg}")  
                row.append(f"{inn_avg}")  
                row.append(f"{gen_avg}")  
        rows.append(row)  
  
    df = pd.DataFrame(rows, columns=cols)  
    return df  
  
##############################################################################  
# Figure 3: pass@k  
##############################################################################  
  
def get_dev_improvement_success(task, lm, pipeline, run_id, threshold=5.0, idea_idx=None):  
    """  
    Return True if dev results exist with improvement>threshold.  
    """  
    dev_res = []  
    if pipeline==SINGLE_AGENT:  
        dev_res = get_dev_results(task, lm, pipeline, run_id)  
    elif pipeline==MULTI_AGENT and idea_idx is not None:  
        dev_res = get_dev_results(task, lm, pipeline, run_id, idea_idx)  
    else:  
        dev_res = get_dev_results(task, lm, pipeline, run_id)  
    return any(r[0]>threshold for r in dev_res)  
  
def compute_pass_at_k_data():  
    pass_at_k_data = {lm: defaultdict(dict) for lm in LMS}  
    m=8  
    kvals = range(1,m+1)  
  
    for lm in LMS:  
        for task in TASKS:  
            # single-agent  
            runs_sa = find_most_recent_8_runs_for_pipeline(task, lm, SINGLE_AGENT)  
            sa_successes = 0  
            for rid in runs_sa:  
                if get_dev_improvement_success(task, lm, SINGLE_AGENT, rid, threshold=SUCCESS_THRESHOLD):  
                    sa_successes+=1  
            c_impl = sa_successes  
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
            pass_at_k_data[lm][task][0] = arr_impl  
  
            # multi-agent  
            c_list=[]  
            for idea_idx in IDEA_IDXS:  
                runs_ = find_most_recent_8_runs_for_pipeline(task, lm, MULTI_AGENT, idea_idx)  
                c_ = 0  
                for rid in runs_:  
                    if get_dev_improvement_success(task, lm, MULTI_AGENT, rid, threshold=SUCCESS_THRESHOLD, idea_idx=idea_idx):  
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
        for N in [0,1,2,4]:  
            sums=[0.0]*m  
            for task in TASKS:  
                y_ = pass_at_k_data[lm][task].get(N, [0.0]*m)  
                sums=[a+b for a,b in zip(sums,y_)]  
            avg=[x/len(TASKS) for x in sums]  
            averaged[lm][N]=avg  
    return pass_at_k_data, averaged  
  
def plot_figure_3(pass_at_k_data, averaged_pass_at_k):  
    xvals=range(1,9)  
    for lm in LMS:  
        for task in TASKS+["Average"]:  
            plt.figure(figsize=(6,4))  
            for N in [0,1,2,4]:  
                if task=="Average":  
                    y=averaged_pass_at_k[lm][N]  
                    t_="Average Over All Tasks"  
                else:  
                    y=pass_at_k_data[lm][task].get(N,[0]*8)  
                    t_=task  
                if N==0:  
                    label=f"{SINGLE_AGENT}"  
                else:  
                    label=f"{MULTI_AGENT},N={N}"  
                plt.plot(xvals, y, marker='o', label=label)  
            plt.title(f"LM={lm}, Task={t_}")  
            plt.xlabel("k (# trials)")  
            plt.ylabel("pass@k")  
            plt.ylim([0,1.05])  
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
            with open(capfn,"w") as f:  
                f.write(caption)  
  
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
    metrics=["improvement_perc","relative_runtime","relative_complexity"]  
    titles=["Dev Improvement over baseline","Dev Relative runtime over baseline","Dev Relative complexity over baseline"]  
    nums=[1,2,3]  
    for i,met in enumerate(metrics):  
        for task in TASKS+["Average"]:  
            plt.figure(figsize=(8,6))  
            for lm in LMS:  
                for pipeline in PIPELINES:  
                    arr=fig4_data[task][(lm,pipeline)][met]  
                    if not arr:  
                        continue  
                    xvals=range(1,len(arr)+1)  
                    style_=PIPELINE_LINESTYLES.get(pipeline,'solid')  
                    color_=LM_COLORS.get(lm,'black')  
                    lab=f"{lm}, {pipeline}"  
                    plt.plot(xvals, arr, marker='o', linestyle=style_, color=color_, label=lab)  
            tt=f"Figure 4.{nums[i]}: {titles[i]} ({task})" if task!="Average" else f"Figure 4.{nums[i]}: {titles[i]} (All Tasks Avg)"  
            plt.title(tt)  
            plt.xlabel("i-th implementation")  
            plt.ylabel(met)  
            plt.grid(True)  
            plt.legend()  
            outfn= os.path.join(RESULTS_DIR,f"figure_4_{nums[i]}_{task.replace(' ','_')}.pdf")  
            plt.savefig(outfn,bbox_inches='tight')  
            plt.close()  
            cap=(  
                f"Figure 4.{nums[i]} for {task}. "  
                f"Plots {met} vs implementation index. Different line styles for pipeline, different colors for LM."  
            )  
            capfn= os.path.join(RESULTS_DIR,f"figure_4_{nums[i]}_{task.replace(' ','_')}_caption.txt")  
            with open(capfn,"w") as f:  
                f.write(cap)  
  
##############################################################################  
# Scatter: cost vs success  
##############################################################################  
  
def compute_api_cost_and_success_for_scatter():  
    data_points=[]  
    for pipeline in PIPELINES:  
        for lm in LMS:  
            costs=[]  
            successes=[]  
            for task in TASKS:  
                if pipeline==MULTI_AGENT:  
                    for idx in IDEA_IDXS:  
                        rids = find_most_recent_8_runs_for_pipeline(task, lm, pipeline, idx)  
                        for rid in rids:  
                            c_=load_api_cost(task,lm,pipeline,rid,idx)  
                            costs.append(c_)  
                            res=get_test_result(task,lm,pipeline,rid,idx)  
                            s=1 if (res and res[0]>SUCCESS_THRESHOLD) else 0  
                            successes.append(s)  
                elif pipeline==SINGLE_AGENT:  
                    rids = find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        c_=load_api_cost(task,lm,pipeline,rid)  
                        costs.append(c_)  
                        res=get_test_result(task,lm,pipeline,rid)  
                        s=1 if(res and res[0]>SUCCESS_THRESHOLD) else 0  
                        successes.append(s)  
                else:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        c_=load_api_cost(task,lm,pipeline,rid)  
                        costs.append(c_)  
                        res=get_test_result(task,lm,pipeline,rid)  
                        s=1 if(res and res[0]>SUCCESS_THRESHOLD) else 0  
                        successes.append(s)  
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
    data=compute_api_cost_and_success_for_scatter()  
    shape_map={SINGLE_AGENT:'o',MULTI_AGENT:'s',HUMAN_SINGLE_AGENT:'^'}  
    plt.figure(figsize=(7,5))  
    for dp in data:  
        x=dp["avg_cost"]  
        y=dp["avg_sr"]  
        lm=dp["lm"]  
        pipeline=dp["pipeline"]  
        sh=shape_map.get(pipeline,'o')  
        col=LM_COLORS.get(lm,'black')  
        plt.scatter(x,y,marker=sh,color=col,s=100,edgecolors='black')  
        plt.text(x+0.01,y,f"{lm}\n({pipeline})",fontsize=9)  
    pf=compute_pareto_front(data)  
    pf_x=[p["avg_cost"] for p in pf]  
    pf_y=[p["avg_sr"] for p in pf]  
    plt.plot(pf_x,pf_y,color='green',linestyle='--',label='Pareto Front')  
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
    with open(os.path.join(RESULTS_DIR,"scatter_api_cost_vs_success_caption.txt"),"w") as f:  
        f.write(cap)  
  
##############################################################################  
# Correlation heatmaps  
##############################################################################  
  
def gather_test_correlation_data():  
    records_with=[]  
    records_without=[]  
    subj_fields=["Clarity","Validity","Rigorousness","Innovativeness","Generalizability"]  
    obj_fields=["improvement_perc","relative_runtime","relative_complexity"]  
  
    def process_imp(imp):  
        base={  
            "improvement_perc":imp.get("improvement_perc",0.0),  
            "relative_runtime":-imp.get("relative_runtime",0.0),  # higher the better
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
  
    for task in TASKS:  
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
                                    if imp.get("phase")=="test":  
                                        process_imp(imp)  
            elif pipeline==SINGLE_AGENT:  
                for lm in LMS:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        test_file=f"logs/{task}/{lm}/{rid}/env_log/test_idea_evals.json"  
                        dat=load_json_safely(test_file)  
                        if dat and "implementations" in dat:  
                            for imp in dat["implementations"]:  
                                if imp.get("phase")=="test":  
                                    process_imp(imp)  
            else:  
                for lm in LMS:  
                    rids=find_most_recent_8_runs_for_pipeline(task,lm,pipeline)  
                    for rid in rids:  
                        test_file=f"logs/{task}--{HUMAN_IDEA_IDX}--{HUMAN_IDEA_PROPOSAL_MODEL}/{lm}/{rid}/env_log/test_idea_evals.json"  
                        dat=load_json_safely(test_file)  
                        if dat and "implementations" in dat:  
                            for imp in dat["implementations"]:  
                                if imp.get("phase")=="test":  
                                    process_imp(imp)  
  
    df_with=pd.DataFrame(records_with)  
    df_without=pd.DataFrame(records_without)  
    return df_with, df_without  
  
def plot_correlation_heatmaps():
    dfw, dfwo = gather_test_correlation_data()
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
        corr_ = df_[ex].corr()
        rAvail = [f for f in subj if f in corr_.index]
        cAvail = [f for f in obj if f in corr_.columns]
        if not rAvail or not cAvail:
            continue
        subcorr = corr_.loc[rAvail, cAvail]
        plt.figure(figsize=(4, 3))
        ax = sns.heatmap(
            subcorr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
            xticklabels=[obj_label_map.get(x, x) for x in subcorr.columns], 
            cbar_kws={'label': 'Correlation'}
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Horizontal labels
        plt.title(f"{nm}")
        outfn = os.path.join(RESULTS_DIR, f"correlation_heatmap_{nm}.pdf")
        plt.savefig(outfn, bbox_inches='tight')
        plt.close()
        cap = (
            f"Correlation heatmap ({nm}) between LLM-as-a-Judge subjective metrics (rows) "
            f"and ML objective metrics (cols) for valid test implementations."
        )
        with open(os.path.join(RESULTS_DIR, f"correlation_heatmap_{nm}_caption.txt"), "w") as f:
            f.write(cap)

  
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
        with open(os.path.join(RESULTS_DIR,"radar_table_2_1_caption.txt"),"w") as f:  
            f.write("Radar chart for Table 2.1 (test) with objective+subjective metrics.\n")  
  
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
        with open(os.path.join(RESULTS_DIR,"radar_table_2_2_caption.txt"),"w") as f:  
            f.write("Radar chart for Table 2.2 (dev) with objective metrics only.\n")  
  
##############################################################################  
# NEW Radar chart with absolute performance + baseline  
##############################################################################  
  
def load_baseline_data_for_task(task):  
    """  
    Read from MLAgentBench/benchmarks_base_exp/{task}/env/output/idea_evals.json  
    Return (perf, runtime, complexity) as the baseline (test).  
    If there's multiple test items, average them. If none, return None.  
    """  
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

    for imp in data.get("implementations", []):  
        if imp.get("phase")=="test":  
            perf_vals.append(imp.get("performance", 0.0))  
            run_vals.append(imp.get("runtime",0.0))  
            comp_vals.append(imp.get("method_complexity",0.0))  
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

    if not perf_vals:  
        return None  
    return (  
        np.mean(perf_vals),  
        np.mean(run_vals),  
        np.mean(comp_vals),  
        np.mean(clar_list),
        np.mean(val_list),
        np.mean(rig_list),
        np.mean(inn_list),
        np.mean(gen_list)
    )  
  
def gather_test_absolute_data_for_task(task):  
    """  
    For each pipeline, LM, gather the average (performance, runtime, method_complexity, clarity, validity, rigorousness, innov, gener).  
    We'll parse logs from up to 8 runs. Each run has test_idea_evals.json with a single 'test' item if valid.  
    We'll average them. Return a dict ( (lm, pipeline) -> (perf, run, comp, clar, val, rig, inn, gen) ).  
    Subjective scores are scaled from [1..5] to [20..100].  
    If no data, 0.  
    """  
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
                                if imp.get("phase")=="test":  
                                    p=imp.get("performance",0.0)  
                                    r=imp.get("runtime",0.0)  
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
                            if imp.get("phase")=="test":  
                                p=imp.get("performance",0.0)  
                                r=imp.get("runtime",0.0)  
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
                            if imp.get("phase")=="test":  
                                p=imp.get("performance",0.0)  
                                r=imp.get("runtime",0.0)  
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
        run_vals  = [-base_run]  
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
  
        # handle degenerate case  
        def normf(val, vmin, vmax):  
            # normalize to 1-5
            if abs(vmax-vmin)<1e-9:  
                return 50.0  
            return 1+4*(val-vmin)/(vmax-vmin) 
  
        # We'll store lines as (label, [8 values], style, color)  
        lines=[]  
  
        # baseline line  
        bperf_ = normf(base_perf, pmin, pmax)  
        brun_  = normf(-base_run , rmin, rmax)  
        bcomp_ = normf(-base_comp, cmin, cmax)  
        # print("baseline")
        # print(bperf_, brun_, bcomp_)
        bline = [bperf_, brun_, bcomp_, bcl_, bva_, bri_, binn_, bge_]  
        lines.append(("Baseline", bline, '--', 'black'))  
  
        # pipeline-lm lines  
        for (lm,pipeline), vals in pipeline_lm_data.items():  
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
            label_ = f"{lm},{pipeline}"  
            lines.append((label_, linevals, style_, color_))  
  
        # Now plot  
        fig=plt.figure(figsize=(6,6))  
        ax=plt.subplot(111, polar=True)  
        ax.set_theta_offset(np.pi/2)  
        ax.set_theta_direction(-1)  
        N=len(spokes)  
        angles=list(np.linspace(0,2*np.pi,N,endpoint=False))  
        ax.set_thetagrids([a*180/np.pi for a in angles], spokes)  
  
        for (lab, vals, ls, col) in lines:  
            # close  
            angles_ = angles+[angles[0]]  
            vals_   = vals+[vals[0]]  
            ax.plot(angles_, vals_, linestyle=ls, color=col, linewidth=1.5, label=lab)  
            ax.fill(angles_, vals_, alpha=0.1, color=col)  
		    
	# Sort the legend alphabetically
        handles, labels = ax.get_legend_handles_labels()
        sorted_legend = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_legend)
        plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.2, 1.05))
        plt.title(f"{task}")  
        outfn = os.path.join(RESULTS_DIR, f"radar_per_task_{task.replace(' ','_')}.pdf")  
        plt.savefig(outfn, bbox_inches='tight')  
        plt.close()  
  
        cap = (  
            f"Radar chart for {task}, showing normalized performance, runtime, complexity, and scaled subjective metrics.\n"  
            "Baseline is dashed in black, pipeline-LM combos use line style + color by pipeline + LM.\n"  
            "Objective metrics are linearly normalized to [0..100]. Subjective (1..5 => [20..100]).\n"  
        )  
        capfn = os.path.join(RESULTS_DIR, f"radar_per_task_{task.replace(' ','_')}_caption.txt")  
        with open(capfn, 'w') as ff:  
            ff.write(cap)  
  
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
  
    outfn=os.path.join(RESULTS_DIR,"api_cost_report.txt")  
    with open(outfn,"w") as f:  
        f.write("\n".join(lines))  
  
##############################################################################  
# Main  
##############################################################################  
  
def main():  
    # Tables 1.1 / 1.2  
    test_succ_data = compute_success_rates_data(phase='test')  
    df_1_1 = construct_table_1(test_succ_data, phase='test')  
    dev_succ_data = compute_success_rates_data(phase='dev')  
    df_1_2 = construct_table_1(dev_succ_data, phase='dev')  
  
    t1_1_latex = df_1_1.to_latex(index=False, escape=False)  
    t1_2_latex = df_1_2.to_latex(index=False, escape=False)  
    with open(os.path.join(RESULTS_DIR,"table_1_1.tex"),"w") as f:  
        f.write(t1_1_latex)  
    with open(os.path.join(RESULTS_DIR,"table_1_2.tex"),"w") as f:  
        f.write(t1_2_latex)  
    with open(os.path.join(RESULTS_DIR,"table_1_1_caption.txt"),"w") as f:  
        f.write(f"Table 1.1: Test success rate (improvement>{SUCCESS_THRESHOLD}%).\n")  
    with open(os.path.join(RESULTS_DIR,"table_1_2_caption.txt"),"w") as f:  
        f.write(f"Table 1.2: Dev success rate (best dev improvement>{SUCCESS_THRESHOLD}%).\n")  
  
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
    with open(os.path.join(RESULTS_DIR,"table_2_1_caption.txt"),"w") as f:  
        f.write("Table 2.1: Test-time improvement_perc, relative_runtime, relative_complexity, plus LLM metrics.\n")  
    with open(os.path.join(RESULTS_DIR,"table_2_2_caption.txt"),"w") as f:  
        f.write("Table 2.2: Dev-time best improvement_perc, relative_runtime, relative_complexity.\n")  
  
    # (Original) Radar charts for 2.1 & 2.2  
    # plot_radar_chart_for_table_2(df_2_1, df_2_2)  
  
    # (NEW) Radar chart per task (skip 'Average'), using absolute performance + baseline  
    plot_radar_chart_for_each_task()  
  
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
