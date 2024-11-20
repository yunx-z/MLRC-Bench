import json
import os
import pandas as pd


IMPLEMENTATION_MODEL = "o1-preview"
DEFAULT_METHOD_NAME = "my_merge"
report_file = "logs/report.csv" 

csv_data = []
for IDEA_PROPOSAL_MODEL in ["o1-mini", "o1-preview"]:
    for paper in ["ties", "dare", "emrmerging"]:
        idea_eval_file = f"workspace/llm-merging--{paper}--{IDEA_PROPOSAL_MODEL}/{IMPLEMENTATION_MODEL}/latest/llm-merging--{paper}--{IDEA_PROPOSAL_MODEL}/output/idea_evals.json"
        api_cost_file = f"logs/llm-merging--{paper}--{IDEA_PROPOSAL_MODEL}/{IMPLEMENTATION_MODEL}/latest/agent_log/api_cost.json"
        with open(api_cost_file, 'r') as reader:
            implementation_cost = json.load(reader)["total_cost"]
        if os.path.exists(idea_eval_file):
            with open(idea_eval_file, 'r') as reader:
                items = json.load(reader)
            best_perf_imp, best_perf = None, 0
            for imp in items["implementations"]:
                # ignore agent running baseline method
                if imp["merge_method_name"] == DEFAULT_METHOD_NAME:
                    continue
                if imp['performance'] > best_perf:
                    best_perf = imp['performance']
                    best_perf_imp = imp
        else:
            best_perf_imp = None

        if best_perf_imp is None:
            row = {
                    "merge_method_name" : "invalid implementation",
                    "anchor_paper" : paper,
                    "proposal_model" : IDEA_PROPOSAL_MODEL,
                    "implementation_model" : IMPLEMENTATION_MODEL,
                    "performance" : None,
                    "implementation_relevance_to_idea" : None,
                    "efficiency" : None,
                    "complexity" : None,
                    "implementation_cost" : implementation_cost,
                    }
        else:
            row = {
                    "merge_method_name" : best_perf_imp["merge_method_name"],
                    "anchor_paper" : paper,
                    "proposal_model" : IDEA_PROPOSAL_MODEL,
                    "implementation_model" : IMPLEMENTATION_MODEL,
                    "performance" : best_perf_imp["performance"],
                    "implementation_relevance_to_idea" : best_perf_imp["relevance_score"],
                    "efficiency" : best_perf_imp["relative_runtime"],
                    "complexity" : best_perf_imp["relative_complexity"],
                    "implementation_cost" : implementation_cost,
                    }
        csv_data.append(row)

df = pd.DataFrame(csv_data)
df.to_csv(report_file, index=False)
print(f"check out {report_file}")
