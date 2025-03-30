import os
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

from MLAgentBench.constants import ALL_BASE_PERFORMANCE as Baselines
from plot import task_name_mapping, HUMAN_PERFORMANCE 

new_human_performance = dict()
for key in HUMAN_PERFORMANCE:
    if key != 'Average':
        new_human_performance[task_name_mapping[key]] = HUMAN_PERFORMANCE[key]

HUMAN_PERFORMANCE = new_human_performance

# Threshold for considering a difference significant
THRESHOLD = 1e-3
# Threshold for considering a value essentially zero
ZERO_THRESHOLD = 1e-3

def is_none_or_nan(value):
    return value is None or (isinstance(value, float) and math.isnan(value))

def get_base_task(task_name):
    """Find the base task for a given task name based on substring matching"""
    for base_task in HUMAN_PERFORMANCE.keys():
        if base_task in task_name:
            return base_task
    return None

def is_better_than_baseline(performance, baseline):
    """Check if performance is significantly better than baseline using threshold"""
    return not is_none_or_nan(performance) and (performance - baseline) > THRESHOLD

def is_essentially_equal_to_baseline(performance, baseline):
    """Check if performance is essentially equal to baseline"""
    return not is_none_or_nan(performance) and abs(performance - baseline) <= THRESHOLD

def assign_level(task, dev_performance, test_performance, has_dev_runs, has_test_runs):
    """Determine the level of a run based on criteria"""
    # Get the base task for reference values
    base_task = get_base_task(task)
    
    # L1: no dev runs and no test runs
    if not has_dev_runs:
        return 1
    
    # L2: dev runs are there but no test runs
    if has_dev_runs and not has_test_runs:
        return 2
    
    # Both dev and test runs exist
    if has_dev_runs and has_test_runs:
        baseline_dev = Baselines[base_task]["dev"]
        baseline_test = Baselines[base_task]["test"]
        
        # For the remaining levels, we need test performance
        if is_none_or_nan(dev_performance):
            return 1
        if not is_none_or_nan(dev_performance) and is_none_or_nan(test_performance):
            return 2
        
        # Calculate margins for human and agent
        if base_task in HUMAN_PERFORMANCE:
            # Human's margin over baseline
            human_perf = HUMAN_PERFORMANCE[base_task]["performance"]
            human_margin = human_perf - baseline_test
            assert human_margin > 0
            
            # Agent's margin over baseline
            agent_margin = test_performance - baseline_test
            # L7: Agent's margin is better than human's margin
            if (agent_margin / human_margin) > 1.0:
                return 7
           
            # L6: Agent's margin is at least 5% of human's margin
            if (agent_margin / human_margin) >= 0.05:
                return 6
            
            # L5: Agent's margin is between 0% and 5% of human's margin (or essentially equal to baseline)
            if (agent_margin / human_margin) >= 0 or abs(agent_margin) < ZERO_THRESHOLD:
                return 5
        
        # L4: best dev run is better than baseline but test run is not better than baseline
        if is_better_than_baseline(dev_performance, baseline_dev) and not is_better_than_baseline(test_performance, baseline_test):
            return 4
        
        # L3: Both dev and test performance are not better than baseline
        if not is_better_than_baseline(dev_performance, baseline_dev) and not is_better_than_baseline(test_performance, baseline_test):
            return 3
    
    assert 0, "should not reach this line"

def get_run_info(run_dir):
    """Extract run information from run directory"""
    dev_file = run_dir / "env_log" / "idea_evals.json"
    test_file = run_dir / "env_log" / "test_idea_evals.json"
    
    run_info = {
        "has_dev_runs": False,
        "has_test_runs": False,
        "dev_performance": None,
        "test_performance": None
    }
    
    # Check for dev runs in idea_evals.json
    if dev_file.exists():
        try:
            with open(dev_file, 'r') as f:
                dev_data = json.load(f)
            
            # Check if there are any dev runs
            if len(dev_data.get('implementations', [])) > 0:
                run_info["has_dev_runs"] = True
                
                # Get the best performance from all dev implementations
                best_dev_perf = None
                for step in dev_data.get('implementations', []):
                    if step.get('phase') == 'dev' and not is_none_or_nan(step.get('performance')):
                        if is_none_or_nan(best_dev_perf) or step.get('performance') > best_dev_perf:
                            best_dev_perf = step.get('performance')
                
                run_info["dev_performance"] = best_dev_perf
        except (json.JSONDecodeError, FileNotFoundError) as e:
            pass
    
    # Check for test performance in test_idea_evals.json
    if test_file.exists():
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Extract test performance (should be in the last entry with phase="test")
            test_perf = None
            for step in test_data.get('implementations', []):
                if step.get('phase') == 'test':
                    test_perf = step.get('performance')
                    if not is_none_or_nan(test_perf):
                        run_info["has_test_runs"] = True
            
            run_info["test_performance"] = test_perf
        except (json.JSONDecodeError, FileNotFoundError) as e:
            pass
    
    return run_info

def get_capability_level(task, model):
    """
    Read `idea_evals.json` and `test_idea_evals.json` from runs under data/{task}/{model}
    compute L1-L6 scores
    return a list of scores like [1,3,4,3,3,5,6,1]
    """
    data_dir = Path("logs")
    task_dir = data_dir / task / model
    
    if not task_dir.exists():
        return []
    
    runs = [d for d in task_dir.iterdir() if d.is_dir()]
    levels = []
    run_details = []
    
    # Get the base task for reference values
    base_task = get_base_task(task)
    
    for run_dir in runs:
        run_info = get_run_info(run_dir)
        
        # Calculate margins for reporting
        agent_margin = None
        agent_margin_percent = None
        
        human_perf = HUMAN_PERFORMANCE[base_task]["performance"]
        baseline_test = Baselines[base_task]["test"]
        human_margin = human_perf - baseline_test
        if not is_none_or_nan(run_info["test_performance"]):
            
            agent_margin = run_info["test_performance"] - baseline_test
            
            if human_margin > 0:
                agent_margin_percent = (agent_margin / human_margin) * 100
        
        level = assign_level(
            task, 
            run_info["dev_performance"], 
            run_info["test_performance"],
            run_info["has_dev_runs"], 
            run_info["has_test_runs"]
        )
        
        levels.append(level)
        
        run_details.append({
            "run": run_dir.name,
            "level": level,
            "dev_performance": run_info["dev_performance"],
            "test_performance": run_info["test_performance"],
            "has_dev_runs": run_info["has_dev_runs"],
            "has_test_runs": run_info["has_test_runs"],
            "human_margin": human_margin,
            "agent_margin": agent_margin,
            "agent_margin_percent_of_human": agent_margin_percent,
            "baseline_dev": Baselines.get(base_task, {}).get("dev"),
            "baseline_test": Baselines.get(base_task, {}).get("test"),
            "human_performance": HUMAN_PERFORMANCE.get(base_task, {}).get("performance"),
            "base_task": base_task
        })
    
    return levels, run_details

def get_all_capability_levels():
    """Get capability levels for all task-model pairs in the data folder"""
    data_dir = Path("logs")
    result = defaultdict(dict)
    
    if not data_dir.exists():
        return result
    
    # Get all tasks from the data directory
    for task_dir in data_dir.iterdir():
        if task_dir.is_dir():
            task_scaffolding = task_dir.name
            if "o1-preview" in task_scaffolding:
                scaffolding = "CoI-Agent (o1) + MLAB"
            elif "human" in task_scaffolding:
                continue
            elif len(task_scaffolding.split('--')) > 1:
                continue
            else:
                scaffolding = "MLAB"

            base_task = task_scaffolding.split('--')[0]
           
            # Get all models for this task
            for model_dir in task_dir.iterdir():
                if model_dir.is_dir():
                    model = model_dir.name
                    if model not in LMS:
                        continue

                    levels, run_details = get_capability_level(task_scaffolding, model)
                    if levels:
                        beautiful_name = f"{scaffolding} ({model})"
                        if beautiful_name in result[base_task]:
                            levels.extend(result[base_task][beautiful_name]["levels"])
                            run_details.extend(result[base_task][beautiful_name]["run_details"])
                        result[base_task][beautiful_name] = {
                            "levels": levels,
                            "run_details": run_details,
                            "average": sum(levels) / len(levels),
                            "base_task": base_task
                        }
    
    return result

if __name__ == "__main__":
    from plot import LMS
    from collections import defaultdict
    # Get capability levels for all tasks and models in the data folder
    all_levels = get_all_capability_levels()
    
    # Save all_levels information to a JSON file
    with open("capability_levels.json", "w") as f:
        json.dump(all_levels, f, indent=2)
    
    print(f"Capability levels saved to capability_levels.json")

    # Save as leaderboard csv
    relative_improvement_to_human = defaultdict(dict)
    absolute_improvement_to_baseline = defaultdict(dict)
    for task in all_levels:
        for model in all_levels[task]:
          
            agent_margin_percent_of_human = [
                    run["agent_margin_percent_of_human"] 
                    for run in all_levels[task][model]["run_details"]
                    if not is_none_or_nan(run["agent_margin_percent_of_human"])
                    ]
            relative_improvement_to_human[task][model] = round(max(agent_margin_percent_of_human), 1)
            relative_improvement_to_human[task]["Top Human in Competition"] = 100.0

            improvement_perc = [
                    100 * run["agent_margin"] / run["baseline_test"]
                    for run in all_levels[task][model]["run_details"]
                    if not is_none_or_nan(run["agent_margin"])
                    ]
            a_run = all_levels[task][model]["run_details"][0]
            absolute_improvement_to_baseline[task][model] = round(max(improvement_perc), 1)
            absolute_improvement_to_baseline[task]["Top Human in Competition"] = round(100 * a_run["human_margin"] / a_run["baseline_test"], 1)


    metric_dir = "leaderboard_metrics/"
    os.makedirs(metric_dir, exist_ok=True)
    with open(os.path.join(metric_dir, "relative_improvement_to_human.json"), 'w') as writer:
        json.dump(relative_improvement_to_human, writer, indent=2)
    with open(os.path.join(metric_dir, "absolute_improvement_to_baseline.json"), 'w') as writer:
        json.dump(absolute_improvement_to_baseline, writer, indent=2)

    print(f"Leaderboard metricd saved under leaderboard_metrics/")
    exit(0)


    
    # Print results in a formatted way
    print("\nCapability Levels by Task and Model:")
    print("===================================")
    
    for task, models in all_levels.items():
        base_task = next(iter(models.values())).get("base_task") if models else None
        base_info = f" (based on {base_task})" if base_task and base_task != task else ""
        print(f"\n{task}{base_info}:")
        for model, data in models.items():
            levels_str = ", ".join(map(str, data["levels"]))
            avg = data["average"]
            print(f"  {model}: [{levels_str}] (Average: {avg:.2f})")
    
    # Print a summary table
    print("\n\nSummary Table (Average Capability Level):")
    print("========================================")
    
    # Get all unique models across all tasks
    all_models = set()
    for task_data in all_levels.values():
        all_models.update(task_data.keys())
    
    # Print header
    header = "Task"
    for model in sorted(all_models):
        header += f" | {model}"
    print(header)
    
    # Print separator
    separator = "-" * len("Task")
    for model in sorted(all_models):
        separator += " | " + "-" * len(model)
    print(separator)
    
    # Print data
    for task in sorted(all_levels.keys()):
        row = task
        for model in sorted(all_models):
            if model in all_levels[task]:
                avg = all_levels[task][model]["average"]
                row += f" | {avg:.2f}"
            else:
                row += " | -"
        print(row)
