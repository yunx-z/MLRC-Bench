from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import ANETdetection, fix_random_seed, AverageMeter, postprocess_results
import torch
import torch.nn as nn
import os
import time
import json
import numpy as np
from pprint import pprint

def valid_one_epoch(val_loader, model, task, evaluator=None, output_file=None, ext_score_file=None, print_freq=20):
    """Run evaluation for one epoch

    Args:
        val_loader: DataLoader for validation set
        model: Model to evaluate
        task: Task name (e.g., 'action_localisation')
        evaluator: ANETdetection evaluator instance
        output_file: Path to save results
        ext_score_file: Path to external classification scores
        print_freq: How often to print progress

    Returns:
        dict: Results dictionary in ActivityNet format
    """
    batch_time = AverageMeter()
    model.eval()

    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            output = model(video_list)

            # Unpack results
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        if (iter_idx != 0) and iter_idx % print_freq == 0:
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # Convert lists to numpy arrays
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    # Apply postprocessing if external scores are provided
    if ext_score_file is not None and isinstance(ext_score_file, str):
        print(f"Applying postprocessing with external scores from {ext_score_file}")
        results = postprocess_results(results, ext_score_file)

    # Save results in ActivityNet format
    if output_file is not None:
        results_dict = {}
        for idx, vid in enumerate(results['video-id']):
            if vid not in results_dict:
                results_dict[vid] = {task: []}

            results_dict[vid][task].append({
                'label': str(results['label'][idx]),
                'score': str(results['score'][idx]),
                'timestamps': [float(results['t-start'][idx]), float(results['t-end'][idx])]
            })

        with open(output_file, 'w') as f:
            json.dump(results_dict, f)

    return results

def evaluate_model(Method, phase):
    """Run model evaluation and save results

    Args:
        Method: Method instance with run() method
        phase: 'dev' or 'test' phase
    """
    if phase == "debug":
        print("debugging")
        return
    # Get model and config from method - use "valid" or "test" mode
    model, cfg = Method.run("valid" if phase == "dev" else "test")
    pprint(cfg)

    # Fix random seeds
    _ = fix_random_seed(0, include_cuda=True)

    # Create dataset and loader
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader'].get('num_workers', 4)
    )

    # Load checkpoint
    ckpt_path = os.path.join("ckpt", Method.get_name(), "model_best.pth.tar")
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(
        ckpt_path,
        map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
    )

    # Load EMA model state which typically performs better
    if 'state_dict_ema' in checkpoint:
        print("Loading EMA model state...")
        model.load_state_dict(checkpoint['state_dict_ema'])
    else:
        print("Loading regular model state...")
        model.load_state_dict(checkpoint['state_dict'])

    # Create output path and run evaluation
    output_file = os.path.join("output", f"{Method.get_name()}_{phase}_results.json")
    _ = valid_one_epoch(
        val_loader,
        model,
        val_dataset.task,
        output_file=output_file,
        ext_score_file=cfg.get('test_cfg', {}).get('ext_score_file', None),
        print_freq=10
    )

def get_score(Method, phase):
    """Calculate mAP score from saved results

    Args:
        Method: Method instance
        phase: 'dev' or 'test' phase

    Returns:
        float: Average mAP score
    """
    # Load results file
    results_file = os.path.join("output", f"{Method.get_name()}_{phase}_results.json")
    if not os.path.exists(results_file):
        print(f"No results file found at {results_file}")
        return 0.0

    # Load and transform results into expected format
    with open(results_file, 'r') as f:
        results_dict = json.load(f)

    # Transform into flat format with numpy arrays
    flat_results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # Load model+cfg so we know the task key
    _, cfg = Method.run("valid" if phase == "dev" else "test")
    dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    task_key = dataset.task       # now "sound_localisation"

    for vid_id, vid_data in results_dict.items():
        for action in vid_data[task_key]:
            flat_results['video-id'].append(vid_id)
            flat_results['t-start'].append(float(action['timestamps'][0]))
            flat_results['t-end'].append(float(action['timestamps'][1]))
            flat_results['label'].append(int(action['label']))
            flat_results['score'].append(float(action['score']))

    # Convert lists to numpy arrays
    flat_results['t-start'] = np.array(flat_results['t-start'])
    flat_results['t-end'] = np.array(flat_results['t-end'])
    flat_results['label'] = np.array(flat_results['label'])
    flat_results['score'] = np.array(flat_results['score'])

    # Setup evaluator
    db_vars = dataset.get_attributes()
    evaluator = ANETdetection(
        dataset.json_file,
        dataset.task,
        dataset.label_dict,
        dataset.split[0],
        tiou_thresholds = db_vars['tiou_thresholds']
    )

    # Load results and calculate mAP
    _, average_mAP, _ = evaluator.evaluate(flat_results, verbose=True)

    print(f"\nEvaluation Results for {Method.get_name()} on {phase} set:")
    print(f"Average mAP: {average_mAP*100:.2f}%")

    return average_mAP
