import os
import sys
import subprocess
import click
import pprint
import warnings
import dotenv
from dev import get_performance_from_jsons, get_quality_from_jsons, QUALITY_METRICS

dotenv.load_dotenv(override=False)
warnings.filterwarnings("ignore")


@click.group()
def cli():
    """Watermark benchmarking tool."""
    pass


def run_command(script_name, path, orgpath, isorg, args):
    cmd = [
        sys.executable,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{script_name}.py",
        ),
    ] + list(args)
    if script_name == "decode":
        # orgpath is not used
        cmd += ["--path", path]
        cmd += ["--isorg"] if isorg else []
    elif script_name == "metric":
        cmd += ["--path", path]
        cmd += ["--orgpath", orgpath]
        cmd += ["--isorg"] if isorg else []
    else:
        assert False, f"Unknown script name: {script_name}"
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


@click.command()
@click.option("--path", "-p", default="./", help="Attacked image directory.")
@click.option("--w_path", "-w", default="./", help="Watermarked image directory.")
@click.option("--uw_path", "-u", default="./", help="Unwatermarked image directory.")
@click.argument("args", nargs=-1)
def eval(path, w_path, uw_path, args):

    run_command("decode", uw_path, None, True, args)
    run_command("decode", path, None, False, args)
    run_command("metric", w_path, path, True, args)
    run_command("metric", path, w_path, False, args)

    performance_dict = get_performance_from_jsons(
        os.path.join(
            os.environ.get("RESULT_DIR"),
            "org-decode.json",
        ),
        os.path.join(
            os.environ.get("RESULT_DIR"),
            "decode.json",
        ),
        "stable_sig",
    )
    quality_dict = get_quality_from_jsons(
        os.path.join(
            os.environ.get("RESULT_DIR"),
            "org-metric.json",
        ),
        os.path.join(
            os.environ.get("RESULT_DIR"),
            "metric.json",
        ),
    )
    print("#" * 20)
    print()
    print("# Evaluation results:")
    print("### Watermark Performance:")
    print(f"Accuracy: {performance_dict['acc_1']*100:.2f}%")
    print(f"AUC Score: {performance_dict['auc_1']*100:.2f}%")
    print(f"TPR@0.1%FPR: {performance_dict['low1000_1']*100:.2f}%")
    print(f"TPR@1%FPR Score: {performance_dict['low100_1']*100:.2f}%")
    print()
    print("### Image Quality:")
    for key, value in quality_dict.items():
        if value is None:
            continue
        print(f"{QUALITY_METRICS[key]}: {value[0]:e} +/- {value[1]:e}")
    print()
    print("Warmup kit evaluation completed.")
    print("#" * 20)


# Add the subcommands to the main group
cli.add_command(eval)
