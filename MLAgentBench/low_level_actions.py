""" This file contains the low level actions that are provided by the environment, mostly file system operations and code execution. """

import os
import json
import subprocess
import selectors
import shutil
import glob
import sys
import inspect
from functools import wraps
import time
from io import StringIO
from .schema import Step, ActionInfo, Action, EnvException
import readline # This is needed to make sure that the input() function works properly

def safe_path_join(*paths):
    cleaned_paths = [p.replace('\\/', '/') for p in paths]
    return os.path.join(*cleaned_paths)

def safe_copy_file(src, dst):
    # make dir for dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def normalize_args_kwargs(f, *args, **kwargs):
    """ This function takes a function and its arguments and returns a dictionary of the arguments, with the keys being the argument names."""
    sig = inspect.signature(f)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()  # This line is optional, it fills in any omitted arguments that have default values
    return bound.arguments

def append_to_low_level_steps(trace, name, args, observation):
    """ This function appends a low level step to the trace. """
    trace.low_level_steps.append(Step(action=Action(name, args),observation=observation,timestamp=time.time()))


def record_low_level_step(func):
    """ This decorator records a low level step in the trace."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
        if "trace" not in new_kwargs["kwargs"]:
            print("Warning: trace not found in kwargs; not recording low level step.")
            print(func)
            return func(*args, **kwargs)
        else:
            trace = new_kwargs["kwargs"]["trace"]
            for a in LOW_LEVEL_ACTIONS:
                if a.function.__name__ == func.__name__:
                    name = a.name
                    input_args = a.usage.keys()
                    break
            new_kwargs = {k: v for k, v in new_kwargs.items() if k in input_args}
            append_to_low_level_steps(trace, name, new_kwargs, observation=None)
            # try:
            #     observation = func(*args, **kwargs)
            #     append_to_low_level_steps(trace, name, new_kwargs, observation)
            #     return observation
            # except EnvironmentError as e:
            #     append_to_low_level_steps(trace, name, new_kwargs, e)
            #     raise EnvException(e)
    return wrapper


def check_file_read_only(arg_names, **kwargs):
    """ This decorator checks if the file is read-only. """
    
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
            for arg_name in arg_names:
                if new_kwargs[arg_name] in new_kwargs["kwargs"]["read_only_files"]:
                    raise EnvException(f"cannot write / copy / modify the file {new_kwargs[arg_name]} because it is a read-only file.")
            return func(*args, **kwargs)
        return wrapper
    return inner


def check_file_in_work_dir(arg_names, **kwargs):
    """ This decorator checks if the file is in the work directory. """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
            work_dir = new_kwargs["work_dir"]
            for arg_name in arg_names:
                file_name = new_kwargs[arg_name]
                if not os.path.abspath(safe_path_join(work_dir, file_name)).startswith(os.path.abspath(work_dir)):
                    raise EnvException(f"cannot access file {file_name} because it is not in the work directory.")
            return func(*args, **kwargs)
        return wrapper
    return inner

@check_file_in_work_dir(["dir_path"])
@record_low_level_step
def list_files(dir_path, work_dir=".", max_length=10000, **kwargs):
    """
    List files in a directory with truncation for long outputs.
    
    Args:
        dir_path (str): Path to the directory to list
        work_dir (str): Working directory (defaults to ".")
        max_length (int): Maximum length of output before truncation (defaults to 10000)
        **kwargs: Additional keyword arguments
        
    Returns:
        str: Directory listing, truncated if exceeds max_length
        
    Raises:
        EnvException: If directory listing fails
    """
    try:
        # avoid "\\"
        observation = subprocess.check_output(["ls", "-F", safe_path_join(work_dir, dir_path)]).decode("utf-8")
        
        # Truncate if observation exceeds max_length
        if len(observation) > max_length:
            return observation[:max_length] + "...TRUNCATED"
        
        return observation
    except:
        raise EnvException(f"Cannot list file in the {dir_path} directory")
    



@check_file_in_work_dir(["file_name"])
@record_low_level_step
def read_file(file_name, work_dir = '.', **kwargs):
    try:
        observation = open(safe_path_join(work_dir,file_name)).read()
        return observation
    except:
        raise EnvException(f"cannot read file {file_name}")


@check_file_in_work_dir(["file_name"])
@check_file_read_only(["file_name"])
@record_low_level_step
def write_file(file_name, content, work_dir = ".", **kwargs):
    try:
        # create dest dir if not exists
        dst = safe_path_join(work_dir,file_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(content)
        observation = f"File {file_name} written successfully."
        return observation
    except:
        raise EnvException(f"cannot write file {file_name}")


@check_file_in_work_dir(["file_name"])
@check_file_read_only(["file_name"])
@record_low_level_step
def append_file(file_name, content, work_dir = ".", **kwargs):
    try:
        with open(safe_path_join(work_dir,file_name), "a") as f:
            f.write(content)
        observation = f"File {file_name} appended successfully."
        return observation
    except:
        raise EnvException(f"cannot append file {file_name}")


@check_file_in_work_dir(["source", "destination"])
@check_file_read_only(["destination"])
@check_file_read_only(["source"])
@record_low_level_step
def copy_file( source, destination, work_dir = ".", **kwargs):
    
    try:
        safe_copy_file(safe_path_join(work_dir,source), safe_path_join(work_dir,destination))
        observation = f"File {source} copied to {destination}"
        return observation
    except:
        raise EnvException(f"File {source} copy to {destination} failed. Check whether the source and destinations are valid.")


@check_file_in_work_dir(["script_name_and_args"])
@record_low_level_step
def undo_edit_script( script_name_and_args, work_dir = ".", **kwargs):
    
    backup_files = glob.glob(safe_path_join(work_dir,"backup", f"{script_name_and_args}_*"))
    if len(backup_files) == 0:
        raise EnvException("There is no change to undo.")
    try:
        backup_files.sort()
        backup_file = backup_files[-1]
        safe_copy_file(backup_file, safe_path_join(work_dir,script_name_and_args))
        # delete the backup file
        os.remove(backup_file)

        new_content = open(safe_path_join(work_dir,script_name_and_args)).read()
        observation = f"Content of {script_name_and_args} after undo the most recent edit:\n" + new_content
        return observation
    except:
        raise EnvException(f"Cannot undo the edit of file name {script_name_and_args}. Check the file name again."
        )


# @check_file_in_work_dir(["script_name_and_args"])
@record_low_level_step
def execute_script(script_name_and_args, work_dir = ".", **kwargs):
    script_name_and_args = script_name_and_args.replace("python ", "")
    script_name = script_name_and_args.split(' ')[0]
    if not os.path.exists(safe_path_join(work_dir,script_name)):
        raise EnvException(f"The file {script_name} does not exist.")
    try:
        script_path = script_name_and_args
        device = kwargs["device"]
        python = kwargs["python"]

        cmd = f"PYTHONPATH=`pwd` CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir)

        stdout_lines = []
        stderr_lines = []

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)

            for key, _ in events:
                line = key.fileobj.readline()
                if key.fileobj == process.stdout:
                    print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                else:
                    print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

        for line in process.stdout:
            line = line
            print("STDOUT:", line, end =" ")
            stdout_lines.append(line)
        for line in process.stderr:
            line = line
            print("STDERR:", line, end =" ")
            stderr_lines.append(line)

        return_code = process.returncode

        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            # printed to stderr only
            observation = "".join(stderr_lines)
        return "The script has been executed. Here is the output:\n" + observation
    except Exception as e:
        raise EnvException(f"Something went wrong in executing `python {script_name_and_args}`: {e}. Please check if it is ready to be executed.")


@record_low_level_step
def python_repl(command, work_dir = ".", **kwargs):
    """Run command and returns anything printed."""
    try:
        cwd = os.getcwd()
        import codeop
        compiler = codeop.CommandCompiler()
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            command = compiler(command)
            os.chdir(work_dir)
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        os.chdir(cwd)
        return output
    except Exception as e:
        raise EnvException(f"Something went wrong in executing {command}: {e}")


@record_low_level_step
def request_help(request, work_dir = ".", **kwargs):
    return input(f"Research Assistant is requesting help: {request}\n")

@record_low_level_step
def final_answer(final_solution, best_score, work_dir = ".", **kwargs):
    try:
        score_float = float(best_score)
    except Exception as e:
        raise EnvException(f"Failed to convert best_score ({best_score}) into a number. Please only include the number.")
    eval_file = safe_path_join(work_dir, "output/idea_evals.json")
    if not os.path.exists(eval_file):
        raise EnvException(f"You haven't made any valid submission to the leaderboard yet")
    with open(eval_file, 'r') as reader:
        all_scores = [eval_result['performance'] for eval_result in json.load(reader)["implementations"]]
    score_float_in_all_scores = False
    for s in all_scores:
        if abs(score_float - s) < 1e-3:
            score_float_in_all_scores = True
            break
    if not score_float_in_all_scores:
        raise EnvException(f"Your submission didn't achieve a score of {best_score} according to the leaderboard records. Please double check and resubmit a valid final answer.")
    return ""

### describe the low level actions
LOW_LEVEL_ACTIONS = [
    ActionInfo(
        name="List Files",
        description="Use this to navigate the file system.",
        usage={   
            "dir_path": "a valid relative path to a directory, such as \".\" or \"folder1/folder2\""
        },
        return_value="The observation will be a list of files and folders in dir_path or current directory is dir_path is empty, or an error message if dir_path is invalid.",
        function=list_files,
        is_primitive=True
    ),
    ActionInfo(
        name="Read File",
        description="Use this to read an existing file.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed"
        },
        return_value="The observation will be the contents of the file read.",
        function=read_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Write File",
        description="Use this to write a file. If the file already exists, it will be overwritten.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be written to the file"
        },
        return_value="A success message if the file is written successfully, or an error message if the file cannot be written.",
        function=write_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Append File",
        description="Use this to append a file to a new location with a new name.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be appended to the file"
        },
        return_value="A success message if the file is appended successfully, or an error message if the file cannot be appended.",
        function=append_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Copy File",
        description="Use this to copy a file to a new location with a new name.",
        usage={
            "source": "a valid file name with relative path to current directory if needed",
            "destination": "a valid file name with relative path to current directory if needed"
        },
        return_value="A success message if the file is copied successfully, or an error message if the file cannot be copied.",
        function=copy_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Undo Edit Script",
        description="Use this to undo the last edit of the python script.",
        usage={
            "script_name_and_args": "a valid python script name with relative path to current directory if needed"
        },
        return_value="The observation will be the content of the script before the last edit. If the script does not exist, the observation will be an error message.",
        function=undo_edit_script,
        is_primitive=True
    ),
    ActionInfo(
        name="Execute Script",
        description="Use this to execute the python script. The script must already exist.",
        usage={
            "script_name_and_args": "a valid python script name with relative path to current directory if needed, followed by any arguments if necessary, such as \"python run.py --model gpt-4o\" or \"python setup.py install\""
        },
        return_value="The observation will be output of the script or errors.",
        function=execute_script,
        is_primitive=True
    ),
    ActionInfo(
        name="Python REPL",
        description="A python REPL. Use this to execute single line python commands.",
        usage={
            "command": "a valid python command"
        },
        return_value="The observation will be output of the command or errors.",
        function=python_repl,
        is_primitive=True 
    ),
    ActionInfo(
        name="Request Help",
        description="Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library. So you should check through the provided tools and files first.",
        usage={
            "request": "a detailed description on what to do"
        },
        return_value="The observation will be the response from human.",
        function=request_help,
        is_primitive=True
    ),
    ActionInfo(
        name="Final Answer",
        description="Use this to provide the best solution and the corresponding evaluation score for the current task. You should not use this tool unless you have exhausted all avenues for improving your solution.",
        usage={
            "final_solution": "a detailed description on the best solution you developed",
            "best_score": "the evaluation score for the best solution, number only"
        },
        return_value="The observation will be empty.",
        function=final_answer,
        is_primitive=True
    ),
]
