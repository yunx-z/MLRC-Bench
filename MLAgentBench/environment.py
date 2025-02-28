"""
This file contains the Environment class, which prepares the environment for the research agent to run in.
"""

try:
    import json
    import os
    import sys
    import subprocess
    import shutil
    import copy
    import time
    import fnmatch
    import signal
    from traceback import format_exception
    from multiprocessing import active_children
    import readline # to make sure input() works properly
    from dacite import from_dict
except Exception as e:
    pass

from .low_level_actions import LOW_LEVEL_ACTIONS
from .high_level_actions import HIGH_LEVEL_ACTIONS
from .schema import Step, Trace, EnvException, TooLongPromptError, LLMError, EnhancedJSONEncoder 
from .LLM import complete_text_claude
from .prepare_task import prepare_task, get_task_info

class TimeoutException(Exception): pass


def create_benchmark_folder_name(research_problem, log_file):
    """Create a benchmark folder name from a research problem in interactive mode"""

    prompt = f"""Can you give a short name for this research problem ? 

{research_problem}"

The name should be short and valid as a folder name, e.g. "my_research_problem". Given the name in this format:
[research problem name]: my_research_problem [end]

"""
    response = complete_text_claude(prompt, stop_sequences=["[end]"], log_file=log_file)
    benchmark_folder_name = response.split("[research problem name]: ")[-1].split(" [end]")[0]
    return benchmark_folder_name


class Environment:
    def __init__(self, args):

        self._args = args
        self._log_dir = os.path.join(args.log_dir, "env_log")
        self._setup_log_dir()

        if not args.interactive:
            self._benchmark_folder_name, self._research_problem = get_task_info(args.task)
            self._work_dir = os.path.join(args.work_dir, self.benchmark_folder_name)
            self._read_only_files = []
            self._initialize_task_env() # set up work dir and log dir

        else:
            self._research_problem = input("What is the task: ")
            log_file = os.path.join(self.log_dir, "create_benchmark_folder_name.log")
            self._benchmark_folder_name = create_benchmark_folder_name(self._research_problem, log_file)
            self._work_dir = args.work_dir
            w = input(f"Enter the folder path (Research Assistant will not read anything outside this folder) default: {args.work_dir}: ")
            if w != "":
                self._work_dir = w
            self._read_only_files = []

            self._initialize_interactive_env() # set up work dir and log dir

        self._action_infos =  {t.name: t for t in LOW_LEVEL_ACTIONS + HIGH_LEVEL_ACTIONS}

        if not args.interactive:
            del self._action_infos["Request Help"]

        self._static_kwargs_for_tools = {
            "device": args.device,
            "python": args.python,
            "work_dir": self.work_dir,
            "args": args,
            "read_only_files": self.read_only_files,
            "research_problem": self.research_problem,
        }
        self._trace = self._initialize_trace()
        self._start_time = time.time()
        self._submit_valid_answer = False
        self.finish_reason = ""

    ############################## getters ########################################

    @property
    def research_problem(self):
        return self._research_problem

    @property
    def benchmark_folder_name(self):
        return self._benchmark_folder_name

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def work_dir(self):
        return self._work_dir
    
    @property
    def read_only_files(self):
        return self._read_only_files

    @property
    def action_infos(self):
        return self._action_infos
    
    @property
    def args(self):
        return self._args

    @property
    def static_kwargs_for_tools(self):
        return self._static_kwargs_for_tools
    
    @property
    def trace(self):
        return copy.deepcopy(self._trace)

    @property
    def start_time(self):
        return self._start_time
    
    ############################## internal functions ########################################
    
    def _setup_log_dir(self):
        # set up log dir
        if os.path.exists(self.args.log_dir):
            print("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(self.log_dir)

        if os.path.exists(os.path.join(self.log_dir, "tool_logs")):
            print("tools_log_dir {} already exists".format(os.path.join(self.log_dir, "tool_logs")))
            # raise ValueError("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(os.path.join(self.log_dir, "tool_logs"))

        if os.path.exists(os.path.join(self.log_dir, "traces")):
            print("tools_log_dir {} already exists".format(os.path.join(self.log_dir, "traces")))
            # raise ValueError("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(os.path.join(self.log_dir, "traces"))

    def _initialize_task_env(self):

        work_dir = self.work_dir

        # remove the workspace folder if it exists
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        benchmark_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "benchmarks", self.benchmark_folder_name)

        # prepare if there is a prepare.py and it has not been prepared
        # prepare_task(benchmark_dir, self.args.python)

        # copy the benchmarks folder to work_dir
        if os.path.exists(os.path.join(benchmark_dir, "env" )):
            shutil.copytree(os.path.join(benchmark_dir, "env"), work_dir, symlinks=True)

        # find all read only files
        if os.path.exists(os.path.join(benchmark_dir, "scripts", "read_only_files.txt")):
            ignore_files = open(os.path.join(benchmark_dir, "scripts", "read_only_files.txt"), "r").read().split("\n")
            for path, subdirs, files in os.walk(os.path.join(work_dir)):

                relpath = os.path.relpath(path, work_dir)
                # filter out the files that are read only
                # "we want 'main.py' as filename not './main.py', as we specify 'main.py' in read_only_files.txt"
                filenames = [os.path.join(relpath, filename) if relpath != "." else filename for filename in files]
                for ignore in ignore_files:
                    ignore_filenames = [n for n in filenames if fnmatch.fnmatch(n, ignore)]
                    self.read_only_files.extend(ignore_filenames)


        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(work_dir, "backup")):
            shutil.rmtree(os.path.join(work_dir, "backup"))
        os.mkdir(os.path.join(work_dir, "backup"))

        if self.args.resume:
            shutil.rmtree(work_dir)
            resume_dir = os.path.join(self.args.resume, "env_log", "traces" , f"step_{self.args.resume_step}_files")
            print("Restoring workspace ing from {}".format(resume_dir))
            shutil.copytree(resume_dir, work_dir, symlinks=True)
            if not os.path.exists(os.path.join(work_dir, "backup")):
                os.mkdir(os.path.join(work_dir, "backup"))


    def _initialize_interactive_env(self):
        # set up read only files
        can_modify_files = input("What existing files can Research Assistant modify (relative paths separated by comma)? default is nothing: ").split(",")
        size = 0
        self._read_only_files = []
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):
            relpath = os.path.relpath(path, self.work_dir)
            # filter out the files that are read only
            filenames = [os.path.join(relpath, filename) for filename in files]
            for not_ignore in can_modify_files:
                ignore_filenames = [n for n in filenames if not fnmatch.fnmatch(n, not_ignore)]
                self.read_only_files.extend(ignore_filenames)
            for f in files:
                size += os.path.getsize(os.path.join(path, f))
                
        # try save this task to a benchmark folder
        os.makedirs(os.path.join(self.log_dir, self.benchmark_folder_name), exist_ok=True)
        if size / 1e6 < 10:
            # save if the size is smaller than 10MB
            shutil.copytree(self.work_dir, os.path.join(self.log_dir, self.benchmark_folder_name, "env"))
        os.makedirs(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts"), exist_ok=True)
        with open(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts", "research_problem.txt"), "w") as f:
            f.write(self.research_problem)
        with open(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts", "read_only_files.txt"), "w") as f:
            f.write("\n".join(self.read_only_files))

        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(self.work_dir, "backup")):
            shutil.rmtree(os.path.join(self.work_dir, "backup"))
        os.mkdir(os.path.join(self.work_dir, "backup"))


    def _initialize_trace(self):
        if self.args.resume:
            print("Restoring trace from {}".format(self.args.resume))
            prev_trace = from_dict(data_class=Trace, data=json.load(open(os.path.join(self.args.resume, "env_log","trace.json"), "r")))
            print("Resetting trace to step {}".format(self.args.resume_step))
            steps = prev_trace.steps[:self.args.resume_step+1]
            t = steps[-1].timestamp
            low_level_steps = [s for s in prev_trace.low_level_steps if s.timestamp < t]
            trace = Trace(
                steps=steps,
                low_level_steps=low_level_steps,
                action_infos=self.action_infos,
                task_description=self.research_problem,
            )
        else:   
            trace = Trace(
            steps=[],
            low_level_steps=[],
            action_infos=self.action_infos,
            task_description=self.research_problem,
        )
        return trace
    
    def __enter__(self):
        # set time out
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.args.max_time)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):  
        # save error message
        active = active_children()
        print(f'Active Children: {len(active)}')
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        print(f'Active Children: {len(active)}')
            
        if traceback is not None:
            print("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(''.join(format_exception(exc_type, exc_value, traceback)))
        with open(os.path.join(self.log_dir, "overall_time.txt"), "w") as writer:
            time_info = {"total_time" : time.time() - self.start_time, "start_time" : self.start_time}
            json.dump(time_info, writer, indent=2)
            
    ################################# public functions ########################################

    def is_final(self):
        """Check if the task has reached a final state, either by reaching the maximum steps or time, or because the agent has submitted a final answer. """
        
        curr_step = len(self.trace.steps)
        api_cost_file = os.path.join(self.log_dir, "api_cost.json")
        if os.path.exists(api_cost_file):
            with open(api_cost_file, 'r') as reader:
                curr_api_cost = json.load(reader)["total_cost"] 
        else:
            curr_api_cost = 0

        if self._submit_valid_answer:
            self.finish_reason = "agent has submitted a final answer"
            return True
        elif curr_step >= self.args.max_steps:
            self.finish_reason = "max_steps are reached"
            return True
        elif time.time() - self.start_time > self.args.max_time:
            self.finish_reason = "max_time is reached"
            return True
        elif curr_api_cost > self.args.max_api_cost:
            self.finish_reason = "max_api_cost is reached"
            return True
        else:
            return False

    def execute(self, action):
        """Execute an action and return the observation."""
        
        trace = self._trace

        curr_step = len(trace.steps)
        os.environ["CURR_STEP"] = str(curr_step)
        action_name = action.name
        action_input = action.args

        if self.is_final():
            observation = f"The environment has shut down because {self.finish_reason}. Please submit your final answer."

        elif action_name not in list(self.action_infos.keys()):
            actions = ", ".join(self.action_infos.keys())
            observation = f"Invalid action: {action_name}. Action did not execute. Please use one of the following actions:\n{actions}"

        else:
            # execute the action and get the observation
            log_file = os.path.join(os.path.join(self.log_dir, "tool_logs") , f"step_{curr_step}_tool_log.log")
            usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action_name].usage.items()])
            usage = f"""{{
            {usage}
}}"""
            invalid_action_error = f"The action input for {action_name} needs to be a valid json with proper entries. You may have missed the comma between entries. Please use the correct format and try again:\n{usage}"

            if isinstance(action_input, dict):
                try:
                    observation = self.action_infos[action_name].function(**action_input, log_file=log_file, trace=trace, **self.static_kwargs_for_tools)
                except TooLongPromptError:
                    observation="EnvError: too long input for the tool"
                except LLMError as e:
                    observation = "LLMError: " + e.message
                except EnvException as e:
                    observation = "EnvError: " + e.message
                except TypeError as e:
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    print(action_input, file=sys.stderr)
                    observation = "EnvError: " + invalid_action_error
                except TimeoutException as e:
                    raise e
                except Exception as e:
                    # should not happen
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    if "Connection aborted." in str(e):
                        raise Exception("Connection aborted for crfm")
                    observation = f"EnvError: Error executing {action_name}."
                else:
                    if action_name == "Final Answer":
                        self._submit_valid_answer = True
                        observation = "end"
            else:
                observation = invalid_action_error


        step_time = time.time()

        trace.steps.append(Step(action, observation, step_time))

        self.save(curr_step, action_name)
        return observation

    def save(self, curr_step, action_name=""):
        """ Save the trace and snapshot of the workspace folder """     
        with open(os.path.join(self.log_dir, f"trace.json"), "w") as f:
            json.dump(self.trace, f, indent=4, cls=EnhancedJSONEncoder)

        ##### save a snapshot of the current step
        save_folder = os.path.join(self.log_dir, f"traces/step_{curr_step}_files")
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)


        # save files in the folder that are not read-only and less than 50MB
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):

            relpath = os.path.relpath(path, self.work_dir)
            dest = os.path.join(save_folder, relpath)

            for file_name in files:
                file_path = os.path.join(relpath, file_name) if relpath != "." else file_name
                abs_file_path = os.path.join(self.work_dir, file_path)
                
                # Check if the file is not read-only and is less than 50MB
                if "pycache" not in file_path and file_path not in self.read_only_files:
                    # in non-code execution step, do not save files bigger than 50 MB; in code execution step, save all files including ckpts
                    if action_name != "Execute Script" and os.path.getsize(abs_file_path) > 50 * 1024 * 1024:
                        continue
                    # Check whether the file to copy is part of self.log_dir
                    if os.path.abspath(abs_file_path).startswith(os.path.abspath(self.log_dir.split("/env_log")[0])):
                        continue
                    
                    if not os.path.exists(dest):
                        os.makedirs(dest)
                    
                    shutil.copyfile(abs_file_path, os.path.join(save_folder, file_path)) 

    ############## for logging convenience ##############

    def get_task_description(self):
        return self.research_problem, self.benchmark_folder_name

    @property
    def low_level_actions(self):
        return list(filter(lambda x: x.is_primitive, self.action_infos.values()))

    @property
    def high_level_actions(self):
        return list(filter(lambda x: not x.is_primitive, self.action_infos.values()))

    def print_action(self, entries):
        return "".join([ k + ": " + v for k,v in  entries.items()])
