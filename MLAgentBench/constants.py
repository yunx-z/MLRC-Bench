import os

# TODO: automatically read from base_exp idea_evals.json

ALL_BASE_RUNTIME = {
        "base-competition" : {
            "dev" : 100,
            "test" : 100,
            },
        "llm-merging" : {
            "dev" : 3338.01,
            "test" : 2428.07,
            },
        "backdoor-trigger-recovery" : {
            "dev" : 597.55,
            "test" : 498.13,
            "debug" : 400,
            },
        "perception_temporal_action_loc" : {
            "dev" : 1025.33,
            "test" : 313.69,
        },
        # TODO: add the runtime (unit in seconds) of your new tasks here.
    }

ALL_BASE_PERFORMANCE = {
        "base-competition" : {
            "dev" : 0.5,
            "test" : 0.5,
            },
        "llm-merging" : {
            # range 0-1
            "dev" : 0.727,
            "test" : 0.493,
            },
        "backdoor-trigger-recovery" : {
            # range 0-100
            "dev" : 3.758,
            "test" : 9.369,
            "debug" : 2,
            },
        "perception_temporal_action_loc" : {
            # range 0-1
            "dev" : 0.2359,
            "test" : 0.1234,
        },
        # TODO: add the baseline performance of your new tasks here.
    }



MLR_BENCH_DIR = os.getenv("MLR_BENCH_DIR", "~/MLAgentBench") # absolute path is preferred
