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
            "dev" : 1238.63,
            "test" : 361.45,
            "debug" : 100,
        },
        "machine_unlearning":{
            "dev": 517.7307,
            "test": 233, # random number, ignore
            "debug": 233,
        },
        # TODO: add the runtime (unit in seconds) of your new tasks here.
        "meta-learning": {
            "val" : 16.12773323059082,
            "test" : 27.63893985748291
        },
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
            "dev" : 0.237,
            "test" : 0.126,
            "debug" : 0.2
        },
         "machine_unlearning":{
             # range 0-1
            "dev": 0.0542,
            "test": 0.0611,
            "debug": 233,
        },
        # TODO: add the baseline performance of your new tasks here.
        "meta-learning": {
            # range 0-1
            "val" : 0.1886189034134081,
            "test" : 0.3657513612634356
        },
    }



MLR_BENCH_DIR = os.getenv("MLR_BENCH_DIR", "~/MLAgentBench") # absolute path is preferred
