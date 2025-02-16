import os

ALL_BASE_RUNTIME = {
        "base-competition" : {
            "dev" : 100,
            "test" : 100,
            },
        "llm-merging" : {
            "dev" : 3338.013678,
            "test" : 2428.072219,
            },
        "backdoor-trigger-recovery" : {
            "dev" : 597.5532175,
            "test" : 498.1251627,
            "debug" : 400,
            },
        "perception_temporal_action_loc" : {
            "dev" : 1238.63,
            "test" : 361.45,
            "debug" : 100,
        },
        "machine_unlearning":{
            "dev": 353.8617296,
            "test": None, # random number, ignore
            "debug": 233,
        },
        # TODO: add the runtime (unit in seconds) of your new tasks here.
        "meta-learning": {
            "dev" : 1631.955898,
            "test" : 7615.067012,
            "debug" : 233,
        },
        "erasing_invisible_watermarks": {
            "beige": {
                "stegastamp": {
                    "dev": 27,     
                    "test": 120    
                },
                "treering": {
                    "dev": 29,     
                    "test": 128    
                }
            },
            "black": {
                "dev": 56,         
                "test": 223       
            }
        },
    }

ALL_BASE_PERFORMANCE = {
        "base-competition" : {
            "dev" : 0.5,
            "test" : 0.5,
            },
        "llm-merging" : {
            # range 0-1
            "dev" : 0.727136371,
            "test" : 0.4933333333,
            },
        "backdoor-trigger-recovery" : {
            # range 0-100
            "dev" : 3.758409347,
            "test" : 9.368725447,
            "debug" : 2,
            },
        "perception_temporal_action_loc" : {
            # range 0-1
            "dev" : 0.2370039379,
            "test" : 0.1263531695,
            "debug" : 0.2
        },
         "machine_unlearning":{
             # range 0-1
            "dev": 0.05389313916,
            "test": 0.06085605833,
            "debug": 233,
        },
        # TODO: add the baseline performance of your new tasks here.
        "meta-learning" : {
            # range 0-1
            "dev" : 0.1821651453,
            "test" : 0.1727912574,
            "debug" : 0.15,
        },
        "erasing_invisible_watermarks": {
            "beige": {
                "stegastamp": {
                    "dev": 0.1700,   # Overall Score
                    "test": 0.1774   # Overall Score
                },
                "treering": {
                    "dev": 0.2494,   # Overall Score
                    "test": 0.2486   # Overall Score
                }
            },
            "black": {
                "dev": 0.2061,     # Overall Score
                "test": 0.1935     # Overall Score
            }
        },
    }



MLR_BENCH_DIR = os.getenv("MLR_BENCH_DIR", "~/MLAgentBench") # absolute path is preferred
