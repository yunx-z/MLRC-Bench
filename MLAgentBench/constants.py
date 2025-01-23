ALL_BASE_RUNTIME = {
        "base-competition" : {
            "dev" : 100,
            "test" : 100,
            },
        "llm-merging" : {
            "dev" : 3447.39514565467,
            "test" : 1811.927922,
            },
        "backdoor-trigger-recovery" : {
            "dev" : 481.1659276,
            "test" : 429.746381,
            "debug" : 400,
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
            "dev" : 0.73,
            "test" : 0.49,
            },
        "backdoor-trigger-recovery" : {
            # range 0-100
            "dev" : 8.331147359458377,
            "test" : 12.972998823683664,
            "debug" : 2,
            },
        # TODO: add the baseline performance of your new tasks here.
    }


MLR_BENCH_DIR = "/data2/gdmurphy/MLAgentBench" # absolute path
