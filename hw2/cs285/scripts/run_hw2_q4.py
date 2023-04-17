import os
import itertools
import re

from cs285.utils.parallel import ParallelProcesses

SCRIPT_PATH = os.path.realpath(__file__)
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(SCRIPT_PATH), "run_hw2.py")

LR_VALUES = [0.005, 0.01, 0.02]
BATCH_SIZE_VALUES = [10000, 30000, 50000]

TIMEOUT = 0
MAX_CONCURRENT = 3


def get_experiment_progress(output_string: str) -> int | None:
    matches = re.findall(r"\*{10} Iteration ([0-9]+) \*{10}", output_string)
    if len(matches) == 0:
        return None
    return int(matches[-1])


experiment_commands = []
parameter_grid = list(itertools.product(LR_VALUES, BATCH_SIZE_VALUES))
for i, (lr, batch_size) in enumerate(parameter_grid):
    exp_name = f"q4_lr{lr}_bs{batch_size}"
    cmd = [
        "python",
        RUN_SCRIPT_PATH,
        "--env_name",
        "HalfCheetah-v4",
        "--ep_len",
        "150",
        "--discount",
        "0.95",
        "-n",
        "100",
        "-l",
        "2",
        "-s",
        "32",
        "-b",
        str(batch_size),
        "-lr",
        str(lr),
        "-rtg",
        "--nn_baseline",
        "--exp_name",
        exp_name,
    ]
    description = f"LR: {lr}, BS: {batch_size}"
    experiment_commands.append(
        {"name": description, "command": cmd, "max_progress": 100}
    )

process_manager = ParallelProcesses(
    experiment_commands, MAX_CONCURRENT, TIMEOUT, get_experiment_progress
)
process_manager.run()
