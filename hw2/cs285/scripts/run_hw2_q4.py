import os
import subprocess
import itertools

SCRIPT_PATH = os.path.realpath(__file__)
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(SCRIPT_PATH), "run_hw2.py")

LR_VALUES = [0.005, 0.01, 0.02]
BATCH_SIZE_VALUES = [10000, 30000, 50000]

parameter_grid = list(itertools.product(LR_VALUES, BATCH_SIZE_VALUES))
for i, (lr, batch_size) in enumerate(parameter_grid):
    exp_name = f"q4_lr{lr}_bs{batch_size}"
    args = [
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
    print()
    print()
    print("=" * 79)
    print(f">> [{i+1}/{len(parameter_grid)}] {exp_name}")
    print(f">> LR: {lr}, BATCH SIZE: {batch_size}")
    print("=" * 79)
    print()
    print()
    subprocess.run(args, check=True)
