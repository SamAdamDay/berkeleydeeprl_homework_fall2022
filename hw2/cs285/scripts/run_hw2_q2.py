import os
import subprocess
import itertools

SCRIPT_PATH = os.path.realpath(__file__)
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(SCRIPT_PATH), "run_hw2.py")

LR_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
BATCH_SIZE_VALUES = [500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
# SEED_VALUES = [8859, 5543, 1853, 2445]
SEED_VALUES = [5543, 1853, 2445]

parameter_grid = list(itertools.product(LR_VALUES, BATCH_SIZE_VALUES, SEED_VALUES))
for i, (lr, batch_size, seed) in enumerate(parameter_grid):
    exp_name = f"q2_lr{lr}_bs{batch_size}_s{seed}"
    args = [
        "python",
        RUN_SCRIPT_PATH,
        "--env_name",
        "InvertedPendulum-v4",
        "--ep_len",
        "1000",
        "--discount",
        "0.9",
        "-n",
        "100",
        "-l",
        "2",
        "-s",
        "64",
        "-b",
        str(batch_size),
        "-lr",
        str(lr),
        "-rtg",
        "--exp_name",
        exp_name,
    ]
    print()
    print()
    print("=" * 79)
    print(f">> [{i+1}/{len(parameter_grid)}] {exp_name}")
    print(f">> LR: {lr}, BATCH SIZE: {batch_size}, SEED: {seed}")
    print("=" * 79)
    print()
    print()
    subprocess.run(args, check=True)
