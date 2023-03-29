import os
import subprocess
import itertools

SCRIPT_PATH = os.path.realpath(__file__)
RUN_SCRIPT_PATH = os.path.join(os.path.dirname(SCRIPT_PATH), "run_hw2.py")

LR_VALUES = [0.1, 0.01, 0.001, 0.0001, 0.00001]
BATCH_SIZE_VALUES = [10, 50, 100, 500, 1000, 5000]
SEED_VALUES = [8859]

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
