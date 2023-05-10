import os
import re

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(r"(q4_lr([0-9.]+)_bs([0-9]+))_HalfCheetah-v4")
GRAPH_TITLE = "Training curve for HalfCheetah-v4 using a neural network baseline"
COLOURMAP = "tab20c"

print("Loading log data...")

# A list of elements `(lr, batch_size, Eval_AverageReturn)`
experiment_logs = []
lrs = set()
batch_sizes = set()

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)

    regex_match = EXPERIMENT_NAME_REGEX.match(filename)

    # Only consider directories whose name matches `EXPERIMENT_NAME_REGEX`
    if not regex_match or not os.path.isdir(filepath):
        continue

    lr = float(regex_match.group(2))
    batch_size = int(regex_match.group(3))

    lrs.add(lr)
    batch_sizes.add(batch_size)

    # Load the logs
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    event_list = event_acc.Scalars("Eval_AverageReturn")
    curve = []
    for i, event in enumerate(event_list):
        assert i == event.step
        curve.append(event.value)
    experiment_logs.append((lr, batch_size, np.array(curve)))

# Sort the logs lexicographically
experiment_logs.sort(key=lambda t: f"{t[0]:.3f}:{t[1]}")

# Get the learning rates and batch sizes as sorted lists
lrs = sorted(lrs)
batch_sizes = sorted(batch_sizes)

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, (lr, batch_size, avg_return) in enumerate(experiment_logs):
    colour_index = 4 * lrs.index(lr) + batch_sizes.index(batch_size)
    label = f"LR: {lr}, Batch: {batch_size}"
    x_values = np.arange(avg_return.shape[0])
    ax.plot(x_values, avg_return, color=cmap(colour_index), label=label)
    ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)

fig.legend()

plt.show()
