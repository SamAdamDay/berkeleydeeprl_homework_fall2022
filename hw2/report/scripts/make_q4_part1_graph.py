import os
import re

import numpy as np

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(
    r"(q4_search_lr([0-9.]+)_bs([0-9]+))_rtg_nnbaseline_HalfCheetah-v4"
)
PLOT_SMOOTHING = 2
GRAPH_TITLE = "Training curve for HalfCheetah-v4 using a neural network baseline"
if PLOT_SMOOTHING > 0:
    GRAPH_TITLE += f" with Gaussian smoothing sigma={PLOT_SMOOTHING}"
COLOURMAP = "tab20c"
LIGHT_COLOUR_ALPHA = 0.2

print("Loading log data...")

# A list of elements `(lr, batch_size, avg_return, std_return)`
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
    avg_events = event_acc.Scalars("Eval_AverageReturn")
    std_events = event_acc.Scalars("Eval_StdReturn")
    avg_return = np.zeros(len(avg_events))
    std_return = np.zeros(len(std_events))
    for i, (avg_event, std_event) in enumerate(zip(avg_events, std_events)):
        assert i == avg_event.step == std_event.step
        avg_return[i] = avg_event.value
        std_return[i] = std_event.value
    experiment_logs.append((lr, batch_size, avg_return, std_return))

print("Processing data...")

# Sort the logs lexicographically
experiment_logs.sort(key=lambda t: f"{t[0]:.3f}:{t[1]}")

# Get the learning rates and batch sizes as sorted lists
lrs = sorted(lrs)
batch_sizes = sorted(batch_sizes)

if PLOT_SMOOTHING > 0:
    for i, (lr, batch_size, avg_return, std_return) in enumerate(experiment_logs):
        avg_return = gaussian_filter1d(avg_return, sigma=PLOT_SMOOTHING)
        std_return = gaussian_filter1d(std_return, sigma=PLOT_SMOOTHING)
        experiment_logs[i] = (lr, batch_size, avg_return, std_return)

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for lr, batch_size, avg_return, std_return in experiment_logs:
    colour_index = 4 * lrs.index(lr) + batch_sizes.index(batch_size)
    label = f"LR: {lr}, Batch: {batch_size}"
    x_values = np.arange(avg_return.shape[0])
    ax.plot(x_values, avg_return, color=cmap(colour_index), label=label)
    ax.fill_between(
        x_values,
        y1=avg_return - std_return,
        y2=avg_return + std_return,
        color=cmap(colour_index, alpha=LIGHT_COLOUR_ALPHA),
    )
    ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)

fig.legend()

plt.show()
