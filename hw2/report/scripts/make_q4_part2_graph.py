import os
import re
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(r"q4_2_lr[0-9.]+_bs[0-9]+([_a-z]*)_HalfCheetah-v4")
TAG_TO_LABEL = OrderedDict(
    [
        ("", "Standard"),
        ("_nnbaseline", "NN baseline"),
        ("_rtg", "Reward-to-go"),
        ("_rtg_nnbaseline", "Reward-to-go, NN baseline"),
    ]
)
GRAPH_TITLE = "Training curve for HalfCheetah-v4 using different addons"
COLOURMAP = "Set1"

print("Loading log data...")

# A list of elements `(tag, Eval_AverageReturn)`
experiment_logs = []

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)

    regex_match = EXPERIMENT_NAME_REGEX.match(filename)

    # Only consider directories whose name matches `EXPERIMENT_NAME_REGEX`
    if not regex_match or not os.path.isdir(filepath):
        continue

    tag = regex_match.group(1)

    # Load the logs
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    event_list = event_acc.Scalars("Eval_AverageReturn")
    curve = []
    for i, event in enumerate(event_list):
        assert i == event.step
        curve.append(event.value)
    experiment_logs.append((tag, np.array(curve)))

# Sort the logs according to `TAG_TO_LABEL`
experiment_logs.sort(key=lambda t: list(TAG_TO_LABEL.keys()).index(t[0]))

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, (tag, avg_return) in enumerate(experiment_logs):
    label = TAG_TO_LABEL[tag]
    x_values = np.arange(avg_return.shape[0])
    ax.plot(x_values, avg_return, color=cmap(i), label=label)
    ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)

fig.legend()

plt.show()
