import os
import re
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(r"q4_2_lr0.02+_bs50000+([_a-z]*)_HalfCheetah-v4")
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
LIGHT_COLOUR_ALPHA = 0.3

print("Loading log data...")

# A list of elements `(tag, avg_return, std_return)`
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
    avg_events = event_acc.Scalars("Eval_AverageReturn")
    std_events = event_acc.Scalars("Eval_StdReturn")
    avg_return = np.zeros(len(avg_events))
    std_return = np.zeros(len(std_events))
    for i, (avg_event, std_event) in enumerate(zip(avg_events, std_events)):
        assert i == avg_event.step == std_event.step
        avg_return[i] = avg_event.value
        std_return[i] = std_event.value
    experiment_logs.append((tag, avg_return, std_return))

# Sort the logs according to `TAG_TO_LABEL`
experiment_logs.sort(key=lambda t: list(TAG_TO_LABEL.keys()).index(t[0]))

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, (tag, avg_return, std_return) in enumerate(experiment_logs):
    label = TAG_TO_LABEL[tag]
    x_values = np.arange(avg_return.shape[0])
    ax.fill_between(
        x_values,
        y1=avg_return - std_return,
        y2=avg_return + std_return,
        color=cmap(i, alpha=LIGHT_COLOUR_ALPHA),
    )
    ax.plot(x_values, avg_return, color=cmap(i), label=label)

ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)
fig.legend()

plt.show()
