import os
import re
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(r"q5_lr0.001+_bs2000+_lambda([0-9.]+)_HalfCheetah-v4")
GRAPH_TITLE = "Training curve for Hopper-v4 using generalised advantage estimation for different lambdas"
COLOURMAP = "Set1"

print("Loading log data...")

# A list of elements `(gae_lambda, Eval_AverageReturn)`
experiment_logs = []

for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)

    regex_match = EXPERIMENT_NAME_REGEX.match(filename)

    # Only consider directories whose name matches `EXPERIMENT_NAME_REGEX`
    if not regex_match or not os.path.isdir(filepath):
        continue

    gae_lambda = float(regex_match.group(1))

    # Load the logs
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    event_list = event_acc.Scalars("Eval_AverageReturn")
    curve = []
    for i, event in enumerate(event_list):
        assert i == event.step
        curve.append(event.value)
    experiment_logs.append((gae_lambda, np.array(curve)))

# Sort the logs according to lambda
experiment_logs.sort(key=lambda t: t[0])

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, (gae_lambda, avg_return) in enumerate(experiment_logs):
    label = f"Lambda {gae_lambda}"
    x_values = np.arange(avg_return.shape[0])
    ax.plot(x_values, avg_return, color=cmap(i), label=label)

ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)
fig.legend()

plt.show()
