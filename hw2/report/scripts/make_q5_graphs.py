import os
import re

import numpy as np

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_REGEX = re.compile(r"q5_lr0.001_bs2000_lambda([0-9.]+)_Hopper-v4")
PLOT_SMOOTHING = 0
GRAPH_TITLE = "Training curve for Hopper-v4 using generalised advantage estimation for different lambdas"
if PLOT_SMOOTHING > 0:
    GRAPH_TITLE += f" with Gaussian smoothing sigma={PLOT_SMOOTHING}"
COLOURMAP = "Set1"
LIGHT_COLOUR_ALPHA = 0.3

print("Loading log data...")

# A list of elements `(gae_lambda, avg_return, std_return)`
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
    avg_events = event_acc.Scalars("Eval_AverageReturn")
    std_events = event_acc.Scalars("Eval_StdReturn")
    avg_return = np.zeros(len(avg_events))
    std_return = np.zeros(len(std_events))
    for i, (avg_event, std_event) in enumerate(zip(avg_events, std_events)):
        assert i == avg_event.step == std_event.step
        avg_return[i] = avg_event.value
        std_return[i] = std_event.value
    experiment_logs.append((gae_lambda, avg_return, std_return))

print("Processing data...")

# Sort the logs according to lambda
experiment_logs.sort(key=lambda t: t[0])

if PLOT_SMOOTHING > 0:
    for i, (gae_lambda, avg_return, std_return) in enumerate(experiment_logs):
        avg_return = gaussian_filter1d(avg_return, sigma=PLOT_SMOOTHING)
        std_return = gaussian_filter1d(std_return, sigma=PLOT_SMOOTHING)
        experiment_logs[i] = (gae_lambda, avg_return, std_return)

print("Plotting graph...")

cmap = mpl.colormaps[COLOURMAP]

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

for i, (gae_lambda, avg_return, std_return) in enumerate(experiment_logs):
    label = f"Lambda {gae_lambda}"
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
