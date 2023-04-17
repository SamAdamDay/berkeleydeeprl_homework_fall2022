import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENT_NAME_START = "q3_b40000_r0.005_LunarLanderContinuous-v2"
GRAPH_TITLE = (
    "Training curve for LunarLanderContinuous-v2 using a neural network baseline"
)
GRAPH_COLOR = mpl.colormaps["Set1"](1)
GRAPH_LIGHT_COLOR = mpl.colormaps["Set1"](1, alpha=0.3)

# Select the first folder which starts with `EXPERIMENT_NAME_START`
for filename in os.listdir(DATA_DIR):
    filepath = os.path.join(DATA_DIR, filename)
    if filename.startswith(EXPERIMENT_NAME_START) and os.path.isdir(filepath):
        break

# Load the data
event_acc = EventAccumulator(filepath)
event_acc.Reload()
avg_events = event_acc.Scalars("Eval_AverageReturn")
std_events = event_acc.Scalars("Eval_StdReturn")
avg_numpy = np.zeros(len(avg_events))
std_numpy = np.zeros(len(std_events))
for i, (avg_event, std_event) in enumerate(zip(avg_events, std_events)):
    assert i == avg_event.step == std_event.step
    avg_numpy[i] = avg_event.value
    std_numpy[i] = std_event.value

# Plot the data
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
x_values = np.arange(avg_numpy.shape[0])
ax.fill_between(
    x_values,
    y1=avg_numpy - std_numpy,
    y2=avg_numpy + std_numpy,
    color=GRAPH_LIGHT_COLOR,
)
ax.plot(
    x_values,
    avg_numpy,
    color=GRAPH_COLOR,
)
ax.set(xlabel="Iteration", ylabel="Return", title=GRAPH_TITLE)
plt.show()