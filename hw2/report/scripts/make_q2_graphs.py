import os
import re

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tqdm import tqdm

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")

EXPERIMENT_REGEX = re.compile("(q2_lr([0-9.]+)_bs([0-9]+)_s([0-9]+))_InvertedPendulum")
OPTIMIAL_VALUE = 1000
OPTIMIUM_PROP_THRESHOLD = 0.1
COLOURMAP = "viridis"
GRAPH_TITLE = (
    "Average proportion of training curves which where the eval return "
    "is optimal, for different learning rates and batch sizes"
)

print("Loading log data...")

# A dict whose keys are `(lr, batch_size)` and whose values are dicts with
# keys `seed` and values 'Eval_AverageReturn' curves
experiment_logs = {}

for filename in tqdm(os.listdir(DATA_DIR)):
    filepath = os.path.join(DATA_DIR, filename)

    regex_match = EXPERIMENT_REGEX.match(filename)

    # Only consider directories whose name matches `EXPERIMENT_REGEX`
    if regex_match is None or not os.path.isdir(filepath):
        continue

    lr = float(regex_match.group(2))
    batch_size = int(regex_match.group(3))
    seed = int(regex_match.group(4))

    # Load the logs
    if (lr, batch_size) not in experiment_logs:
        experiment_logs[(lr, batch_size)] = {}
    elif seed in experiment_logs[(lr, batch_size)]:
        continue
    event_acc = EventAccumulator(filepath)
    event_acc.Reload()
    event_list = event_acc.Scalars("Eval_AverageReturn")
    curve = []
    for i, event in enumerate(event_list):
        assert i == event.step
        curve.append(event.value)
    experiment_logs[(lr, batch_size)][seed] = np.array(curve)

print("Computing statistics...")

# Compute the average propotion of the curve which is at the optimial value.
# Make two sets of arrays, one where this is below `OPTIMIUM_PROP_THRESHOLD`
# and one where this is above. Record in these the learning rates, batch sizes
# and average propotions
lrs_below, batch_sizes_below = [], []
lrs_above, batch_sizes_above, optimium_props_above = [], [], []

for (lr, batch_size), curves_dict in experiment_logs.items():
    # Compute the average optimum proportion across all seeds
    total_optimium_prop = 0.0
    for i, (seed, curve) in enumerate(curves_dict.items()):
        total_optimium_prop += float(np.count_nonzero(curve == OPTIMIAL_VALUE)) / 100
    average_optimium_prop = total_optimium_prop / len(curves_dict)

    # Add to the corresponding lists
    if average_optimium_prop < OPTIMIUM_PROP_THRESHOLD:
        lrs_below.append(lr)
        batch_sizes_below.append(batch_size)
    else:
        lrs_above.append(lr)
        batch_sizes_above.append(batch_size)
        optimium_props_above.append(average_optimium_prop)

lrs_below = np.array(lrs_below)
batch_sizes_below = np.array(batch_sizes_below)
lrs_above = np.array(lrs_above)
batch_sizes_above = np.array(batch_sizes_above)
optimium_props_above = np.array(optimium_props_above)

print("Plotting graph...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Normalisation for colour-mapping the optimium proportions
norm = mpl.colors.Normalize(
    vmin=np.min(optimium_props_above), vmax=np.max(optimium_props_above)
)

# A scatter plot for those above and those below the threshold
ax.scatter(lrs_below, batch_sizes_below, facecolors="none", edgecolors="k")
ax.scatter(
    lrs_above, batch_sizes_above, c=optimium_props_above, cmap=COLOURMAP, norm=norm
)

# Make the colour bar
sm = plt.cm.ScalarMappable(cmap=COLOURMAP, norm=norm)
fig.colorbar(sm, label="Average proportion of training curve at optimial return")

ax.set(xlabel="Learning rate", ylabel="Batch size", xscale="log")
ax.set_title(GRAPH_TITLE, loc="center", wrap=True)

plt.show()
