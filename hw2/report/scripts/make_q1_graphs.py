import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT_PATH = os.path.realpath(__file__)
DATA_DIR = os.path.normpath(SCRIPT_PATH + "/../../../data")
EXPERIMENTS = {
    "q1_sb_no_rtg_dsa": {"label": "Basic", "graph": "small_batch", "colour": 0},
    "q1_sb_rtg_dsa": {
        "label": "Reward-to-go",
        "graph": "small_batch",
        "colour": 1,
    },
    "q1_sb_rtg_na": {
        "label": "Reward-to-go, standardised",
        "graph": "small_batch",
        "colour": 2,
    },
    "q1_lb_no_rtg_dsa": {"label": "Basic", "graph": "large_batch", "colour": 0},
    "q1_lb_rtg_dsa": {
        "label": "Reward-to-go",
        "graph": "large_batch",
        "colour": 1,
    },
    "q1_lb_rtg_na": {
        "label": "Reward-to-go, standardised",
        "graph": "large_batch",
        "colour": 2,
    },
}
GRAPHS = {
    "small_batch": {"title": "Eval return for batch size 1000"},
    "large_batch": {"title": "Eval return for batch size 5000"},
}
COLOURMAP = "Set1"
SCALARS_TO_LOAD = ["Eval_AverageReturn", "Eval_StdReturn"]

print("Loading log data...")

experiment_logs = {}

for name in EXPERIMENTS.keys():
    print(f"Experiment {name!r}...")

    # Find all log dirs starting with the experiment name
    matching_dirs = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.startswith(name) and os.path.isdir(filepath):
            matching_dirs.append(filepath)
    if len(matching_dirs) == 0:
        raise RuntimeError(f"Could not find experiment logs for {name!r}")

    # Get the most recent one
    if len(matching_dirs) > 1:
        print("Found multiple experiment logs for {name!r}. Using most recent.")
        matching_dirs = sorted(matching_dirs)
    experiment_dir = matching_dirs[-1]

    # Load the logs
    experiment_logs[name] = {}
    event_acc = EventAccumulator(experiment_dir)
    event_acc.Reload()
    for scalar_name in SCALARS_TO_LOAD:
        event_list = event_acc.Scalars(scalar_name)
        scalar_list = []
        for i, event in enumerate(event_list):
            assert i == event.step
            scalar_list.append(event.value)
        experiment_logs[name][scalar_name] = np.array(scalar_list)

print()
print("Generating graphs...")

cmap = mpl.colormaps[COLOURMAP]

for graph_name, graph_specs in GRAPHS.items():

    print(f"Graph {graph_name!r}...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    for exp_name, exp_specs in EXPERIMENTS.items():

        # Only display the plots for this graph
        if exp_specs["graph"] != graph_name:
            continue
        
        # Compute the colours we'll use
        mean_colour = cmap(exp_specs["colour"])
        error_colour = cmap(exp_specs["colour"], alpha=0.3)

        # Get the data points
        eval_avg = experiment_logs[exp_name]["Eval_AverageReturn"]
        eval_std = experiment_logs[exp_name]["Eval_StdReturn"]
        x_values = np.arange(eval_avg.shape[0])

        # Plot the mean and std of the eval f1 values
        ax.fill_between(
            x_values,
            y1=eval_avg - eval_std,
            y2=eval_avg + eval_std,
            color=error_colour,
        )
        ax.plot(
            x_values,
            eval_avg,
            color=mean_colour,
            label=exp_specs["label"],
        )

    ax.set(xlabel="Iteration", ylabel="Return", title=graph_specs["title"])

    fig.legend()

    plt.show()
