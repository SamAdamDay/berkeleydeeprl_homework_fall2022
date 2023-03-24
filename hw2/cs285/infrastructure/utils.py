import numpy as np
from numpy.typing import NDArray
import time
import copy
from typing import Tuple

from gym import Env
from numpy.typing import ArrayLike

from cs285.policies.base_policy import BasePolicy

############################################
############################################


def calculate_mean_prediction_error(
    env: Env, action_sequence: list, models: list, data_statistics
) -> tuple[NDArray, dict, NDArray]:
    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)["observation"]

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env: Env, actions: list) -> dict:
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    return np.mean((a - b) ** 2)


############################################
############################################


def sample_trajectory(
    env: Env, policy: BasePolicy, max_path_length: int, render: bool = False
) -> dict:
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if hasattr(env, "sim"):
                image_obs.append(
                    env.sim.render(camera_name="track", height=500, width=500)[::-1]
                )
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # End the rollout if the rollout ended
        rollout_done = int(done or steps >= max_path_length)
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(
    env: Env,
    policy: BasePolicy,
    min_timesteps_per_batch: int,
    max_path_length: int,
    render: bool = False,
) -> Tuple["list[dict]", int]:
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(
    env: Env,
    policy: BasePolicy,
    ntraj: int,
    max_path_length: int,
    render: bool = False,
) -> "list[dict]":
    """
    Collect `ntraj` rollouts.
    """
    paths = []

    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

    return paths


############################################
############################################


def Path(
    obs: list,
    image_obs: list,
    acs: list,
    rewards: list,
    next_obs: list,
    terminals: list,
) -> dict:
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def convert_listofrollouts(
    paths: "list[dict]",
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, "list[NDArray]"]:
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


############################################
############################################


def get_pathlength(path: dict) -> int:
    return len(path["reward"])


def normalize(
    data: ArrayLike, mean: ArrayLike, std: ArrayLike, eps: float = 1e-8
) -> ArrayLike:
    return (data - mean) / (std + eps)


def unnormalize(data: ArrayLike, mean: ArrayLike, std: ArrayLike) -> ArrayLike:
    return data * std + mean


def add_noise(data_inp: NDArray, noiseToSignal: float = 0.01) -> NDArray:
    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(
            data[:, j]
            + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],))
        )

    return data
