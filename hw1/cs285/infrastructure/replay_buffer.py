from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cs285.infrastructure.utils import convert_listofrollouts


class ReplayBuffer(object):
    def __init__(self, max_size: int = 1000000):
        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs: Optional[NDArray] = None
        self.acs: Optional[NDArray] = None
        self.rews: Optional[NDArray] = None
        self.next_obs: Optional[NDArray] = None
        self.terminals: Optional[NDArray] = None

    def __len__(self):
        if self.obs:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths: "list[dict]", concat_rew: Optional[bool] = True):
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        (
            observations,
            actions,
            rewards,
            next_observations,
            terminals,
        ) = convert_listofrollouts(paths, concat_rew)

        if self.obs is None:
            self.obs = observations[-self.max_size :]
            self.acs = actions[-self.max_size :]
            self.rews = rewards[-self.max_size :]
            self.next_obs = next_observations[-self.max_size :]
            self.terminals = terminals[-self.max_size :]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size :]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size :]
            if concat_rew:
                self.rews = np.concatenate([self.rews, rewards])[-self.max_size :]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size :]
            self.next_obs = np.concatenate([self.next_obs, next_observations])[
                -self.max_size :
            ]
            self.terminals = np.concatenate([self.terminals, terminals])[
                -self.max_size :
            ]

    ########################################
    ########################################

    def sample_random_data(
        self, batch_size: int
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        assert (
            self.obs.shape[0]
            == self.acs.shape[0]
            == self.rews.shape[0]
            == self.next_obs.shape[0]
            == self.terminals.shape[0]
        )

        permutor = np.random.permutation(self.obs.shape[0])

        ## Return batch_size number of random entries from each of the 5 component arrays above

        return (
            self.obs[permutor],
            self.acs[permutor],
            self.rews[permutor],
            self.next_obs[permutor],
            self.terminals[permutor],
        )

    def sample_recent_data(
        self, batch_size: int = 1
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )
