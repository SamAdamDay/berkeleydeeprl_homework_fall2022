from typing import Tuple

from numpy.typing import NDArray

from gym import Env

from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env: Env, agent_params: dict):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params["max_replay_buffer_size"])

    def train(
        self,
        ob_batch: NDArray,
        ac_batch: NDArray,
        re_batch: NDArray,
        next_ob_batch: NDArray,
        terminal_batch: NDArray,
    ) -> dict:
        # training a BC agent refers to updating its actor using
        # the given observations and corresponding action labels
        log = self.actor.update(ob_batch, ac_batch)  # HW1: you will modify this
        return log

    def add_to_replay_buffer(self, paths: list[dict]):
        self.replay_buffer.add_rollouts(paths)

    def sample(
        self, batch_size: int
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        return self.replay_buffer.sample_random_data(
            batch_size
        )  # HW1: you will modify this

    def save(self, path: dict):
        return self.actor.save(path)
