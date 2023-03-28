from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from gym import Env

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.infrastructure.utils as utils


class PGAgent(BaseAgent):
    def __init__(self, env: Env, agent_params: dict):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params["gamma"]
        self.standardize_advantages = self.agent_params["standardize_advantages"]
        self.nn_baseline = self.agent_params["nn_baseline"]
        self.reward_to_go = self.agent_params["reward_to_go"]
        self.gae_lambda = self.agent_params["gae_lambda"]

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
            nn_baseline=self.agent_params["nn_baseline"],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(
        self,
        observations: NDArray,
        actions: NDArray,
        rewards_list: "list[NDArray]",
        next_observations: NDArray,
        terminals: NDArray,
    ) -> dict:
        """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.
        """

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_values, terminals
        )

        # Do gradient descent to update the parameters
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list: "list[NDArray]") -> NDArray:
        """
        Monte Carlo estimation of the Q function.
        """

        # Return the estimated qvals based on the given rewards, using
        # either the full trajectory-based estimator or the reward-to-go
        # estimator

        # Note: rewards_list is a list of lists of rewards with the inner list
        # being the list of rewards for a single trajectory.

        # HINT: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these).

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should first be a 2D list where the first dimension corresponds to
        # trajectories and the second corresponds to timesteps,
        # then flattened to a 1D numpy array.

        if not self.reward_to_go:
            q_values_iter = map(self._discounted_return, rewards_list)

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            q_values_iter = map(self._discounted_cumsum, rewards_list)

        q_values = np.concatenate(list(q_values_iter))

        return q_values

    def estimate_advantage(
        self,
        obs: NDArray,
        rews_list: "list[NDArray]",
        q_values: NDArray,
        terminals: NDArray,
    ) -> NDArray:
        """
        Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values

        Parameters
        ----------
        obs : 2D array, a concatenation of all rollouts
        rews_list : list of 1D arrays, grouped by rollout
        q_values : 1D array, a concatenation of all rollouts
        terminals : 1D array, a concatenation of all rollouts
        """

        # Estimate the advantage when nn_baseline is True, by querying the
        # neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## Values were trained with standardized q_values, so ensure
            ## that the predictions have the same mean and standard deviation as
            ## the current batch of q_values
            q_values_mean, q_values_std = np.mean(q_values), np.std(q_values)
            values_mean, values_std = np.mean(values), np.std(values)
            values_normalized = utils.normalize(values, values_mean, values_std)
            values = utils.unnormalize(values_normalized, q_values_mean, q_values_std)

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in range(batch_size - 1, -1, -1):
                    ## Recursively compute advantage estimates starting from
                    ## timestep T.
                    ## HINT: use terminals to handle edge cases. terminals[i]
                    ## is 1 if the state is the last in its trajectory, and
                    ## 0 otherwise.
                    if terminals[i] == 1:
                        advantages[i] = rews[i] - values[i]
                    else:
                        delta = rews[i] + self.gamma * values[i + 1] - values[i]
                        advantages[i] = (
                            delta + self.gamma * self.gae_lambda * advantages[i + 1]
                        )

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## Compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            advantages_mean, advantages_std = np.mean(advantages), np.std(advantages)
            advantages = utils.normalize(advantages, advantages_mean, advantages_std)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths: "list[dict]"):
        self.replay_buffer.add_rollouts(paths)

    def sample(
        self, batch_size: int
    ) -> Tuple[NDArray, NDArray, list[NDArray], NDArray, NDArray]:
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards: NDArray) -> NDArray:
        """
        Helper function

        Input: An array of rewards {r_0, r_1, ..., r_t', ... r_T} from a
        single rollout

        Output: An array full of sum_{t'=0}^T gamma^t' r_{t'} repeated
        """

        # Compute the 2D array of exponentiated gammas
        gamma_exponents = np.arange(rewards.shape[0])
        exponentiated_gamma = np.power(self.gamma, gamma_exponents)

        # Multiply the rewards by the exponentiated gammas and sum them
        discounted_rewards = rewards * exponentiated_gamma
        summed_discounted_rewards = np.sum(discounted_rewards)
        summed_discounted_rewards = summed_discounted_rewards * np.ones_like(rewards)

        return summed_discounted_rewards

    def _discounted_cumsum(self, rewards: NDArray) -> NDArray:
        """
        Helper function which
        -takes array of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        traj_len = rewards.shape[0]

        # Compute the 2D array of exponentiated gammas
        gamma_exp_horiz = np.arange(traj_len).reshape(1, traj_len)
        gamma_exp_vert = np.arange(traj_len).reshape(traj_len, 1)
        gamma_exponents = gamma_exp_horiz - gamma_exp_vert
        exponentiated_gamma = np.power(self.gamma, gamma_exponents)

        # Create a 2D array with rows which look like
        # {0, ..., 0, r_it', ..., r_iT} for 0 ≤ t' ≤ T
        expanded_rewards = np.ones((traj_len, traj_len)) * rewards
        masked_expanded_rewards = np.triu(expanded_rewards)

        # Multiply the masked rewards by the exponentiated gammas and sum
        discounted_expanded_rewards = masked_expanded_rewards * exponentiated_gamma
        summed_discounted_rewards = np.sum(discounted_expanded_rewards, axis=1)

        return summed_discounted_rewards
