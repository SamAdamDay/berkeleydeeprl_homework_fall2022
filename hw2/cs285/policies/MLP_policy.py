import abc
import itertools
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: NDArray) -> NDArray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = torch.from_numpy(observation)
        observation = observation.to(device=ptu.device, dtype=torch.float)

        # Build the probability distribution over actions
        distribution = self.forward(observation)

        # Sample an action
        action = distribution.sample()

        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution


#####################################################
#####################################################


class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(
        self,
        observations: NDArray,
        actions: NDArray,
        advantages: NDArray,
        q_values: NDArray,
    ) -> dict:
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        q_values = ptu.from_numpy(q_values)

        # Update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method

        # Build the probability distribution over actions
        action_distribution = self.forward(observations)

        # The log-prob part of the utility
        utility = action_distribution.log_prob(actions)

        if self.nn_baseline:
            # Disable gradient calculation for the baseline while we're
            # updating the policy
            self.baseline.requires_grad_(False)

            # Compute the baseline values
            baseline_values = self.baseline.forward(observations)

            # Multiply the log-prob elementwise
            utility = utility * (q_values - baseline_values)

        else:
            # Multiply the log-prob elementwise
            utility = utility * q_values

        # Sum everything up to get the final utility value
        utility = torch.sum(utility)

        # We do gradient descengt, so we need to take the negative
        loss = -utility

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_log = {
            "Training Loss": ptu.to_numpy(loss),
        }

        if self.nn_baseline:
            ## Update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.

            # Normalise the Q values
            std, mean = torch.std_mean(q_values)
            q_values_normalized = (q_values - mean) / (std + 1e-8)

            # We now want to compute the gradients
            self.baseline.requires_grad_(True)

            # Compute the loss
            baseline_values = self.baseline.forward(observations)
            baseline_loss = F.mse_loss(baseline_values, q_values_normalized)

            # Backpropagation
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

            train_log["Baseline Training Loss"] = ptu.to_numpy(baseline_loss)

        return train_log

    def run_baseline_prediction(self, observations: NDArray) -> NDArray:
        """
        Helper function that converts `observations` to a tensor,
        calls the forward method of the baseline MLP,
        and returns a np array

        Input: `observations`: np.ndarray of size [N, 1]
        Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())
