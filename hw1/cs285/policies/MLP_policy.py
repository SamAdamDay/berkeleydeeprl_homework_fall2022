import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
from numpy.typing import NDArray
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        n_layers: int,
        size: int,
        discrete: bool = False,
        learning_rate: float = 1e-4,
        training: bool = True,
        nn_baseline: bool = False,
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
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate,
            )

    ##################################

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: NDArray) -> NDArray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = torch.from_numpy(observation)

        # Build the probability distribution over actions
        distribution = self.forward(observation)

        # Sample an action
        action = distribution.sample()

        return action.detatch().cpu().numpy()

    # update/train this policy
    def update(self, observations: NDArray, actions: NDArray, **kwargs) -> dict:
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        if self.discrete:
            logits = self.logits_na(observation)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(observation)
            std = torch.exp(self.logstd)
            return distributions.Normal(loc=mean, scale=std)


#####################################################
#####################################################


class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim: int, ob_dim: int, n_layers: int, size: int, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)

        if self.discrete:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def update(
        self,
        observations: NDArray,
        actions: NDArray,
        adv_n=None,
        acs_labels_na=None,
        qvals=None,
    ) -> dict:
        observations = torch.from_numpy(observations)
        actions = torch.from_numpy(actions)

        observations.to(ptu.device)
        actions.to(ptu.device)

        # Build the probability distribution over actions
        distribution = self.forward(observations)

        # Compute the loss either using cross entropy or MSE on sampled
        # actions
        if self.discrete:
            loss = F.cross_entropy(distribution.probs, actions)
        else:
            sampled_actions = distribution.rsample()
            loss = F.mse_loss(sampled_actions, actions)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            "Training Loss": ptu.to_numpy(loss),
        }
