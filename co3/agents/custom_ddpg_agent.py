import itertools as it
import logging
import os
from collections import deque, namedtuple
from contextlib import contextmanager
from random import choices

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from replay_buffer import ReplayBuffer
from torch import Tensor
from torch.optim import Adam

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)


try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x


class Actor(ContinuousActionSpaceModel):
    def __init__(self, env, config):

        super().__init__(env)

        model_config = config.agent.torch_models.Actor

        self._model = FullyConnected(
            input_size=self.state_size,
            hidden_dims=model_config.hidden_dims,
            output_size=self.action_size,
            activations=model_config.activations,
        )

        self.to(DEVICE)

        self.train() if config.agent.training else self.eval()

    def forward(self, state) -> Tensor:
        action = torch.as_tensor(
            np.random.random([1]), dtype=torch.float, device=DEVICE
        )
        return action.unsqueeze(0)


class Critic(ContinuousActionSpaceModel):
    def __init__(self, env, config):

        super().__init__(env)

        model_config = config.agent.torch_models.Critic

        self._model = FullyConnected(
            input_size=self.state_size,
            hidden_dims=model_config.hidden_dims,
            activations=model_config.activations,
        )

        self.action_value = nn.Linear(
            env.action_space.shape[0], model_config.hidden_dims[-1]
        )
        self.output_layer = nn.Linear(model_config.hidden_dims[-1], 1)

        self.to(DEVICE)

    def forward(self, state, action) -> Tensor:

        state_value = self._model(state)

        xx = self.action_value(action)

        action_value = F.relu(xx)

        state_action_value = F.relu(torch.add(state_value, action_value))

        return self.output_layer(state_action_value)


class Agent(ContinuousActionSpaceAgent):
    """"""

    def __init__(self, *, config, env):
        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

    def predicted(self, state, action):
        return self.critic(state, action).item()

    @profile  # type:ignore
    def train(self, *args):

        states, actions, next_states, rewards, dones = self.replay_buffer.sample()

        targets: Tensor = rewards

        self.optimize_critic(
            loss=((targets - self.critic(states, actions)) ** 2).mean()
        )

        if next(self.training_counter) % self.target_update_interval == 0:
            super().soft_update(self.critic_target, self.critic)

    def __str__(self):
        return str(self.actor) + str(self.critic)
