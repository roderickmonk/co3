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
import torch.optim as optim
import util
from agent.continuous_action_space_agent import ContinuousActionSpaceAgent
from sentient_util.exceptions import Anomaly
from constants import DEVICE
from continuous_action_space_model import ContinuousActionSpaceModel
from networks.fully_connected import FullyConnected
from torch import Tensor
from replay_buffer import ReplayBuffer

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(7)
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


try:
    profile  # type: ignore
except NameError:
    profile = lambda x: x

test_counter = it.count(0)


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
            np.random.uniform(low=0.02, high=0.98), dtype=torch.float, device=DEVICE
        )
        # print(f'Actor action = {action}')
        return action.unsqueeze(0)


class Critic(ContinuousActionSpaceModel):
    """"""

    def __init__(self, env, config):

        super(Critic, self).__init__(env)
        self.fc1 = nn.Linear(7, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 1)

        self.apply(weights_init_)

        # for name, param in self.named_parameters():
        #     print(f"{name}:\n{param}")

    def forward(self, state: Tensor, action: Tensor):

        if action.ndim == 2:
            x = torch.cat([state, action], 1)
        elif action.ndim == 1:
            x = torch.cat([state, action.unsqueeze(-1)], 1)
        else:
            raise Anomaly(f"action wrong shape: {action.shape}")

        if x.shape[1] != 7:
            raise ValueError(f"Anomaly: {x.shape=}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent(ContinuousActionSpaceAgent):
    """"""

    def __init__(self, *, config, env):

        super().__init__(config=config, env=env, actor_class=Actor, critic_class=Critic)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic.parameters(), lr=0.00001)

        self.replay_buffer = ReplayBuffer(
            batch_size=config.agent.batch_size,
            buffer_size=config.agent.buffer_size,
        )

    def predicted(self, state, action):

        output = self.critic(state.unsqueeze(0), action).item()
        return output

    def train(self, *args):

        self.optimizer.zero_grad()

        states, actions, _, rewards, _ = self.replay_buffer.sample()

        targets: Tensor = torch.log10(rewards)

        output = self.critic(states, actions)

        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def test(self):

        counter = next(test_counter)

        states, actions, _, rewards, _ = self.replay_buffer.sample()

        targets: Tensor = torch.log10(rewards).view([-1])

        test_results = util.test_df(
            states=states,
            actions=actions,
            labels=targets,
            net=self.critic,
        )

        os.makedirs(f"{self.config.misc.csv_path}", mode=0o775, exist_ok=True)

        test_results.to_csv(
            f"{self.config.misc.csv_path}/co3_testing_frame-{counter}.csv"
        )
        print(f"Test Pass {counter} Complete")

    def __str__(self):
        return str(self.actor) + str(self.critic)
