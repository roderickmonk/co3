import itertools as it
import logging
import os

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

_log = logging.critical

torch.set_printoptions(precision=10, sci_mode=True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.manual_seed(7)
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


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

        # super().__init__(env)
        # self.fa1 = nn.Linear(6, 400)
        # self.fa2 = nn.Linear(400, 200)
        # self.fa3 = nn.Linear(200, 150)
        # self.fa4 = nn.Linear(150, 1)

        # self.to(DEVICE)

        # self.train() if config.agent.training else self.eval()

    def forward(self, state) -> Tensor:
        # x = F.relu(self.fa1(state))
        # x = F.relu(self.fa2(x))
        # x = F.relu(self.fa3(x))
        # x = self.fa4(x)
        # return x

        return torch.tanh(self._model(state)) * self.action_scale + self.action_bias


class Critic(ContinuousActionSpaceModel):
    """"""

    def __init__(self, env, config):

        super().__init__(env)
        self.fc1 = nn.Linear(7, 250)
        self.fc2 = nn.Linear(250, 200)
        self.fc3 = nn.Linear(200, 150)
        self.fc4 = nn.Linear(150, 1)

        self.apply(weights_init_)

        # self._model = FullyConnected(
        #     input_size=7,
        #     hidden_dims=model_config.hidden_dims,
        #     output_size=1,
        #     activations=model_config.activations,
        # )

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

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.agent.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.agent.critic_lr
        )

    def predicted(self, state, action):

        output = self.critic(state.unsqueeze(0), action).item()
        return output

    def train(self, *args):

        loss = nn.SmoothL1Loss()

        states, actions, _, rewards, _ = self.replay_buffer.sample()

        # targets: Tensor = torch.log10(rewards)
        targets = rewards

        critic_loss = loss(self.critic(states, actions), targets)
        print(f"Critic Loss: {critic_loss}")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.optimize_actor(loss=-self.critic(states, self.actor(states)).mean())
        # print(self.requires_grad_false(self.critic))
        with self.requires_grad_false(self.critic):

            self.actor_optimizer.zero_grad()
            actor_loss = loss(-self.critic(states, self.actor(states)), targets)
            print(f"Actor  Loss:  {actor_loss}")
            actor_loss.backward()
            self.actor_optimizer.step()

        if next(self.training_counter) % self.target_update_interval == 0:
            super().soft_update(self.critic_target, self.critic)
            super().soft_update(self.actor_target, self.actor)

        self.misc_losses(states=states, actions=actions, targets=targets, training=True)

    def test(self):

        counter = next(test_counter)
        states, actions, _, rewards, _ = self.replay_buffer.sample()

        # targets: Tensor = torch.log10(rewards).view([-1])
        targets: Tensor = rewards.view([-1])

        self.misc_losses(
            states=states, actions=actions, targets=targets, training=False
        )

        print(f"Test Pass {counter} Complete")

    def __str__(self):
        return str(self.actor) + str(self.critic)

    def misc_losses(
        self, *, states: Tensor, actions: Tensor, targets: Tensor, training: bool
    ) -> None:

        loss_functions = [
            nn.MSELoss,
            nn.L1Loss,
            nn.HuberLoss,
            nn.SmoothL1Loss,
            nn.PoissonNLLLoss,
            nn.KLDivLoss,
        ]

        with torch.no_grad():
            output = self.critic(states, actions)
            for loss in loss_functions:
                loss_str = str(loss)
                loss_str = loss_str[loss_str.rindex(".") + 1 :]
                loss_str = loss_str[: loss_str.index("'")]
                print(
                    f"{'Training' if training else 'Testing'} {loss_str:>16}: {loss()(output, targets):11.8f}"
                )
