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

        # super().__init__(env)

        # model_config = config.agent.torch_models.Actor

        # self._model = Models[model_config.name](
        #    input_size=self.state_size,
        #    hidden_dims=model_config.hidden_dims,
        #    output_size=self.action_size,
        #    activations=model_config.activations,
        # )

        super(Actor, self).__init__(env)
        self.fa1 = nn.Linear(6, 400)
        self.fa2 = nn.Linear(400, 200)
        self.fa3 = nn.Linear(200, 150)
        self.fa4 = nn.Linear(150, 1)

        self.apply(weights_init_)

        # self.to(DEVICE)

        self.train() if config.agent.training else self.eval()

    def forward(self, state) -> Tensor:
        x = F.relu(self.fa1(state))
        x = F.relu(self.fa2(x))
        x = F.relu(self.fa3(x))
        x = self.fa4(x)

        return torch.tanh(x) * self.action_scale + self.action_bias
        # return torch.tanh(self._model(state)) * self.action_scale + self.action_bias


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

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.00001)

    def predicted(self, state, action):

        output = self.critic(state.unsqueeze(0), action).item()
        return output

    def train(self, *args):

        self.critic_optimizer.zero_grad()

        states, actions, _, rewards, _ = self.replay_buffer.sample()

        targets: Tensor = torch.log10(rewards)

        critic_loss = nn.MSELoss()(self.critic(states, actions), targets)

        critic_loss.backward()
        self.critic_optimizer.step()

        # self.optimize_actor(loss=-self.critic(states, self.actor(states)).mean())
        # print(self.requires_grad_false(self.critic))
        with self.requires_grad_false(self.critic):

            self.actor_optimizer.zero_grad()
            # print(self.actor(states))
            # print(self.critic(states, self.actor(states)))

            actor_loss = nn.MSELoss()(self.critic(states, self.actor(states)), targets)

            actor_loss.backward()
            self.actor_optimizer.step()

            # quit()

    def test(self):

        counter = next(test_counter)

        states, actions, _, rewards, _ = self.replay_buffer.sample()

        targets: Tensor = torch.log10(rewards).view([-1])

        # test_results = util.test_df(
        #    states=states, actions=actions, labels=targets, net=self.critic,
        # )

        # os.makedirs(f"{self.config.misc.csv_path}", mode=0o775, exist_ok=True)

        # test_results.to_csv(
        #    f"{self.config.misc.csv_path}/co3_testing_frame-{counter}.csv"
        # )
        # print(f"Test Pass {counter} Complete")

    def __str__(self):
        return str(self.actor) + str(self.critic)
