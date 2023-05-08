import gym
import numpy as np

import torch
from torch.utils.data import IterableDataset
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

#from torch.utils.tensorboard import SummaryWriter

__cnn_default_kwargs__ = {
    "kernel_sizes": [7, 3, 3, 3],
    "filters": [64, 128, 512, 256],
    "dropout": [0.25, 0.4, 0.4, 0.3],
    "pooling": "GAP",
    "hidden_fc_units": [],
    "fc_dropout": [0.2],
    "out_bias": True,
    "activation": "relu",
    "padding": "same",
    "padding_mode": "replicate"
}

__mlp_default_kwargs__ = {
    "layers": [256, 128, 128],
    "dropout": [0.4, 0.25, 0.25],
    "out_bias": True,
    "activation": "relu"
}

__activation_dict__ = {
    "relu": torch.nn.ReLU,
    "ReLU": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh
}


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: Union[tuple, list, np.ndarray]):
        self.capacity = capacity
        if isinstance(observation_shape, torch.Tensor):
            observation_shape = observation_shape.int()
        self.observation_shape = torch.Size(observation_shape)

        self.current_size = 0
        self.at_full_capacity = False
        self.empty = True

        self.states = torch.Tensor()
        self.actions = torch.Tensor()
        self.rewards = torch.Tensor()
        self.next_states = torch.Tensor()

    def __len__(self):
        return self.current_size

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]

    def __iter__(self):
        return zip(self.states, self.actions, self.rewards, self.next_states)

    def clear(self):
        self.current_size = 0
        self.at_full_capacity = False

        self.states = torch.Tensor()
        self.actions = torch.Tensor()
        self.rewards = torch.Tensor()
        self.next_states = torch.Tensor()

    def add(
        self,
        states: torch.Tensor,
        actions: Union[int, torch.Tensor],
        rewards: Union[float, torch.Tensor],
        next_states: torch.Tensor
    ):
        """
        Adds a single or a batch of experiences to the replay buffer.
        An experience is characterized as a 4-uple:
            - State at step t
            - Action taken at step t
            - Reward obtained at step t
            - State at step t+1
        :param states: original observation(s) tensor
        :param actions: taken action(s) tensor
        :param rewards: obtained reward(s) tensor
        :param next_states: following observation(s) tensor
        :return: None
        """
        # Assertions
        assert all(isinstance(param, torch.Tensor) for param in (states, actions, rewards, next_states)), \
            "Argument should be torch Tensors"
        if actions.ndim != 0:
            assert actions.ndim == 1, "Input action should be a scalar or a 1d-tensor"
            # Check same number n of experiences
            n = states.size(0)
            if any(param.size(0) != n for param in (actions, rewards, next_states)):
                err_str = " and ".join(
                    [f"({value.size(0)}) {key}" for key, value in
                     {"actions": actions, "rewards": rewards, "next_states": next_states}.items()
                     if value.size(0) != n]
                )
                raise AssertionError(
                    f"If adding a batch of experiences, make sure that all parameters have length equal to the number \
                    of experiences. There were ({n}) states provided, but {err_str}.")

            assert states.size()[1:] == self.observation_shape and next_states.size()[1:] == self.observation_shape, \
                f"Observation shape mismatch, expected {tuple(self.observation_shape)}, \
                got {tuple(states.size()[1:])} and {tuple(next_states.size()[1:])}, \
                for states and next_states respectively"
        else:
            n = 1
            assert states.size() == self.observation_shape and next_states.size() == self.observation_shape

        n_remove = 0
        if self.at_full_capacity:
            n_remove = n
        elif self.capacity < (self.current_size + n):
            n_remove = (self.current_size + n) - self.capacity

        self.states = torch.cat((
            self.states[n_remove:], states
        ), dim=0)
        self.actions = torch.cat((
            self.actions[n_remove:], actions
        ), dim=0)
        self.rewards = torch.cat((
            self.rewards[n_remove:], rewards
        ), dim=0)
        self.next_states = torch.cat((
            self.next_states[n_remove:], next_states
        ), dim=0)

        self.current_size += (n - n_remove)

    def sample(self, size):
        assert size <= self.current_size, \
            f"You tried sampling {size} experiences from a replay buffer with {self.current_size} stored experiences"
        indices = np.random.permutation(self.current_size)[:size]
        return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices]


class DQN:
    """
    This class is a solving instance of the Deep Q Learning algorithm to be applied to an environment
    :param policy: which type of policy should be used 'CnnPolicy' or 'MlpPolicy', can also be a nn.Module
        compatible with the environment's observation shape
    :param env: the environment on which to apply the Deep Q Learning
    :param learning_rate: the learning rate for the gradient descent steps to update the DQN weights
    :param buffer_size: the size of the replay buffer to store past experiences in
    :param learning_starts: timestep to start the training, before which only experience sampling occurs
    :param batch_size: size of the mini-batches of experiences to train the network with
    :param gamma: the discounting rate
    :param train_freq: number of time steps between each gradient descent step update of the DQN
    :param gradient_steps: number of gradient steps to do at each update of the DQN, defaults to 1

    :param target_update_interval: number of time steps between each update of the target value network
    :param exploration_fraction: the fraction of training where the exploration factor (epsilon) will decrease
        after doing `exploration_fraction` of the training, epsilon remains equal to its specified final value
    :param exploration_initial_eps: initial value of the exploration factor (epsilon), defaults to 1.0
    :param exploration_final_eps: final value of the exploration factor (epsilon)
    :param max_grad_norm: the maximum norm of the gradient to update the network weights with, if higher, the gradient
        will be normalized to have norm equal to this value. This is to overcome issues of exploding gradients

    :param stats_window_size:
    :param tensorboard_log: tensorboard_log instance to track the metrics into
    :param policy_kwargs:
    :param verbose:
    :param seed:
    :param device: device to use, 'cuda' or 'cpu', defaults to 'auto' which will select 'cuda' if available, else 'cpu'
    :param _init_setup_model:

    :param replay_buffer_class: IDK
    :param replay_buffer_kwargs: IDK
    :param optimize_memory_usage: IDK
    """
    def __init__(
            self,
            policy: str,
            env: gym.Env,
            learning_starts: int = 50000,
            batch_size: int = 32,
            gamma: float = 0.99,
            train_freq: int = 4,
            gradient_steps: int = 1,
            target_update_interval: int = 10000,
            replay_buffer_size: int = 1_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            exploration_decay_type: str = "exponential",
            max_grad_norm=10,
            device='auto',
            seed=None,
            optimizer=None,
            optimizer_kwargs=None,
            loss=None,
            loss_kwargs=None,

            # Might add them or not
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=None,
            verbose=0,
    ):
        # --- Register environment ---
        self._register_environment(env)

        # --- Check policy ---
        self._check_policy(policy, policy_kwargs)

        # --- Register device ---
        self._register_device(device)

        # --- Initialize model ---
        self._initialize_q_nets()

        # --- Initialize optimizer ---
        self._initialize_optimizer()

        # --- General RL hyperparameters ---
        self.gamma = gamma  # discount factor

        # --- DQN-specific hyperparameters ---
        self.replay_buffer = ReplayBuffer(replay_buffer_size, observation_shape=self.obs_shape)
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval  # C in the pseudocode

        # --- Network update parameters ---
        self.gradient_steps = gradient_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # --- Exploration factor, epsilon ---
        self.initial_eps = exploration_initial_eps
        self.final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.eps_schedule_type = exploration_decay_type

        # --- Intialize tensorboard logger ---
        self._initialize_tensorboard()

    def _check_policy(self, policy, policy_kwargs):
        # ADD SUPPORT FOR CUSTOM CNN / MLP POLICIES LATER
        if policy_kwargs:
            raise NotImplementedError("Support for custom CNN and MLP policies is not supported yet")

        assert not (policy and policy_kwargs), "If policy module is provided, policy_kwargs must be None"
        assert policy in ("CnnPolicy", "MlpPolicy") or isinstance(policy, nn.Module), \
            "Policy must be 'CnnPolicy', 'MlpPolicy' or an instance of nn.Module (a PyTorch neural network)"
        if isinstance(policy, nn.Module):
            try:
                obs_sample = torch.zeros_like(torch.tensor(self.obs_shape))
                policy.eval()
                with torch.no_grad():
                    output = policy(obs_sample)
            except (RuntimeError, Exception) as err:
                raise err.with_traceback(err.__traceback__)

            assert output.ndim == 1 and output.size(0) == self.n_actions, \
                "The output of the given policy has to be 1-d, with length equal to the number of possible actions"

        else:
            default_kwargs = __cnn_default_kwargs__ if policy == "CnnPolicy" else __mlp_default_kwargs__
            if policy_kwargs is not None:
                # Check that the policy kwargs work
                if any(key not in default_kwargs.keys() for key in policy_kwargs.keys()):
                    keys = set(policy_kwargs.keys()) - set(default_kwargs)
                    raise KeyError(
                        f"Argument{'s' if len(keys>1) else ''}"
                        f"{', '.join(key for key in keys)} not recognized for CnnPolicy keyword arguments")
            else:
                policy_kwargs = default_kwargs

            if policy == "CnnPolicy":
                # Assert that given parameters work and merge with default parameters if needed
                ...
            else:
                # Assert that given parameters work and merge with default parameters if needed
                ...

    def _register_environment(self, env):
        # Check compatibility and so on
        assert isinstance(env, gym.Env), "Environment must be wrapped as a gym.Env object"
        self.env = env
        try:
            obs_space = env.observation_space
            act_space = env.action_space
            try:
                self.obs_shape = obs_space.shape
                self.n_actions = act_space.n
            except AttributeError as err:
                print("Observation space and action spaces must have attributes shape and n, respectively.")
                raise err.with_traceback(err.__traceback__)
        except AttributeError as err:
            print("Environment must have its action space and observation space accessible as attributes")
            raise err.with_traceback(err.__traceback__)

    def _register_device(self, device):
        assert device in ("auto", "cpu", "cuda") or isinstance(device, torch.device), \
            "Device must be 'cpu', 'cuda', 'auto' or a torch.device instance"
        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _get_eps_scheduler(self, n_training_steps):
        """
        Returns the epsilon scheduling action to be used during training. Called at the beginning of training, when
        the number of total training steps is specified.
        :param n_training_steps: number of desired training steps, the exploration factor epsilon will be scheduled
            to decrease over a specified fraction of this number of training steps, called at class instantiation
        :return: the epsilon scheduling function (callable fn with signature (timestep) :-> (epsilon))
        """
        n_eps_scheduling_steps = round(n_training_steps * self.exploration_fraction)
        if self.eps_schedule_type == "exponential":
            decay_rate = (self.final_eps / self.initial_eps) ** (1 / n_training_steps)
            return lambda k: max(self.initial_eps * (decay_rate ** k), self.final_eps)
        else:
            slope = (self.initial_eps - self.final_eps) / n_eps_scheduling_steps
            return lambda k: max(self.initial_eps - k * slope, self.final_eps)

    def _initialize_q_nets(self, policy, policy_kwargs, seed=None):
        if isinstance(policy, torch.nn.Module):
            q_net = self.policy.__class__(**policy_kwargs)
            q_target_net = self.policy.__class__(**policy_kwargs)
            return q_net, q_target_net

        elif policy == "CnnPolicy":
            kernels = policy_kwargs["kernel_sizes"]
            filters = policy_kwargs["filters"]
            dropout = policy_kwargs["dropout"]
            padding = policy_kwargs["padding"]
            padding_mode = policy_kwargs["padding_mode"]
            activation_module = __activation_dict__[policy_kwargs["activation"]]
            pooling_module = torch.nn.AdaptiveAvgPool2d((1, 1)) \
                if policy_kwargs["pooling"] in ("gap", "GAP") \
                else torch.nn.Flatten()

            layers = []
            prev_depth = self.obs_shape[-1]
            for out, size, drop in zip(filters, kernels, dropout):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=prev_depth,
                        out_channels=out,
                        kernel_size=size,
                        padding=padding,
                        padding_mode=padding_mode,
                        bias=False))
                if drop:
                    layers.append(torch.nn.Dropout(p=drop))
                layers.append(activation_module())
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(pooling_module)

            fc_units = policy_kwargs["hidden_fc_units"]
            fc_dropout = policy_kwargs["fc_dropout"]

            prev_len = out
            for out, drop in zip(fc_units, fc_dropout):
                layers.append(
                    torch.nn.Linear(
                        in_features=prev_len,
                        out_features=out))
                if drop:
                    layers.append(torch.nn.Dropout(p=drop))
                layers.append(activation_module())
                prev_len = out

            layers.append(torch.nn.Flatten())

        else:
            units = policy_kwargs["layers"]
            dropout = policy_kwargs["dropout"]
            activation_module = __activation_dict__[policy_kwargs["activation"]]

            layers = []
            if len(self.obs_shape) != 1:
                layers.append(torch.nn.Flatten())
            prev_len = self.obs_shape.numel()
            for out, drop in zip(units, dropout):
                layers.append(torch.nn.Linear(prev_len, out))
                if drop:
                    layers.append(torch.nn.Dropout(p=drop))
                layers.append(activation_module())
                prev_len = out

        layers.append(torch.nn.Linear(prev_len, out_features=self.n_actions, bias=policy_kwargs["out_bias"]))
        # Probability-interpretable outputs for stochastic decisions
        layers.append(torch.nn.Softmax(dim=1))

        q_net = torch.nn.Sequential(*layers)
        q_target_net = torch.nn.Sequential(*layers)
        return q_net, q_target_net

    def _update_q_target_net(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def set_scheduler_params(self, initial_eps, final_eps, exploration_frac):
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.exploration_fraction = exploration_frac

    def set_DQN_hyperparameters(self, replay_buffer_size, train_freq, learning_starts, target_update_interval):
        self.replay_buffer = ReplayBuffer(replay_buffer_size, observation_shape=self.obs_shape)
        self.train_freq = train_freq
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval  # C in the pseudocode

    def set_discount_rate(self, gamma):
        self.gamma = gamma

    def sample(self, sampling_steps):
        ...

    def train(self, training_steps):
        for trn_step in range(training_steps):
            # Sample from memory buffer
            batch = self.replay_buffer.sample(self.batch_size)
            # Compute Q-Values and Q-Targets
            with torch.no_grad():
                q_values = ...
                q_targets = ...

