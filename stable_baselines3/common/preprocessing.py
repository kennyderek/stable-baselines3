from typing import Tuple

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F


def is_image_space(observation_space: spaces.Space, channels_last: bool = True, check_channels: bool = False) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False
    if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space: (spaces.Space)
    :param channels_last: (bool)
    :param check_channels: (bool) Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return: (bool)
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if channels_last:
            n_channels = observation_space.shape[-1]
        else:
            n_channels = observation_space.shape[0]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False

# Preprocessing is run in ActorCritic extract_features() function. ActorCritic is the self.policy of PPO.
def preprocess_obs(obs: th.Tensor, observation_space: spaces.Space, normalize_images: bool = True) -> th.Tensor:
    """
    Preprocess observation to be to a neural network.
    For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
    For discrete observations, it create a one hot vector.

    :param obs: (th.Tensor) Observation
    :param observation_space: (spaces.Space)
    :param normalize_images: (bool) Whether to normalize images or not
        (True by default)
    :return: (th.Tensor)
    """
    if isinstance(observation_space, spaces.Box):
        if is_image_space(observation_space) and normalize_images:
            return obs.float() / 255.0
        return obs.float()

    elif isinstance(observation_space, spaces.Discrete):
        # One hot encoding and convert to float to avoid errors
        return F.one_hot(obs.long(), num_classes=observation_space.n).float()

    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Tensor concatenation of one hot encodings of each Categorical sub-space
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))

    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()

    elif isinstance(observation_space, spaces.Tuple):
        # This would probably not work if the inner obs spaces were images.
        # We need to flatten every inner obs to get the shape (1, x)
        # for idx in range(len(observation_space)):
        #     print(preprocess_obs(obs[idx], observation_space[idx]).shape)
        return th.cat(
            [
                preprocess_obs(obs[idx], observation_space[idx])
                for idx in range(len(observation_space))
            ],
            dim=-1,
        )
    else:
        raise NotImplementedError()


def get_obs_shape(observation_space: spaces.Space) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space: (spaces.Space)
    :return: (Tuple[int, ...])
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return (int(observation_space.n),)
    elif isinstance(observation_space, spaces.Tuple):
        return (sum([get_obs_shape(subobs)[0] for subobs in observation_space]),)
    else:
        raise NotImplementedError()


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    :param observation_space: (spaces.Space)
    :return: (int)
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    if isinstance(observation_space, spaces.Tuple):
        return sum([get_flattened_obs_dim(subspace) for subspace in observation_space])
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space: (spaces.Space)
    :return: (int)
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        return int(action_space.n)
    else:
        raise NotImplementedError()
