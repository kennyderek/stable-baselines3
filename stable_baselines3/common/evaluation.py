import typing
from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common import context_utils

if typing.TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

def evaluate_policy(
    model: "BaseAlgorithm",
    env: Union[gym.Env, VecEnv], # TODO: not actually a VecEnv
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseAlgorithm) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (Optional[float]) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        obs = env.reset()
        if "__contexts__" in obs.keys():
            contexts = obs.pop("__contexts__")
        num_agents = len(obs)
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while True:
            if "__contexts__" in obs:
                obs.pop("__contexts__")
            keylist = list(obs.keys())

            ctx_array = context_utils.dict_obs_to_array(contexts, keylist)
            obs_array = context_utils.dict_obs_to_array(obs, keylist)
            actions, _states = model.predict(obs_array, ctx_array, deterministic=deterministic)
            action_dict = context_utils.array_to_dict_actions(actions, keylist)
            obs, rewards, done, _info = env.step(action_dict)
            episode_reward += sum([rewards[k] for k in rewards])

            if callback is not None:
                callback(locals(), globals())
            episode_length += len(action_dict)
            if render:
                env.render()
            
            if done["__all__"]:
                break


        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length/num_agents)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
