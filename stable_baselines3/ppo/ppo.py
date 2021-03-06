from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from stable_baselines3.common import logger
from stable_baselines3.common.buffers import TrajectoryBufferSamples, TrajRolloutBuffer
from stable_baselines3.common.decider import Decider
from stable_baselines3.common.on_policy_algorithm import TrajectoryOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


class PPO(TrajectoryOnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: (float or callable) The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: (int) Minibatch size
    :param n_epochs: (int) Number of epoch when optimizing the surrogate loss
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: (float or callable) Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: (float or callable) Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: (float) Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param create_eval_env: (bool) Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param verbose: (int) the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: (int) Seed for the pseudo random generators
    :param device: (str or th.device) Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        n_trajectories: int = 50, # total number of trajectories to collect in each buffer
        batch_size: Optional[int] = 10, # this is now # OF TRAJECTORIES per grad update
        n_epochs: int = 10, # number of times to loop over all of the data
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        rollout_buffer = TrajRolloutBuffer,
        use_context=False,
        context_size=None
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_trajectories=n_trajectories,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            rollout_buffer=rollout_buffer,
            use_context = use_context,
            context_size = context_size
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.use_context = use_context

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _get_decider_batches(self, trajectory_batch: List[TrajectoryBufferSamples]):
        self.decision_length = None
        self.sample_frequency = 1 # TODO: hardcoded
        # observation_shape = trajectory_batch[0].observations.shape[1:] # excludes the length N of the trajectory
        max_trajectory_length = max(t.buffer_size for t in trajectory_batch)
        final_length = (max_trajectory_length if self.decision_length == None else self.decision_length)//self.sample_frequency + 1
        num_trajectories = len(trajectory_batch) # TODO: modify to match decison length
        
        lengths = np.zeros((num_trajectories,))
        trajectories = np.zeros((num_trajectories, final_length,) + (self.policy.features_dim,)) # (B, N, Obs)
        targets = np.zeros((num_trajectories, self.env.context_size))
        for idx, t in enumerate(trajectory_batch):
            with th.no_grad():
                features = self.policy.features_extractor(th.as_tensor(t.observations[0:t.buffer_size:self.sample_frequency], dtype=th.float).to(self.device))
            trajectories[idx][0:((t.buffer_size - 1)//self.sample_frequency)+1] = features
            lengths[idx] = t.buffer_size//self.sample_frequency
            targets[idx] = t.context
        return trajectories, targets, lengths # (B, N, Obs), # (B, Cnxt), # (B,)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs = [], []
        pg_losses, value_losses, decider_losses = [], [], []
        clip_fractions = []

        # train for gradient_steps epochs
        num_steps_in_epoch = 0
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                if self.use_context:
                    trajectory_features, context_targets, lengths = self._get_decider_batches(trajectory_batch)
                    t_packed = pack_padded_sequence(th.tensor(trajectory_features).transpose(0, 1), lengths, enforce_sorted=False) # takes (L, B)
                    t_packed = t_packed.to(self.device)
                    with th.no_grad():
                        context_predictions = self.decider(t_packed) # (batch, context_size)
                    context_mse = np.sum(np.square(context_targets - context_predictions.cpu().detach().numpy()), axis=-1)
                    context_mse_normalized = (context_mse - context_mse.mean()) / (context_mse.std() + 1e-8)
                    context_mse_normalized = context_mse_normalized.reshape((len(trajectory_batch), 1))
                    for idx, t in enumerate(trajectory_batch):
                        t.context_error = np.broadcast_to(context_mse_normalized[idx], shape=(t.buffer_size,))
                
                rollout_data = self.rollout_buffer.format_trajectories(trajectory_batch)
                num_steps_in_epoch += rollout_data.actions.shape[0]

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions) # TODO, return features HERE
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.use_context:
                    advantages -= rollout_data.context_error # TODO, change to log, evaluate whether this is correct?

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())
                
            all_kl_divs.append(np.mean(approx_kl_divs))

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

            if self.use_context:
                # Decider update
                for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                    # here, do the Decider pass on each trajectory, and modify trajectory advantage
                    trajectory_features, context_targets, lengths = self._get_decider_batches(trajectory_batch)
                    t_packed = pack_padded_sequence(th.tensor(trajectory_features).transpose(0, 1), lengths, enforce_sorted=False) # takes (L, B)
                    t_packed = t_packed.to(self.device)
                    context_predictions = self.decider(t_packed) # (batch, context_size)
                    loss = F.mse_loss(context_predictions, th.tensor(context_targets).to(self.device))
                    decider_losses.append(loss.item())
                    self.decider_opt.zero_grad()
                    loss.backward()
                    self.decider_opt.step()

        self._n_updates += self.n_epochs
        # explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        logger.record("train/avg_episode_len", num_steps_in_epoch/(self.n_epochs * self.n_trajectories))
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/context_loss", np.mean(decider_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fraction))
        logger.record("train/loss", loss.item())
        # logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
