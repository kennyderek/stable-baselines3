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
import copy

from torch import nn
import torch.optim as optim

class SampleChooser(nn.Module):

    def __init__(self, obs_size):
        super(SampleChooser, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, input):
        return self.layers(input)
    
    # def predict(self, input):
    #     return nn.Softmax(self.forward(input)

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

        self.sample_chooser = SampleChooser(env.observation_space.sample().flatten().shape[0]).to(self.device)
        self.sample_optimizer = optim.Adam(self.sample_chooser.parameters(), lr=3e-4)

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
        self.sample_frequency = 4 # TODO: hardcoded
        # observation_shape = trajectory_batch[0].observations.shape[1:] # excludes the length N of the trajectory
        max_trajectory_length = max(t.buffer_size for t in trajectory_batch)
        final_length = (max_trajectory_length if self.decision_length == None else self.decision_length)//self.sample_frequency + 1
        num_trajectories = len(trajectory_batch) # TODO: modify to match decison length
        
        lengths = np.zeros((num_trajectories,))
        trajectories = np.zeros((num_trajectories, final_length,) + (int(self.policy.features_dim/2),)) # (B, N, Obs)
        targets = np.zeros((num_trajectories, self.env.context_size))
        for idx, t in enumerate(trajectory_batch):
            with th.no_grad():
                features = self.policy.features_extractor(th.as_tensor(t.observations[0:t.buffer_size:self.sample_frequency], dtype=th.float).to(self.device))
            print("feaaaaut", features.shape)
            trajectories[idx][0:((t.buffer_size - 1)//self.sample_frequency)+1] = features.cpu()
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
        divs, vvals = [], []
        sampler_density, sampler_loss = [], []

        use_decider = True
        if self.use_context and use_decider:
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                trajectory_features, context_targets, lengths = self._get_decider_batches(trajectory_batch)
                t_packed = pack_padded_sequence(th.tensor(trajectory_features).transpose(0, 1), lengths, enforce_sorted=False) # takes (L, B)
                t_packed = t_packed.to(self.device)
                with th.no_grad():
                    context_predictions = self.decider(t_packed) # (batch, context_size)
                context_mse = np.sum(np.square(context_targets - context_predictions.cpu().detach().numpy()), axis=-1)
                # context_mse = np.log10(context_mse + 1) # rescale so rewards look even
                # print(context_mse)
                for idx, t in enumerate(trajectory_batch):
                    blank = np.ones(t.rewards.shape) * context_mse[idx]
                    # blank[-1] = context_mse[idx]
                    # t.context_error = np.ones(t.rewards.shape)# * context_mse[idx]
                    t.context_error = blank
                    # t.rewards = blank
        else:
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                for idx, t in enumerate(trajectory_batch):
                    t.context_error = np.ones(t.rewards.shape)
        
        self.rollout_buffer.compute_returns_and_advantage()

        # calculate average reward

        # train for gradient_steps epochs
        num_steps_in_epoch = 0
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):

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

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, rollout_data.contexts, actions) # TODO, return features HERE
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.use_context and use_decider:
                    advantages -= (rollout_data.context_error - rollout_data.context_error.mean()) / (rollout_data.context_error.std() + 1e-8) # TODO, change to log, evaluate whether this is correct?

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

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # define exploration loss
                expl = False
                if expl:
                    # samples = rollout_data.observations[:50]
                    # # l = []
                    # # for i in range(-3, 4):
                    # #     for j in range(-3, 4):
                    # #         l.append([i, j])
                    # # samples = th.tensor(np.array(l), dtype=th.float).to(self.device)
                    # log_p = {}
                    # for t in trajectory_batch[:5]:
                    #     # print(t.context)
                    #     # print(np.broadcast_to(t.context, (samples.shape[0],) + t.context.shape))
                    #     # print(th.tensor(np.broadcast_to(t.context, (samples.shape[0],) + t.context.shape)))
                    #     c = th.tensor(copy.deepcopy(np.broadcast_to(t.context, (samples.shape[0],) + t.context.shape)))#.to(self.device)
                    #     ca = c.to(self.device, dtype=th.float)
                    #     # print("YAYAYAAY")
                    #     log_p[tuple(t.context)] = {}
                    #     for action in range(0, 5):
                    #         a = th.tensor(np.ones(samples.shape[0]) * action).to(self.device, dtype=th.float)
                    #         values_c, log_prob_c, _ = self.policy.evaluate_actions(samples, ca, a) # TODO, return features HERE
                    #         log_p[tuple(t.context)][action] = log_prob_c, values_c

                    # div_loss = 0
                    # val_loss = 0
                    # for c in log_p:
                    #     for o in log_p:
                    #         if c != o:
                    #             div = 0
                    #             vals = 0
                    #             for a in range(0, 5):
                    #                 kl = th.exp(log_p[c][a][0]) * (log_p[c][a][0] - log_p[o][a][0])
                    #                 div += th.clamp(th.mean(kl), 0, 0.1)
                    #                 vals += ((log_p[c][a][1] - log_p[o][a][1])**2)**(1/2)
                    #             div_loss += div * np.sum((np.array(c) - np.array(o))**2)**(1/2)
                    #             val_loss += th.clamp(th.mean(vals), 0, 0.1) * np.sum((np.array(c) - np.array(o))**2)**(1/2)
                    # div_loss = div_loss/(len(log_p)**2 - len(log_p))
                    # val_loss = val_loss/(len(log_p)**2 - len(log_p))
                    # loss -= div_loss
                    # loss -= val_loss
                    # divs.append(div_loss.cpu().item())
                    # vvals.append(val_loss.cpu().item())

                    num_context_samples = self.context_size
                    num_obs_samples = 100
                    action_space_size = 5

                    samples = rollout_data.observations[:num_obs_samples]
                    # print(choices)

                    contexts_to_use = np.identity(self.context_size)

                    log_p = {}
                    context_samples = []
                    action_samples = []
                    state_samples = []
                    for c in contexts_to_use:
                        c = th.tensor(copy.deepcopy(np.broadcast_to(c, (samples.shape[0],) + t.context.shape)))#.to(self.device)
                        for action in range(0, action_space_size):
                            a = th.tensor(np.ones(samples.shape[0]) * action)
                            action_samples.append(a)
                            context_samples.append(c)
                            state_samples.append(samples)
                    context_samples = th.cat(context_samples, 0).to(self.device, dtype=th.float)
                    action_samples = th.cat(action_samples, 0).to(self.device, dtype=th.float)
                    state_samples = th.cat(state_samples, 0).to(self.device, dtype=th.float)
                    div_vals, div_log_probs, _ = self.policy.evaluate_actions(state_samples, context_samples, action_samples)
                    # normalize div_vals
                    # div_vals = (div_vals - div_vals.mean()) / (div_vals.std() + 1e-8)
                    div = 0
                    div_sampler = 0
                    val_div = 0
                    s_num = samples.shape[0] * action_space_size # 6 is the number of actions
                    count = 0

                    choices = self.sample_chooser.forward(state_samples[0:s_num])
                    # # print(choices)
                    # # print(choices[:,:1])
                    # impt = state_samples[th.argmax(choices)]
                    # weight = choices[th.argmax(choices)]
                    # print("most important sample: ", impt, " with weight ", weight)
                    # print(choices)

                    for i in range(0, num_context_samples):
                        p_choices = div_log_probs[i*s_num:(i+1)*s_num]
                        # p_vals = div_vals[i*s_num:(i+1)*s_num]
                        for j in range(0, num_context_samples):
                            dist_btw = th.sum((context_samples[i*s_num] - context_samples[j*s_num]) ** 2)
                            
                            if i != j and dist_btw != 0:
                                # print(context_samples[i*s_num], context_samples[j*s_num])
                                count += 1
                                q_choices = div_log_probs[j*s_num:(j+1)*s_num]
                                # q_vals = div_vals[j*s_num:(j+1)*s_num]
                                kl_original = th.exp(p_choices) * (p_choices - q_choices)

                                kl = kl_original * choices.detach()

                                kl_sampler = kl_original.detach() * choices#[:,:1]

                                # div += th.clamp(th.mean(kl), 0, 10)# * dist_btw
                                div += th.clamp(th.sum(kl), 0, 10)
                                # div_sampler += th.clamp(th.mean(kl_sampler), 0, 10)# * dist_btw
                                div_sampler += th.clamp(th.sum(kl_sampler), 0, 10)
                                # val_div += (th.sum(p_vals - q_vals)**2)**(1/2) * dist_btw
                    average_div = div
                    # val_loss = val_div / count
                    loss += -average_div
                    # loss -= val_loss
                    divs.append(average_div.cpu().item())
                    # vvals.append(val_loss.cpu().item())

                    average_div_sampler = div_sampler

                    self.sample_optimizer.zero_grad()
                    # we want to minimize divergence, and minimize target density error
                    entropy_of_choices = th.sum(-choices * th.log(choices))
                    sample_loss = average_div_sampler# + entropy_of_choices# + (0.25 - th.sum(choices[:,:1])/len(choices))**2 # we want NON ENTROPIC choices!!
                    sample_loss.backward(retain_graph=True)
                    self.sample_optimizer.step()
                    sampler_density.append(entropy_of_choices.item())
                    sampler_loss.append(sample_loss.cpu().item())

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

        if self.use_context and use_decider:
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
        # logger.record("train/avg_rew", rew/avg_rew)
        # logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/div_divergence", np.mean(np.array(divs)))
        logger.record("train/sampler_entropy", np.mean(np.array(sampler_density)))
        logger.record("train/sampler_loss", np.mean(np.array(sampler_loss)))
        # logger.record("train/decider_loss", np.mean(np.array(decider_losses)))
        # logger.record("train/vval_loss", np.mean(np.array(vvals)))
        # logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # logger.record("train/clip_fraction", np.mean(clip_fraction))
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
