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
        context_size=None,
        use_decoder=False,
        use_exploration_kl=False,
        decoder_method='diayn',
        use_learned_sampler=True,
        continuous_contexts=False,
        kl_clip_val=100,
        pretrain_kl=0,
        **kwargs
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
            context_size = context_size,
            use_decoder=use_decoder,
            use_exploration_kl=use_exploration_kl,
            decoder_method=decoder_method,
            use_learned_sampler=use_learned_sampler,
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.use_context = use_context
        self.continuous_contexts = continuous_contexts
        self.kl_clip_val = kl_clip_val
        self.pretrain_kl = pretrain_kl

        if self.use_learned_sampler:
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

    def _get_decider_batches_valor(self, trajectory_batch: List[TrajectoryBufferSamples]):
        decision_length = None
        sample_frequency = 2 # TODO: hardcoded
        observation_shape = trajectory_batch[0].observations.shape[1:] # excludes the length N of the trajectory
        max_trajectory_length = max(t.buffer_size for t in trajectory_batch)
        final_length = (max_trajectory_length if decision_length == None else decision_length)//sample_frequency + 1
        num_trajectories = len(trajectory_batch) # TODO: modify to match decison length
        
        lengths = np.zeros((num_trajectories,))
        trajectories = np.zeros((num_trajectories, final_length,) + observation_shape) # (B, N, Obs)
        targets = np.zeros((num_trajectories, self.env.context_size))
        for idx, t in enumerate(trajectory_batch):
            features = t.observations[0:t.buffer_size:sample_frequency]
            trajectories[idx][0:((t.buffer_size - 1)//sample_frequency)+1] = features
            lengths[idx] = t.buffer_size//sample_frequency
            targets[idx] = t.context
        t_packed = pack_padded_sequence(th.tensor(trajectories).transpose(0, 1), lengths, enforce_sorted=False) # takes (L, B)
        t_packed = t_packed.to(self.device, dtype=th.float)
        return t_packed, targets, lengths # (B, N, Obs), # (B, Cnxt), # (B,)

    def _get_decider_batches_diayn(self, trajectory_batch):
        skip_frequency = 1 # TODO parameter tuning
        observation_shape = trajectory_batch[0].observations.shape[1:] # excludes the length N of the trajectory
        obs_len = observation_shape[0]
        total_states = sum([t.buffer_size for t in trajectory_batch])
        states = np.zeros((total_states,) + (obs_len*2,) + observation_shape[1:]) # we make room for the current, and next observation
        ctx_targets = np.zeros((total_states, self.env.context_size))
        current_idx = 0
        for idx, t in enumerate(trajectory_batch):
            states[current_idx:current_idx+t.buffer_size,0:obs_len] = t.observations
            states[current_idx:current_idx+t.buffer_size-1, obs_len:] = t.observations[1:]
            ctx_targets[current_idx:current_idx+t.buffer_size] = np.broadcast_to(t.context, (t.buffer_size,) + t.context.shape)
            current_idx += t.buffer_size
        states = th.tensor(states).to(self.device, dtype=th.float)
        return states, ctx_targets, None

    def _get_decider_batches_state_based(self, trajectory_batch: List[TrajectoryBufferSamples]):
        observation_shape = trajectory_batch[0].observations.shape[1:] # excludes the length N of the trajectory
        total_states = sum([t.buffer_size for t in trajectory_batch])
        states = np.zeros((total_states,) + observation_shape)
        ctx_targets = np.zeros((total_states, self.env.context_size))
        current_idx = 0
        for idx, t in enumerate(trajectory_batch):
            states[current_idx:current_idx+t.buffer_size] = t.observations
            ctx_targets[current_idx:current_idx+t.buffer_size] = np.broadcast_to(t.context, (t.buffer_size,) + t.context.shape)
            current_idx += t.buffer_size
        states = th.tensor(states).to(self.device, dtype=th.float)
        return states, ctx_targets, None

    def _get_context_error(self, trajectory_batch, with_grad = False):
        d = {'valor': self._get_decider_batches_valor,
             'state': self._get_decider_batches_state_based,
             'diayn': self._get_decider_batches_diayn}
        trajectory_features, context_targets, lengths = d[self.decoder_method](trajectory_batch)
        if with_grad:
            context_predictions = self.decider(trajectory_features) # (batch, context_size)
        else:
            with th.no_grad():
                context_predictions = self.decider(trajectory_features) # (batch, context_size)
        if self.continuous_contexts:
            error = th.sum(F.mse_loss(context_predictions,
                th.tensor(context_targets).to(self.device, dtype=th.float),
                reduce=with_grad), axis=-1)
        else:
            error = F.cross_entropy(context_predictions,
                th.argmax(th.tensor(context_targets).to(self.device, dtype=th.float), dim=-1),
                reduce=with_grad)
        return error

    def _annotate_context_error(self, trajectory_batch):
        error = self._get_context_error(trajectory_batch, False).cpu()
        
        if len(trajectory_batch) == error.shape[0]:
            # then we are using VALOR, and there is 1 error per trajectory
            for idx, t in enumerate(trajectory_batch):
                t.context_error = np.broadcast_to(error[idx].numpy(), (t.buffer_size,))
        else:
            # then there is 1 error term per obs
            current_idx=0
            for idx, t in enumerate(trajectory_batch):
                t.context_error = error[current_idx:current_idx+t.buffer_size]
                current_idx += t.buffer_size

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
        exploration_divs = []
        embedding_divs = []
        sampler_density, sampler_loss = [], []

        # use_decider = False
        if self.use_context and self.use_decoder:
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                self._annotate_context_error(trajectory_batch)
        else:
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                for idx, t in enumerate(trajectory_batch):
                    t.context_error = np.ones(t.rewards.shape)
        
        # self.rollout_buffer.compute_returns_and_advantage()

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
                if self.use_context and self.use_decoder:
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
                # if self.num_timesteps < self.pretrain_kl:
                #     loss = 0
                # loss = 0

                # define exploration loss
                if self.use_exploration_kl:
                    if self.continuous_contexts:
                        num_context_samples = 8  # TODO: provide 10 as an cmd line argument
                    else:
                        num_context_samples = self.context_size
                    num_obs_samples = 100 # TODO: find best parameter
                    action_space_size = self.env.max_action_num

                    # what if we only care about diversifying between actions that actually make a difference in the outcome of the next state?

                    samples = rollout_data.observations[:num_obs_samples] # This is a random sample of 50 states
                    # samples = np.stack([self.env.observation_space.sample() for _ in range(0, 10)])
                    # samples = th.tensor(samples, dtype=th.float)

                    with th.no_grad():
                        if not self.continuous_contexts:
                            contexts_to_use = np.identity(self.context_size)
                        else:
                            contexts_to_use = np.random.random((num_context_samples, self.context_size))

                        context_samples = []
                        action_samples = []
                        state_samples = []
                        for c in contexts_to_use:
                            c = th.tensor(copy.deepcopy(np.broadcast_to(c, (samples.shape[0],) + (self.context_size,))))#.to(self.device)
                            for action in range(0, action_space_size):
                                a = th.tensor(np.ones(samples.shape[0]) * action)
                                action_samples.append(a)
                                context_samples.append(c)
                                state_samples.append(samples)
                        context_samples = th.cat(context_samples, 0).to(self.device, dtype=th.float)
                        action_samples = th.cat(action_samples, 0).to(self.device, dtype=th.float)
                        state_samples = th.cat(state_samples, 0).to(self.device, dtype=th.float)


                    _, div_log_probs, _, latent_pi, latent_vf, _ = self.policy.evaluate_actions(state_samples, context_samples, action_samples, return_all=True)

                    # state_samples_singular = th.tensor(samples).to(self.device, dtype=th.float)
                    # latent_pi, latent_vf, _ = self.policy._get_latent(state_samples_singular, context_samples)

                    kl_policy = 0
                    s_num = samples.shape[0] * action_space_size
                    max_div_val = samples.shape[0] * 2
                    count = 0
                    min_kl_policy = 1e8
                    max_kl_policy = -1e8
                    kl_emb = 0
                    for i in range(0, num_context_samples):
                        p_choices = div_log_probs[i*s_num:(i+1)*s_num]

                        '''
                        NEW
                        '''
                        p_emb_pi, p_emb_vf = latent_pi[i*s_num:(i+1)*s_num], latent_vf[i*s_num:(i+1)*s_num]

                        for j in range(i+1, num_context_samples):

                            '''
                            As suggested in BigGAN-AM paper:
                            '''
                            # q_choices = div_log_probs[j*s_num:(j+1)*s_num]
                            ci = context_samples[i*s_num]
                            cj = context_samples[j*s_num]
                            # p = th.exp(p_choices)
                            # q = th.exp(q_choices)
                            # kl_policy -= (th.sum((p - q)**2))**(1/2) / (th.sum((ci - cj)**2))**(1/2) # we want to maximize this value
                            # count += 1

                            ci_first, ci_second = th.split(ci, self.context_size // 2, dim=-1)
                            cj_first, cj_second = th.split(cj, self.context_size // 2, dim=-1)

                            '''
                            Homebrewed CL loss:
                            '''
                            # dist_btw = th.sum((context_samples[i*s_num] - context_samples[j*s_num]) ** 2)**(1/2)
                            dist_btw = th.sum(th.abs(ci - cj))

                            dist_first = th.sum(th.abs(ci_first - cj_first))
                            dist_second = th.sum(th.abs(ci_second - cj_second))

                            if self.continuous_contexts:
                                dist_btw = dist_btw / self.context_size

                                dist_btw_first = dist_first / self.context_size * 2
                                dist_btw_second = dist_second / self.context_size * 2
                                # print(ci_first, ci_second, dist_btw_first)
                            else:
                                dist_btw = 1 if dist_btw != 0 else 0
                            if i != j and dist_btw != 0:
                                count += 1
                                q_choices = div_log_probs[j*s_num:(j+1)*s_num]
                                divs = th.sum(th.abs(th.exp(p_choices) - th.exp(q_choices)))
                                assert (0 <= divs.cpu().item() <= max_div_val), "divs is " + str(divs.cpu().item())
                                kl_policy += th.abs((divs / max_div_val) - (dist_btw))

                                '''
                                New term to encourage distance between embeddings:
                                '''
                                q_emb_pi, q_emb_vf = latent_pi[j*s_num:(j+1)*s_num], latent_vf[j*s_num:(j+1)*s_num]

                                max_emb_div = p_emb_pi.shape[0] * p_emb_pi.shape[1] * 2 # Number of (state, action, context) samples x Hidden layer dimension x Max variation of hidden layer
                                
                                emb_divs = th.sum(th.abs(p_emb_pi - q_emb_pi))
                                kl_emb += th.abs(emb_divs / max_emb_div - dist_btw_first)
                                
                                val_divs = th.sum(th.abs(p_emb_vf - q_emb_vf))
                                kl_emb += th.abs(val_divs / max_emb_div - dist_btw_first)
                                # kl_emb -= th.norm(p_emb_vf - q_emb_vf) / th.norm(ci - cj)

                    loss += kl_policy / count # TODO: didn't do this for the field experiments
                    loss += kl_emb / count

                    exploration_divs.append(kl_policy.cpu().item() / count)
                    embedding_divs.append(kl_emb.cpu().item() / count)

                    # what if we want to maximize the Reward, by choosing which states to use in our KL-term?
                    # or we choose the highest value state and maximize divergence there?
                    # ORRRR we only consider states that have a low entropy of actions!!!

                    if self.use_learned_sampler:
                        self.sample_optimizer.zero_grad()
                        # we want to minimize divergence
                        entropy_of_choices = th.sum(-choices * th.log(choices))
                        sample_loss = kl_sampler# + entropy_of_choices# + (0.25 - th.sum(choices[:,:1])/len(choices))**2 # we want NON ENTROPIC choices!!
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

        if self.use_context and self.use_decoder:
            # Decider update
            for trajectory_batch in self.rollout_buffer.get(self.batch_size):
                decider_loss = self._get_context_error(trajectory_batch, with_grad=True)
                decider_losses.append(decider_loss.item())
                self.decider_opt.zero_grad()
                decider_loss.backward()
                self.decider_opt.step()

        self._n_updates += self.n_epochs
        # explained_var = explained_variance(self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        # Logs
        if True:   
            logger.record("train/avg_episode_len", num_steps_in_epoch/(self.n_epochs * self.n_trajectories))
            # logger.record("train/avg_rew", rew/avg_rew)
            logger.record("train/entropy_loss", np.mean(entropy_losses))
            logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            logger.record("train/value_loss", np.mean(value_losses))
            logger.record("train/div_divergence", np.mean(np.array(exploration_divs)))
            logger.record("train/embedding_divergence", np.mean(np.array(embedding_divs)))
            logger.record("train/sampler_entropy", np.mean(np.array(sampler_density)))
            logger.record("train/sampler_loss", np.mean(np.array(sampler_loss)))
            logger.record("train/decider_loss", np.mean(np.array(decider_losses)))
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
        else:
            print({"eps_len": num_steps_in_epoch/(self.n_epochs * self.n_trajectories),
                    "sampler_entropy": np.mean(np.array(sampler_density)),
                    "sampler_loss": np.mean(np.array(sampler_loss)),
                    "decider_loss": np.mean(np.array(decider_losses)),
                    "n_updates": self._n_updates})

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
