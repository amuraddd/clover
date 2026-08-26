# MD3PO-SAC Baseline

## Scope and attribution

`md3po_sac.py` combines MD3PO-style diversity replay with an off-policy,
maximum-entropy policy-gradient update for a diffusion model. The implementation
is **SAC-inspired, but it is not canonical Soft Actor-Critic**. In particular,
it does not train a Q-network and does not use SAC's reparameterized actor loss.

The implementation combines ideas from several sources:

- The likelihood-ratio policy gradient comes from REINFORCE and the policy
  gradient theorem.
- Reward-plus-entropy optimization comes from maximum-entropy reinforcement
  learning and Soft Actor-Critic.
- Diffusion denoising transitions are treated as policy actions, following the
  policy-gradient view used by diffusion-model RL methods such as DDPO.
- Stored behavior likelihoods provide a bounded off-policy importance weight.

Consequently, the exact surrogate below should be described as this baseline's
custom entropy-regularized, off-policy policy-gradient estimator—not as an
equation reproduced directly from the SAC paper.

## Diffusion-policy formulation

For trajectory `i` and denoising step `t`, define

\[
s_{i,t}=\text{current latent},\qquad
a_{i,t}=\text{next latent},
\]

and let

\[
\ell_{i,t}(\theta)
=\log\pi_\theta(a_{i,t}\mid s_{i,t},c_i),
\]

where `c_i` is the prompt conditioning. The terminal image reward is copied to
each trainable denoising transition and standardized across the selected batch:

\[
\hat A_i=\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+10^{-8}}.
\]

The code then applies the configured reward scale `beta`:

\[
Q_i^{\mathrm{scaled}}=\beta\hat A_i,
\qquad \beta=\texttt{reward\_scale}.
\]

This is a normalized Monte Carlo return/advantage estimate, not the output of a
learned soft-Q critic.

## Off-policy correction

Replay samples may have been collected by an earlier behavior policy `mu`. The
update computes the bounded importance ratio

\[
\rho_{i,t}=\min\left(
\exp(\ell_{i,t}(\theta)-\log\mu(a_{i,t}\mid s_{i,t},c_i)),
\rho_{\max}
\right).
\]

The implementation detaches this ratio. It therefore acts as a fixed weight
during each backward pass rather than contributing derivatives of its own.

## Reward and entropy loss

Let `sg(x)` denote stop-gradient (`x.detach()` in PyTorch). The reward component
is

\[
L^Q_{i,t}
=-\operatorname{sg}(\rho_{i,t})
  \operatorname{sg}(Q_i^{\mathrm{scaled}})
  \ell_{i,t}(\theta).
\]

The Shannon entropy gradient satisfies

\[
\nabla_\theta H(\pi_\theta)
=-\mathbb E_{a\sim\pi_\theta}
\left[(\log\pi_\theta(a\mid s)+1)
\nabla_\theta\log\pi_\theta(a\mid s)\right].
\]

The corresponding sampled entropy-loss surrogate is

\[
L^H_{i,t}
=\operatorname{sg}(\rho_{i,t})
\left[\operatorname{sg}(\ell_{i,t}(\theta))+1\right]
\ell_{i,t}(\theta).
\]

The complete transition-level surrogate is therefore

\[
L_{i,t}(\theta)
=-\operatorname{sg}(\rho_{i,t})
\left[
\operatorname{sg}(Q_i^{\mathrm{scaled}})
-\left(\operatorname{sg}(\ell_{i,t}(\theta))+1\right)
\right]\ell_{i,t}(\theta).
\]

The stop-gradient notation is essential. Without detaching the log probability
inside the coefficient, automatic differentiation would produce an unintended
`2 log pi + 1` entropy coefficient.

After minibatch averaging and division by the number `T` of denoising
transitions, automatic differentiation produces

\[
\nabla_\theta L
=\frac{1}{T}\mathbb E\left[
\rho_{i,t}\left(-\beta\hat A_i+\ell_{i,t}+1\right)
\nabla_\theta\ell_{i,t}
\right].
\]

`step_loss.backward()` computes this gradient; it is not written explicitly in
the Python implementation.

## Interpreting `reward_scale`

The optimized reward-plus-entropy objective has the nominal form

\[
J(\theta)=\beta\,\mathbb E[\hat A]+H(\pi_\theta).
\]

Thus, `reward_scale` directly scales the reward component, while entropy retains
unit weight. It changes entropy regularization only **relative to reward**:

\[
\beta Q+H=\beta\left(Q+\frac{1}{\beta}H\right).
\]

The effective relative entropy coefficient is therefore `1 / beta`. This does
not set or guarantee a particular entropy value. The current source default is
`reward_scale = 5.0`, which gives reward five times its unit coefficient and an
effective relative entropy coefficient of `0.2`. A value of `1.0` would mean
unit nominal coefficients—not maximum entropy.

This objective-level equivalence does not imply identical optimization dynamics
under Adam, gradient clipping, finite minibatches, or bounded importance ratios.
An explicit entropy coefficient would be clearer if entropy must be controlled
independently of the reward-gradient magnitude.

## MD3PO replay and training flow

Each epoch collects fresh diffusion trajectories, evaluates their terminal
images with the configured project reward function, and records states, actions,
behavior log probabilities, timesteps, prompts, rewards, and images. The latest
saved rollout is compared with the new rollout using prompt similarity and image
diversity. Qualifying replay samples are appended to the fresh batch, and the
combined batch is passed to `sac_update`.

