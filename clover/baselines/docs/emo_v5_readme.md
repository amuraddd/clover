# EMO v5: Objective, Learned Variance, and Experimental Setup

This document follows the mathematical presentation of [EMO v2](emo_v2_readme.md) and describes the current implementation in [emo_v5.py](../emo_v5.py). The configuration class retains the name `EMOV2V2Config`.

## 1. Diffusion policy and learned variance head

The variance head is a trainable FP32 convolution alongside the frozen noise-output convolution, with matching kernel geometry and one variance prediction per latent channel and spatial location. Its weights start at zero, and its bias initializes the variance near the DDPM posterior-variance bound; a bounded transformation keeps predictions within the scheduler's learned log-variance range. Learning variance is necessary for the analytic entropy term to influence optimization: Gaussian entropy depends on variance, so a fixed scheduler variance would give zero entropy gradient even when the policy mean is trainable.

For conditioning prompt \(c\), the policy state and action are

$$
s_t=(x_t,c,t),\qquad a_t=x_{t-1}.
$$

The stochastic reverse transition is a diagonal Gaussian:

$$
\pi_\theta(a_t\mid s_t)
=\mathcal N\left(a_t;\mu_\theta(s_t),
\operatorname{diag}(v_\theta(s_t))\right),
\qquad v_\theta=\eta^2\sigma_\theta^2.
$$

The DDPM mean is computed from the noise prediction. The UNet output contains \(2C\) channels: \(C\) noise channels followed by \(C\) learned variance-range channels.

`EMOV2OutputHead` preserves the pretrained output convolution as a frozen noise head and adds a trainable convolution with matching geometry for variance prediction. LoRA adapters in the UNet remain trainable, so the noise prediction and reverse mean can still change through their input features. The variance convolution runs in FP32.

For a variance logit \(z_{\theta,j}\), the head computes

$$
u_{\theta,j}=\operatorname{softplus}(z_{\theta,j}),
\qquad f_{\theta,j}=1-\exp(-u_{\theta,j}),
\qquad r_{\theta,j}=2f_{\theta,j}-1.
$$

Equivalently, \(f_{\theta,j}=\operatorname{sigmoid}(z_{\theta,j})\) and \(r_{\theta,j}=\tanh(z_{\theta,j}/2)\). The variance weights start at zero and the bias is initialized to \(2\operatorname{atanh}(-0.9)\), giving \(r=-0.9\) and \(f=0.05\) initially.

Let \(t'\) be the preceding scheduler timestep, \(\bar\alpha_t\) the cumulative noise schedule, and

$$
\beta_t^{\mathrm{step}}=1-\frac{\bar\alpha_t}{\bar\alpha_{t'}},
\qquad
\widetilde\beta_t
=\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}\beta_t^{\mathrm{step}}.
$$

The learned-range variance interpolates in log space:

$$
\boxed{
\log\sigma_{\theta,j}^2
=(1-f_{\theta,j})\log\widetilde\beta_t
+f_{\theta,j}\log\beta_t^{\mathrm{step}}
}.
$$

Both scheduler variances are clamped below by \(10^{-20}\) before taking logarithms. Classifier-free guidance is applied only to the noise channels; the variance channels come directly from the conditional prediction. The final deterministic transition is excluded from policy training.

---

## 2. Reward-weighted policy objective

Let \(R_i\) be the terminal image reward for trajectory \(i\). The baseline uses fresh trajectories with the same exact prompt. For a fresh trajectory, its own reward is excluded; for replay, all fresh same-prompt trajectories are eligible. If there are no eligible peers, the baseline is zero:

$$
b_i=\begin{cases}
\displaystyle\frac{1}{|\mathcal F_i|}\sum_{k\in\mathcal F_i}R_k,
& |\mathcal F_i|>0,\\
0,& |\mathcal F_i|=0,
\end{cases}
\qquad A_i=R_i-b_i.
$$

For \(K\) retained stochastic transitions, indexed in sampling order by \(k=0,\ldots,K-1\), the update uses

$$
Q_{i,k}=\operatorname{round}_5
\left(w_R\gamma^{K-1-k}A_i\right),
$$

where \(w_R\) is `reward_scale`. These quantities are detached and are not standardized. Despite the variable name `soft_q`, there is no learned critic or entropy-augmented Bellman target.

With \(D\) latent dimensions and \(\lambda\) denoting `likelihood_scale`, the implemented log-likelihood statistic is

$$
\ell_{\theta,i,k}
=-\frac{\lambda}{2D}\sum_{j=1}^{D}
\left[
\frac{(a_{i,k,j}-\mu_{\theta,i,k,j})^2}{v_{\theta,i,k,j}}
+\log(2\pi v_{\theta,i,k,j})
\right].
$$

This is a scaled mean over latent dimensions. Consequently, the following importance weight is based on that statistic, rather than the full joint Gaussian likelihood:

$$
\widehat\rho_{i,k}
=\operatorname{stopgrad}\left[
\operatorname{clip}\left(
\exp(\ell_{\theta,i,k}-\ell_{\mathrm{old},i,k}),
1-\epsilon,1+\epsilon
\right)\right].
$$

Here, \(\ell_{\mathrm{old}}\) is stored at trajectory collection time, including for replay. The policy term is

$$
J_{\mathrm{policy}}
=\mathbb E_{i,k}\left[
\widehat\rho_{i,k}Q_{i,k}\ell_{\theta,i,k}
\right].
$$

Its gradient is \(\mathbb E[\widehat\rho Q\nabla_\theta\ell_\theta]\). The ratio is detached, so this is a weighted score-function surrogate, not the usual PPO minimum of clipped and unclipped objectives.

---

## 3. Entropy and cross-entropy regularization

The analytic entropy helper averages over latent dimensions:

$$
\overline H_\theta(s_t)
=\frac{1}{2D}\sum_{j=1}^{D}
\left[1+\log(2\pi)+\log\sigma_{\theta,j}^2(s_t)\right].
$$

It uses the learned variance before multiplication by \(\eta^2\). For fixed positive \(\eta\), the actual transition entropy per dimension is \(\overline H_\theta+\log\eta\); this additive constant does not change the gradient. The two values coincide at the default \(\eta=1\).

Unlike a fixed scheduler variance, the learned variance allows

$$
\nabla_\theta\overline H_\theta
=\frac{1}{2D}\sum_j\nabla_\theta\log\sigma_{\theta,j}^2.
$$

In particular, at fixed scheduler bounds,

$$
\frac{\partial\overline H_\theta}{\partial z_{\theta,j}}
=\frac{1}{2D}f_{\theta,j}(1-f_{\theta,j})
\log\frac{\beta_t^{\mathrm{step}}}{\widetilde\beta_t}.
$$

Entropy therefore supplies a direct variance gradient where the bounds differ and the head is not saturated. It has no direct derivative with respect to the Gaussian mean, although shared trainable features can affect both outputs.

At the beginning of each `sac_update`, the code creates a frozen policy snapshot \(\pi_{\mathrm{ref}}\), provided `cross_entropy_coefficient` is positive. The snapshot remains fixed across every minibatch and inner update pass in that call. It is distinct from the collection-time policy represented by stored log probabilities.

The cross entropy is oriented from the current policy to the reference:

$$
\overline C(\pi_\theta,\pi_{\mathrm{ref}})
=-\frac{1}{D}\mathbb E_{a\sim\pi_\theta}
\left[\log\pi_{\mathrm{ref}}(a\mid s)\right]
$$

$$
\boxed{
\overline C
=\frac{1}{2D}\sum_j
\left[
\log(2\pi v_{\mathrm{ref},j})
+\frac{v_{\theta,j}+(\mu_{\theta,j}-\mu_{\mathrm{ref},j})^2}
{v_{\mathrm{ref},j}}
\right]
}.
$$

Both variances include \(\eta^2\). Reference parameters are detached; gradients flow through the current mean and variance. The penalty discourages displacement from the reference mean and increasing variance relative to the reference.

### Complete implemented objective

Writing \(\kappa\) for `cross_entropy_coefficient`, the current code maximizes

$$
\boxed{
J(\theta)=\mathbb E_{i,k}\left[
\widehat\rho_{i,k}Q_{i,k}\ell_{\theta,i,k}
+\overline H_\theta(s_{i,k})
-\kappa\overline C(\pi_\theta,\pi_{\mathrm{ref}})
\right],\qquad \mathcal L=-J
}.
$$

**The effective entropy coefficient is 1.** `entropy_scale` is validated and logged but does not multiply entropy in the current objective or `entropy_bonus` metric. Changing that setting alone does not change the update.

For fixed positive \(\eta\), the identity \(\overline C=\overline H_{\mathrm{actual}}+D^{-1}\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})\) gives, up to a parameter-independent constant,

$$
\overline H_\theta-\kappa\overline C
=(1-\kappa)\overline H_\theta
-\frac{\kappa}{D}\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}})
+\text{constant}.
$$

Thus, cross entropy is not itself a KL penalty: combined with the unit entropy bonus, it changes both the effective entropy pressure and the reference-policy constraint.

The corresponding per-timestep PyTorch form is

```python
policy_objective = (ratio.detach() * q_value.detach() * log_prob).mean()
entropy_objective = entropy.mean()
cross_entropy_objective = cross_entropy_to_reference.mean()
objective = (
    policy_objective
    + entropy_objective
    - config.cross_entropy_coefficient * cross_entropy_objective
)
step_loss = -objective / trajectory_len
step_loss.backward()
```

Gradients accumulate across retained timesteps, followed by gradient clipping and one optimizer step per minibatch.

---

## 4. Experimental setup

The base model is `runwayml/stable-diffusion-v1-5`, generating 512 × 512 images with 50 denoising steps, guidance scale 5, and \(\eta=1\). The negative prompt is `blurry, low quality, distorted`. Train and evaluation prompt sets come from `get_prompts(seed=123, save=True)`; training samples prompt batches from the training set. Training uses CLIP reward by default through `make_reward_fn`, which resolves functions in `clover/utils/rewards_utils.py`.

The table separates standalone configuration defaults from the current standard experiment launcher, `main.py`. Launcher settings are a code snapshot, not evidence that every historical run used them.

| Setting | `EMOV2V2Config` default | Current `main.py` setup |
| --- | --- | --- |
| Training seeds | 17 | 123, 456 |
| Training epochs | 10 | 25 |
| Fresh trajectories per epoch | 256 | 256 |
| Minibatch size | 32 | 64 |
| Update passes (`sac_epochs`) | 2 | 3 |
| Initial learning rate | 3e-4 | 1e-4 |
| AdamW betas | (0.9, 0.999) | (0.9, 0.999) |
| Adam epsilon | 1e-4 | 1e-8 |
| LoRA rank / alpha | 16 / 16 | 32 / 32 |
| Gradient norm limit | 0.1 | 0.1 |
| Discount factor | 1 − 1/50 = 0.98 | 0.9 |
| Reward scale | 4 | 4 (baseline default) |
| Ratio clipping radius | 1e-4 | 1e-4 |
| Cross-entropy coefficient | 0.01 | 0.10 |
| `entropy_scale` (inactive) | 0.05 | 0.20 |
| Effective entropy coefficient | 1 | 1 |
| Replay diversity threshold | 0.5 | 0.5 |
| Replay reset interval | 5 epochs | 15 epochs |
| Evaluation interval | 2 epochs | 2 epochs |
| Checkpoint interval | 5 epochs | 5 epochs |

LoRA targets `to_v`, `to_k`, `to_q`, and `to_out.0`, with zero adapter dropout. Mixed precision and gradient checkpointing are enabled by default. Learning rate follows `CosineAnnealingWarmRestarts` with `T_0=buffer_reset`, `T_mult=1`, and minimum learning rate `3e-6`. The separate ten-epoch decay helper is unused by training.

### Replay sampling

Each epoch combines all fresh trajectories with qualifying saved trajectories from the current replay interval. A saved sample must have an exact prompt match, terminal reward at least the fresh same-prompt mean, and diversity score strictly above `diversity_threshold`.

The diversity score uses squared Inception feature distance divided by feature dimension. Valid same-prompt pair distances are min-max normalized within each saved-rollout comparison; the minimum normalized distance to a fresh same-prompt image becomes the candidate's score. This is a singleton feature-distance criterion, not a dataset-level FID estimate. A zero distance range produces zero scores.

Only fresh trajectories are appended to the replay buffer. The buffer is cleared after each `buffer_reset` completed epochs, so the next epoch begins fresh. Replay contributes policy samples but never contributes to the fresh-sample reward baselines.

### Evaluation and parameter comparisons

Evaluation uses the same learned-variance sampler and noise-only guidance, with generation seeds 123, 456, and 789 (separate from the training seed). Every configured evaluation interval, the implementation computes both CLIP and BERT rewards, recording per-image values, means, sample standard deviations, counts, and normal-approximation 95% confidence intervals in `<output_dir>/evals/eval_metrics.json`. Standard launcher outputs are under `outputs/emo_v5/seed_<seed>/`. These intervals summarize evaluation images within a run, not variability across training seeds.

The separate `emo_v5_experiments.py` defines a parameter grid over reward scales {5, 10, 20}, cross-entropy coefficients {0.01, 0.05, 0.1}, and diversity thresholds {0.25, 0.5, 0.75}, using seeds 123 and 456. This specifies 54 combinations before its completed-run exclusions; fixed settings are inherited from `main.py`. It is separate from the standard launcher.

For the standard setup, submit from the project root with `sbatch run_experiments.sh`. This script launches `main.py` in the existing project environment and requests two GPUs. No experiments are required to use or review this documentation.


### Periodic FID and DreamSim evaluation

Periodic evaluation generates 10 images for each of the standard 10 held-out prompts for each generation seed in `(123, 456, 789)` (100 images per seed, 300 total), using the learned-variance sampler. Prompt index `i` uses seed `eval_seed + i`, held fixed across epochs and recorded in `prompt_seeds`. CLIP and BERT rewards are retained for all images. Seeds run sequentially, with a separate FID and within-prompt DreamSim calculation for each seed.

`clover/evaluate/emo_v5_metrics.py` computes pooled 2048-dimensional [pytorch-fid](https://github.com/mseitzer/pytorch-fid) against 10 unmodified SD 1.5 reference images per prompt. The reference pipeline uses DDIM with eta 0, the same resolution, step count, guidance, negative prompt and per-prompt seeds, with no LoRA, learned variance head, or safety checker. References are cached by generation seed and these generation settings under `clover/data/emo_v5/evals/sd15_reference/`. This measures distributional distance from SD 1.5, not distance from real images. With only 100 images per set, FID is a noisy, small-sample comparison; it is not a large-scale benchmark FID.

DreamSim reuses the project's locked uv backend and [official ensemble embeddings](https://github.com/ssundaram21/dreamsim). For each prompt, diversity is the sum of sample variances of L2-normalized embeddings (`correction=1`), equivalently mean cosine distance over distinct image pairs. The reported overall value weights prompts equally. Higher values mean more diversity. The worker runs sequentially on the training device, exposing only that GPU. SD 1.5 uses model CPU offload on CUDA to limit additional resident memory.

Generated images, manifests, and DreamSim worker requests/results/logs live under `clover/data/emo_v5/evals/<output-path-hash>/epoch_<epoch>/seed_<eval_seed>/`. `<output_dir>/evals/eval_metrics.json` retains reward fields and adds `fid`, `fid_metadata`, `dreamsim_diversity`, and `dreamsim_diversity_details` (including per-prompt scores). Existing historical entries remain readable. Model weights may download on first use; missing DreamSim dependencies are installed locally through its locked uv project. Submit experiments with `sbatch run_experiments.sh`.

Each completed seed appends its own record to `<output_dir>/evals/eval_metrics.json`, identified by `epoch` and `eval_seed`, so completed seed scores survive a later seed failure. To summarize generation variability for a fixed training checkpoint, group records by `epoch` and compute the mean and sample standard deviation (`ddof=1`) across the three distinct `eval_seed` values. Use `fid`, `dreamsim_diversity`, `clip_reward_summary.mean`, and `bert_reward_summary.mean` as the seed-level scores. The reward summaries' existing `std` fields describe variation among images within a seed; they are not across-seed standard deviations. Per-prompt DreamSim values remain in `dreamsim_diversity_details.per_prompt`. Historical records are preserved, so check seed coverage and deduplicate repeated `(epoch, eval_seed)` records if evaluation was rerun.
