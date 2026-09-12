# Entropy-Regularized Diffusion Policy with Learned Variance

We begin with the entropy-regularized objective

$$
J(\theta)
=
\mathbb{E}\left[
\rho_t Q_t
\log \pi_\theta(a_t \mid s_t)
+
\alpha
H\left(\pi_\theta(\cdot \mid s_t)\right)
\right].
$$

For the diffusion-policy setting,

$$
s_t = (x_t, c, t),
$$

and

$$
a_t = x_{t-1},
$$

so that the policy is the reverse diffusion transition

$$
\pi_\theta(a_t \mid s_t)
=
p_\theta(x_{t-1} \mid x_t, c).
$$

Therefore, the objective becomes

$$
J(\theta)
=
\mathbb{E}\left[
\rho_t Q_t
\log p_\theta(x_{t-1} \mid x_t, c)
+
\alpha
H\left(
p_\theta(\cdot \mid x_t, c)
\right)
\right].
$$

Here,

$$
\rho_t
=
\frac{
p_\theta(x_{t-1} \mid x_t,c)
}{
p_{\text{old}}(x_{t-1} \mid x_t,c)
}
$$

is the importance-sampling ratio, \(Q_t\) is the return or advantage-like quantity, and \(\alpha\) controls the strength of entropy regularization.

---

## 1. Gaussian reverse diffusion policy

The reverse DDPM transition can be modeled as a Gaussian distribution,

$$
p_\theta(x_{t-1} \mid x_t,c)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,c,t),
\Sigma_\theta(x_t,c,t)
\right).
$$

For a diagonal covariance matrix,

$$
\Sigma_\theta
=
\operatorname{diag}
\left(
\sigma_{\theta,1}^2,
\ldots,
\sigma_{\theta,D}^2
\right).
$$

The model therefore determines both the reverse-process mean

$$
\mu_\theta(x_t,c,t)
$$

and, when variance is learned, the reverse-process variance

$$
\sigma_{\theta,j}^2(x_t,c,t).
$$

---

## 2. Entropy of the diffusion policy

The conditional entropy of the reverse transition is

$$
H_t
=
H\left(
p_\theta(\cdot \mid x_t,c)
\right).
$$

By definition,

$$
H_t
=
-
\mathbb{E}_{
x_{t-1}
\sim
p_\theta(\cdot \mid x_t,c)
}
\left[
\log
p_\theta(x_{t-1} \mid x_t,c)
\right].
$$

For a \(D\)-dimensional Gaussian,

$$
H_t
=
\frac{1}{2}
\log
\left[
(2\pi e)^D
\det \Sigma_\theta
\right].
$$

For diagonal covariance,

$$
\det \Sigma_\theta
=
\prod_{j=1}^{D}
\sigma_{\theta,j}^2.
$$

Therefore,

$$
H_t
=
\frac{1}{2}
\sum_{j=1}^{D}
\left[
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right].
$$

Hence,

$$
\boxed{
H_t
=
\frac{1}{2}
\sum_j
\left[
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right]
}.
$$

This expression makes explicit that Gaussian entropy depends on the variance, but not on the mean.

---

## 3. Why fixed variance gives zero entropy gradient

Suppose the reverse transition uses a fixed scheduler variance,

$$
p_\theta(x_{t-1} \mid x_t,c)
=
\mathcal{N}
\left(
\mu_\theta(x_t,c,t),
\sigma_t^2 I
\right),
$$

where \(\sigma_t^2\) is determined entirely by the diffusion scheduler.

Then

$$
\nabla_\theta \sigma_t^2 = 0.
$$

The entropy becomes

$$
H_t
=
\frac{D}{2}
\left[
1
+
\log(2\pi)
+
\log \sigma_t^2
\right].
$$

Differentiating with respect to \(\theta\),

$$
\nabla_\theta H_t
=
\frac{D}{2}
\nabla_\theta
\log \sigma_t^2.
$$

Because the variance is fixed,

$$
\nabla_\theta \log \sigma_t^2 = 0.
$$

Therefore,

$$
\boxed{
\nabla_\theta H_t = 0
}.
$$

This is true even though

$$
p_\theta(x_{t-1} \mid x_t,c)
$$

depends on \(\theta\) through the mean \(\mu_\theta\).

The reason is that Gaussian entropy is invariant to translation of the mean. For example,

$$
\mathcal{N}(0,1)
$$

and

$$
\mathcal{N}(100,1)
$$

have different means but exactly the same entropy.

Thus, changing only \(\mu_\theta\) cannot change the entropy.

---

## 4. Consequence for the original objective

With fixed variance,

$$
J(\theta)
=
\mathbb{E}
\left[
\rho_t Q_t
\log p_\theta(x_{t-1} \mid x_t,c)
+
\alpha H_t
\right].
$$

Differentiating gives

$$
\nabla_\theta J
=
\mathbb{E}
\left[
\rho_t Q_t
\nabla_\theta
\log p_\theta(x_{t-1} \mid x_t,c)
+
\alpha
\nabla_\theta H_t
\right].
$$

Since

$$
\nabla_\theta H_t = 0,
$$

we obtain

$$
\boxed{
\nabla_\theta J
=
\mathbb{E}
\left[
\rho_t Q_t
\nabla_\theta
\log p_\theta(x_{t-1} \mid x_t,c)
\right]
}.
$$

Therefore, with fixed variance, the entropy coefficient \(\alpha\) does not affect the model gradients.

---

## 5. Why the variance must be learned

To make entropy regularization influence the policy update, the variance must itself depend on the model parameters.

We therefore change the reverse transition from

$$
p_\theta
=
\mathcal{N}
\left(
\mu_\theta,
\sigma_t^2 I
\right)
$$

to

$$
\boxed{
p_\theta
=
\mathcal{N}
\left(
\mu_\theta,
\Sigma_\theta
\right)
}.
$$

For diagonal covariance,

$$
\Sigma_\theta
=
\operatorname{diag}
\left(
\sigma_{\theta,1}^2,
\ldots,
\sigma_{\theta,D}^2
\right).
$$

Now,

$$
\sigma_{\theta,j}^2
=
\sigma_{\theta,j}^2(x_t,c,t),
$$

and consequently,

$$
\nabla_\theta \sigma_{\theta,j}^2
\neq 0.
$$

This makes the entropy directly dependent on trainable parameters.

---

## 6. Derivative of entropy with learned variance

Recall that

$$
H_t
=
\frac{1}{2}
\sum_j
\left[
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right].
$$

Differentiating,

$$
\nabla_\theta H_t
=
\frac{1}{2}
\sum_j
\nabla_\theta
\log \sigma_{\theta,j}^2.
$$

Using

$$
\nabla_\theta \log z
=
\frac{1}{z}
\nabla_\theta z,
$$

we obtain

$$
\nabla_\theta H_t
=
\frac{1}{2}
\sum_j
\frac{
\nabla_\theta
\sigma_{\theta,j}^2
}{
\sigma_{\theta,j}^2
}.
$$

Therefore,

$$
\boxed{
\nabla_\theta H_t
=
\frac{1}{2}
\sum_j
\nabla_\theta
\log \sigma_{\theta,j}^2
}
$$

or equivalently,

$$
\boxed{
\nabla_\theta H_t
=
\frac{1}{2}
\sum_j
\frac{
\nabla_\theta
\sigma_{\theta,j}^2
}{
\sigma_{\theta,j}^2
}
}.
$$

Because the variance is now learned,

$$
\boxed{
\nabla_\theta H_t \neq 0
}.
$$

This is the key reason that learning the variance is necessary for meaningful entropy regularization.

---

## 7. Gradient of the complete objective

Returning to

$$
J(\theta)
=
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\alpha H_t
\right],
$$

the gradient becomes

$$
\nabla_\theta J
=
\mathbb{E}
\left[
\rho_t Q_t
\nabla_\theta
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\alpha
\nabla_\theta H_t
\right].
$$

Substituting the analytic entropy gradient,

$$
\boxed{
\nabla_\theta J
=
\mathbb{E}
\left[
\rho_t Q_t
\nabla_\theta
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\frac{\alpha}{2}
\sum_j
\nabla_\theta
\log \sigma_{\theta,j}^2
\right]
}.
$$

Equivalently,

$$
\boxed{
\nabla_\theta J
=
\mathbb{E}
\left[
\rho_t Q_t
\nabla_\theta
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\frac{\alpha}{2}
\sum_j
\frac{
\nabla_\theta
\sigma_{\theta,j}^2
}{
\sigma_{\theta,j}^2
}
\right]
}.
$$

The entropy term now contributes a genuine optimization direction.

---

## 8. The log probability also depends on the learned variance

For a diagonal Gaussian,

$$
\log
p_\theta(x_{t-1} \mid x_t,c)
=
-
\frac{1}{2}
\sum_j
\left[
\frac{
\left(
x_{t-1,j}
-
\mu_{\theta,j}
\right)^2
}{
\sigma_{\theta,j}^2
}
+
\log
\left(
2\pi
\sigma_{\theta,j}^2
\right)
\right].
$$

Therefore,

$$
\nabla_\theta
\log
p_\theta(x_{t-1} \mid x_t,c)
$$

contains gradients with respect to both

$$
\mu_\theta
$$

and

$$
\sigma_\theta^2.
$$

Thus the reward term

$$
\rho_t Q_t
\nabla_\theta \log p_\theta
$$

can train both the mean and the variance.

The entropy term,

$$
\alpha \nabla_\theta H_t,
$$

provides an additional explicit pressure on the variance.

Conceptually,

$$
Q_t
\quad\Longrightarrow\quad
\text{learn mean and variance that improve reward},
$$

while

$$
\alpha H_t
\quad\Longrightarrow\quad
\text{discourage collapse of the learned variance}.
$$

---

## 9. Final entropy-regularized objective

Substituting the Gaussian entropy explicitly into the objective gives

$$
J(\theta)
=
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\frac{\alpha}{2}
\sum_j
\left(
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right)
\right].
$$

Hence,

$$
\boxed{
J(\theta)
=
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
+
\frac{\alpha}{2}
\sum_j
\left(
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right)
\right]
}.
$$

Since PyTorch optimizers minimize rather than maximize, define

$$
\mathcal{L}(\theta)
=
-J(\theta).
$$

The final loss is therefore

$$
\boxed{
\mathcal{L}(\theta)
=
-
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
\right]
-
\frac{\alpha}{2}
\mathbb{E}
\left[
\sum_j
\left(
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right)
\right]
}.
$$

The terms

$$
1+\log(2\pi)
$$

are constant with respect to \(\theta\), so they can be removed without changing the gradient.

Thus an equivalent optimization loss is

$$
\boxed{
\mathcal{L}(\theta)
=
-
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
\right]
-
\frac{\alpha}{2}
\mathbb{E}
\left[
\sum_j
\log
\sigma_{\theta,j}^2
\right]
}.
$$

---

## 10. Corresponding PyTorch form

The mathematical objective translates directly to

```python
policy_objective = (
    ratio.detach()
    * q_value.detach()
    * log_prob
).mean()

entropy = 0.5 * (
    1.0
    + math.log(2.0 * math.pi)
    + log_variance
)

entropy = entropy.flatten(1).sum(dim=1).mean()

objective = (
    policy_objective
    + alpha * entropy
)

loss = -objective
```

The resulting gradient is

$$
\boxed{
\nabla_\theta J
=
\rho_t Q_t
\nabla_\theta
\log p_\theta(x_{t-1} \mid x_t,c)
+
\alpha
\nabla_\theta
H\left(
p_\theta(\cdot \mid x_t,c)
\right)
}.
$$

Because the variance is now trainable,

$$
\nabla_\theta H
=
\frac{1}{2}
\sum_j
\nabla_\theta
\log \sigma_{\theta,j}^2
\neq 0.
$$

---

## Summary

The complete reasoning can be summarized as

$$
\text{fixed variance}
\Longrightarrow
H
\text{ depends only on the scheduler variance}
\Longrightarrow
\nabla_\theta H = 0,
$$

so

$$
\text{entropy regularization cannot affect the UNet update}.
$$

In contrast,

$$
\text{learned variance}
\Longrightarrow
\sigma_\theta^2
=
\sigma_\theta^2(x_t,c,t),
$$

which gives

$$
H
=
\frac{1}{2}
\sum_j
\left[
1
+
\log(2\pi)
+
\log \sigma_{\theta,j}^2
\right],
$$

and therefore

$$
\nabla_\theta H
=
\frac{1}{2}
\sum_j
\nabla_\theta
\log \sigma_{\theta,j}^2
\neq 0.
$$

The resulting entropy-regularized diffusion-policy loss is

$$
\boxed{
\mathcal{L}(\theta)
=
-
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
\right]
-
\alpha
\mathbb{E}
\left[
H\left(
p_\theta(\cdot \mid x_t,c)
\right)
\right]
}.
$$

For a learned diagonal Gaussian variance, this becomes

$$
\boxed{
\mathcal{L}(\theta)
=
-
\mathbb{E}
\left[
\rho_t Q_t
\log
p_\theta(x_{t-1} \mid x_t,c)
\right]
-
\frac{\alpha}{2}
\mathbb{E}
\left[
\sum_j
\log \sigma_{\theta,j}^2
\right]
}
$$

up to constants that do not affect the gradient.
