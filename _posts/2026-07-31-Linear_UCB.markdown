---
layout: post
title:  "Stochastic Linear Bandits와 LinUCB"
date:   2026-07-31 16:00:00 +0900
tags: [Bandit Algorithms]
published: true
last_modified_at: 2026-07-31 16:05:08 +0900
---

본 포스트에서는 stochastic linear bandit problem과 이를 해결하는 LinUCB 알고리즘을 다룹니다.

##### stochastic k-armed $1$-subgaussian bandit

본격적인 진행에 앞서, stochastic k-armed $1$-subgaussian bandit에서의 UCB 알고리즘을 복습해보겠습니다. $\sigma$-subgaussian으로의 확장은 크게 다르지 않습니다.

<div class="math-box math-box--definition" markdown="1">
<div class="math-box__title">Definition (σ-subgaussian Random Variable)</div>

Random variable $$X$$가 모든 $$\lambda\in\mathbb{R}$$에 대해 아래를 만족하면 $$X$$를 $$\sigma$$-subgaussian이라 합니다.

$$
\mathbb{E}\left[
\exp\left(\lambda X\right)
\right]
\leq
\exp\left(\frac{\lambda^2\sigma^2}{2}\right)
$$

</div>

$X_t$가 $1$-subgaussian이므로 $\hat{\mu}=\frac{1}{n}\sum_{t=1}^nX_t$가 $\frac{1}{\sqrt{n}}$-subgaussian이며 Cramer-Chernoff method를 사용하여 다음과 같이 $\hat{\mu}$의 confidence region을 구할 수 있습니다.

<div class="math-box math-box--theorem" markdown="1">
<div class="math-box__title">Theorem (Cramér-Chernoff Method)</div>

Random variable $$X$$와 임의의 $$a\in\mathbb{R}$$에 대해

$$
\mathbb{P}(X\geq a)
= \mathbb{P}(\exp(\lambda X)\geq \exp(\lambda a))
\leq
\inf_{\lambda>0}
\exp(-\lambda a)\,
\mathbb{E}[\exp(\lambda X)].
$$

$$X$$가 mean $$\mu$$인 $$\sigma$$-subgaussian random variable이면 위 식을 $$\lambda$$에 대해 최적화하여 아래 식을 얻습니다.

$$
\mathbb{P}(X-\mu\geq\epsilon)
\leq
\exp\left(-\frac{\epsilon^2}{2\sigma^2}\right)
$$

</div>

위의 정리에서 $$\epsilon$$ 대신 적절한 상수가 붙은 $$\delta$$를 대입하면 

$$
\mathbb{P}\left(\mu \geq \hat{\mu} + \sqrt{\frac{2 \log (1/\delta)}{n}}\right) \leq \delta \quad \forall \delta \in (0,1).
$$

이를 이용하여 자연스럽게 $\mathrm{UCB}_i$을 정의하고, 이를 최대로 하는 arm을 고르면 regret $R_n$을 아래와 같이 bound 할 수 있게 됩니다!

$$
A_t=\underset{i \in [k]}{\operatorname{argmax}} \mathrm{UCB}_i(t-1, \delta)=\underset{i \in [k]}{\operatorname{argmax}} \left(\hat{\mu}_i(t-1)+\sqrt{\frac{2 \log (1/\delta)}{T_i(t-1)}}\right) \\
R_n \leq 8\sqrt{nk \log n} + 3 \sum_{i=1}^k \Delta_i
$$

이번 포스트의 주제인 Stochastic Linear Bandit에서도 이와 유사하게 진행이 되지만 confidence region을 구하는 과정이 이보다 어렵습니다. 그 과정에 약간의 선형대수학 지식과 martingale에 대한 지식이 필요하지만 최대한 쉽게 풀어보도록 하겠습니다.

#### Stochastic Linear Bandits

Stochastic linear bandit에서는 매 round $t$마다 learner에게 decision set $\mathcal{A}_t\subset\mathbb{R}^d$가 주어지며 action $A_t\in\mathcal{A}_t$를 고를 때 reward $X_t$는 아래 식과 같이 주어집니다. 이때 $\theta^* \in \mathbb{R}^d$는 unknown parameter를, $\eta_t$는 noise를 의미하며 과거의 관측과 현재 action이 주어졌을 때 conditionally $1$-subgaussian이라 가정합니다. $$\mathcal{F}_{t-1}$$는 filtration이라 하며, 쉽게 생각하면 $$A_t$$까지의 모든 과거 정보를 표현한다고 생각하시면 됩니다.

$$
\mathcal{F}_{t-1}=\sigma(\mathcal{A}_1, A_1, X_1,\dots,\mathcal{A}_{t-1}, A_{t-1}, X_{t-1},\mathcal{A}_{t}, A_{t})\\
\mathbb{E}[\exp(\lambda \eta_t)\mid\mathcal{F}_{t-1}] \leq \exp\left(\frac{\lambda^2}{2}\right) \quad \mathrm{a.s.}\\
X_t = \langle \theta^*,A_t\rangle+\eta_t
$$

또한 random pseudo regret과 regret은 자연스럽게 다음과 같이 정의됩니다.

$$
\widehat{R}_n
=
\sum_{t=1}^{n}
\max_{a \in \mathcal{A}_t}\langle\theta^*,a-A_t\rangle\\
R_n
=\mathbb{E}[\widehat{R}_n]=
\mathbb{E}\left[\sum_{t=1}^{n}
\max_{a \in \mathcal{A}_t}\langle\theta^*,a\rangle-\sum_{t=1}^{n}X_t\right]
$$

#### Regularized Least Squares

본격적인 접근을 시작해보겠습니다. 우선 $$X_t = \langle \theta^*,A_t\rangle+\eta_t$$ 은 중회귀 분석에서 접하는 꼴로 Ridge Regression을 통해 각 round마다 $\theta^*$에 대한 estimator를 구할 수 있습니다. $\lambda>0$은 regularization parameter입니다.

$$
\widehat{\theta}_t
=
\operatorname*{argmin}_{\theta\in\mathbb{R}^d}
\left\{
\sum_{s=1}^{t}
(X_s-\langle\theta,A_s\rangle)^2
+\lambda\|\theta\|_2^2
\right\}\\
V_t=\lambda I+\sum_{s=1}^{t}A_sA_s^\top,
\quad
\hat{\theta}_t=V_t^{-1}\sum_{s=1}^t A_s X_s
$$

이제 $$\theta^*$$에 대한 estimator를 구했습니다! Ridge Regression을 진행했기 때문에 자연스럽게 이는 biased estimator일 것입니다.

다음 단계로 Confidence region을 구할 차례입니다.

#### Confidence Ellipsoid

Finite-armed UCB에서는 각 arm의 mean에 대한 confidence interval을 만들었습니다. 하지만 Linear bandit에서는 하나의 parameter $$\theta^*$$가 모든 action의 reward를 결정하므로, interval 대신 $$\mathbb{R}^d$$ 위의 $$\theta^*$$를 포함하는 ellipsoid를 떠올리는 것이 자연스럽습니다.

우선 $$V_t$$을 관찰해보겠습니다. $$V_t$$는 symmetric positive defitnie이며 직교대각화 가능하며 eigen-decomposition을 통해 다음을 얻을 수 있습니다.

$$
V_t=\lambda_1 v_1 v_1^\top + \cdots + \lambda_d v_d v_d^\top\\
\lambda_i= v_i^\top V_t v_i= v_i^\top (\lambda I+\sum_{s=1}^{t}A_sA_s^\top) v_i
= \lambda I + \sum_{s=1}^{t} \langle A_s, v_i \rangle^2
$$

따라서 고윳값 $$\lambda_i$$가 크다는 뜻은 $$t$$까지의 action들이 $$v_i$$와 유사한 방향으로 많이 선택되었다는 뜻입니다. 이를 이용하여 아래와 같은 ellipsoid를 생각해볼 수 있습니다.

$$
\|x\|_{V_t}^2=x^\top V_t x=\sum_{i=1}^d \lambda_i \langle v_i, x \rangle^2,\\
\mathcal{C}_t
=
\left\{
\theta\in\mathbb{R}^d:
\|\theta-\widehat{\theta}_{t-1}\|_{V_{t-1}}^2
\leq \beta_t
\right\}
$$

$\beta_t$는 confidence radius를 결정하는 값입니다. $V_{t-1}$의 eigenvalue가 큰 방향은 이미 많은 정보를 얻은 방향이며 ellipsoid의 폭이 작아집니다. 반대로 거의 관측하지 않은 방향에서는 폭이 크기 때문에 exploration이 필요한 방향을 자연스럽게 표현하게 됩니다!

#### Martingale

$\beta_t$의 값을 구하기 전에, 간단한 스케치를 그려보도록 하겠습니다. 단순화를 위해 $\lambda=0$인 경우를 생각해 봅시다. $ S_t=\sum_{s=1}^t \eta_s A_s $ 라고 두면 아래와 같이 표현됩니다.

$$
\hat{\theta}_t
=V_t^{-1} \sum_{s=1} ^t X_s A_s
= V_t^{-1} \sum_{s=1} ^t (\langle \theta^*, A_s\rangle+\eta_s) A_s
= \theta_* + V_t^{-1}S_t\\
\|\hat{\theta}_t-\theta_*\|_{V_t}=\|V_t^{-1} S_t\|_{V_t} = \|S_t\|_{V_t^{-1}}
$$

따라서 $$\|\hat{\theta}_t-\theta_*\|_{V_t}=\|V_t^{-1} $$를 bound하기 위해서 $$\|S_t\|_{V_t^{-1}}$$을 적절히 bound하면 좋겠다는 직관을 얻을 수 있습니다. 추가로 아래의 식이 만족합니다. 이는 Fenchel duality의 특별한 케이스이며, 우변의 max 내부를 미분해보면 만족한다는 것을 확인할 수 있습니다.

$$
\frac{1}{2} \|S_t\|_{V_t^{-1}}^2=\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle
-\frac{1}{2}\|x\|_{V_t}^2\right)
$$

위의 식의 우변에 exponential을 취한 process $$M_t(x)$$를 생각하면, supermatringale이 됩니다.

$$
M_t(x)
=
\exp\left(
\langle x,S_t\rangle
-\frac{1}{2}\|x\|_{V_t(\lambda)}^2
\right).
$$

<div class="math-box math-box--definition" markdown="1">
<div class="math-box__title">Definition (Supermartingale)</div>

Filtration $$(\mathcal{F}_t)_{t\geq0}$$에 adapted된 integrable process $$(M_t)_{t\geq0}$$가 모든 $$t\geq1$$에 대해 아래를 만족하면 supermartingale이라고 합니다.

$$
\mathbb{E}[M_t\mid\mathcal{F}_{t-1}]
\leq M_{t-1}
\quad\mathrm{a.s.}
$$

</div>

$$M_t(x)$$이 supermartingale임을 증명하는 과정은 다음과 같습니다.

$$
\|x\|_{A_tA_t^\top}^2
=
x^\top A_tA_t^\top x
=
\langle x,A_t\rangle^2
$$

이므로, $$\eta_t$$의 conditional $$1$$-subgaussian 가정에 $$\alpha=\langle x,A_t\rangle$$를 대입하면 아래 부등식을 얻습니다.

$$
\mathbb{E}\left[
\exp\left(\eta_t\langle x,A_t\rangle\right)
\mid\mathcal{F}_{t-1}
\right]
\leq
\exp\left(
\frac{\langle x,A_t\rangle^2}{2}
\right)
=
\exp\left(
\frac{\|x\|_{A_tA_t^\top}^2}{2}
\right)
\quad\mathrm{a.s.}
$$

이제 $$M_{t-1}(x)$$가 $$\mathcal{F}_{t-1}$$-measurable이므로 다음을 확인할 수 있습니다.

$$
\begin{aligned}
\mathbb{E}\left[M_t(x)\mid\mathcal{F}_{t-1}\right]
&=
M_{t-1}(x)
\mathbb{E}\left[
\exp\left(
\eta_t\langle x,A_t\rangle
-\frac{1}{2}\|x\|_{A_tA_t^\top}^2
\right)
\mathrel{}\middle|\mathcal{F}_{t-1}
\right]\\
&\leq
M_{t-1}(x)
\quad\mathrm{a.s.}
\end{aligned}
$$

마지막으로 $$S_0=0$$이고 $$V_0(\lambda)=\lambda I$$이므로 $$M_0(x)=\exp\left(-\frac{\lambda}{2}\|x\|_2^2\right)\leq1.$$

따라서 반복해서 기댓값을 취하면 $$\mathbb{E}[M_t(x)]\leq\mathbb{E}[M_0(x)]\leq1$$이며 모든 고정된 $$x$$에 대해 $$(M_t(x))_{t\geq0}$$는 nonnegative supermartingale임을 증명하였습니다.

이제 Cramer-Chernoff method로 다음과 같이 전개할 수 있습니다.

$$
\begin{aligned}
\mathbb{P}\left(\frac{1}{2} \|\hat{\theta}_t-\theta_*\|_{V_t}^2 \geq \log (1/\delta)\right)
&=\mathbb{P}\left(\exp\left(\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle-\frac{1}{2}\|x\|_{V_t}^2\right)\right) \geq 1/\delta \right)\\
&\leq \delta \mathbb{E}\left[\exp\left(\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle-\frac{1}{2}\|x\|_{V_t}^2\right)\right)\right]\\
&=\delta \mathbb{E} \left[\max_{x \in \mathbb{R}^d} M_t(x) \right]
\end{aligned}
$$

앞서 보았듯 $$M_t(x)$$는 nonnegative supermartingale이므로 각 고정된 $$x$$와 deterministic time $$t$$에 대해서 $$ \mathbb{E}[M_t(x)] \leq \mathbb{E}[M_0(x)] \leq1 $$를 이용할 수 있습니다.

하지만 아쉽게도 우리가 필요한 것은 고정된 $x$에서의 bound가 아닙니다. $\mathbb{E} \left[\max_{x \in \mathbb{R}^d} M_t(x) \right] \geq \max_{x \in \mathbb{R}^d} \mathbb{E} \left[ M_t(x) \right]$가 되기 때문에 이를 이용하여 $$\|\hat{\theta}_t-\theta_*\|_{V_t}^2$$를 적절히 bound할 수 없습니다.


#### Method of Mixtures

이 문제를 해결하는 방법이 method of mixtures입니다. $x$를 Gaussian distribution에서 뽑는다고 생각하고 $M_t(x)$를 평균내면 supermartingale의 성질은 유지하면서 closed form으로 계산할 수 있게 됩니다! (책에서는 Laplace Method로부터의 아이디어를 언급하였습니다.)

<div class="math-box math-box--lemma" markdown="1">
<div class="math-box__title">Lemma (The Mixture is a Nonnegative Supermartingale)</div>

$$h$$가 $$\mathbb{R}^d$$ 위의 probability measure이고

$$
\overline{M}_t
=
\int_{\mathbb{R}^d}M_t(x)\,dh(x)
$$

라고 하겠습니다. 모든 $$x$$에 대해 $$M_t(x)\geq0$$이므로 $$\overline{M}_t\geq0$$이고

$$
\mathbb{E}[\overline{M}_t]
=
\int_{\mathbb{R}^d}\mathbb{E}[M_t(x)]\,dh(x)
\leq
\int_{\mathbb{R}^d}1\,dh(x)
=1.
$$

따라서 $$\overline{M}_t$$는 integrable합니다. 또한 $$(\omega,x)\mapsto M_t(x)(\omega)$$의 joint measurability로부터 $$\overline{M}_t$$가 $$\mathcal{F}_t$$-measurable임을 알 수 있습니다.

이제 supermartingale property가 성립하지 않는다고 가정하겠습니다. 그러면 어떤 $$\epsilon>0$$에 대해

$$
A
=
\left\{
\mathbb{E}[\overline{M}_t\mid\mathcal{F}_{t-1}]
-\overline{M}_{t-1}
>\epsilon
\right\}
\in\mathcal{F}_{t-1},
\qquad
\mathbb{P}(A)>0
$$

인 사건 $$A$$가 존재합니다. 하지만 conditional expectation의 정의와 Fubini-Tonelli theorem을 사용하면

$$
\begin{aligned}
0
&<
\int_A
\left(
\mathbb{E}[\overline{M}_t\mid\mathcal{F}_{t-1}]
-\overline{M}_{t-1}
\right)d\mathbb{P}\\
&=
\int_{\mathbb{R}^d}
\int_A
\left(
M_t(x)-M_{t-1}(x)
\right)
d\mathbb{P}\,dh(x)
\leq0.
\end{aligned}
$$

마지막 등식은 각 $$M_t(x)$$가 supermartingale이고 $$A\in\mathcal{F}_{t-1}$$이기 때문에 성립합니다. 이는 모순이므로

$$
\mathbb{E}[\overline{M}_t\mid\mathcal{F}_{t-1}]
\leq
\overline{M}_{t-1}
\quad\mathrm{a.s.}
$$

입니다. 따라서 $$(\overline{M}_t)_{t\geq0}$$ 역시 nonnegative supermartingale이며, $$\overline{M}_0=1$$입니다.

</div>

위의 과정을 통해 $$\overline{M}_t$$가 supermartingale임을 증명하였으며, closed form으로 계산하면 다음과 같습니다.

$$
\overline{M}_t
=
\int_{\mathbb{R}^d}M_t(x)\,dh(x),
\qquad
h=\mathcal{N}(0,\lambda^{-1}I).\\
\begin{aligned}
\overline{M}_t
&=\frac{1}{(2\pi)^d \det (H^{-1})} \int_{\mathbb{R}^d} \exp \left( \langle x, S_t \rangle - \frac{1}{2} \|x\|_{V_t}^2 - \frac{1}{2} \|x\|_{H}^2 \right)\,dx\\
&=\left(\frac{\det (H)}{\det (H + V_t)}\right)^{1/2} \exp \left(\frac{1}{2} \|S_t\|^2_{(H+V_t)^{-1}}\right)
\end{aligned}
$$

이제 아래 maximal inequality를 이용하면 원했던 $$\|S_t\|_{V_t^{-1}}^2$$을 bound할 수 있게 됩니다.

<div class="math-box math-box--theorem" markdown="1">
<div class="math-box__title">Theorem (Nonnegative Supermartingale Maximal Inequality)</div>

$$(M_t)_{t\geq0}$$가 nonnegative supermartingale이면 모든 $$c>0$$에 대해

$$
\mathbb{P}\left(
\sup_{t\geq0}M_t\geq c
\right)
\leq
\frac{\mathbb{E}[M_0]}{c}.
$$

</div>

$$
\mathbb{P}\left(
\exists t \in \mathbb{N}:
\|S_t\|_{V_t^{-1}}^2
\geq
2\log\frac{1}{\delta}
+\log\frac{\det(V_t)}{\lambda^d}
\right)
\leq\delta.
$$

이 bound를 estimator의 error bound로 옮겨보겠습니다. $$X_s=\langle\theta^*,A_s\rangle+\eta_s$$이고 $$V_t=\lambda I+\sum_{s=1}^tA_sA_s^\top$$이므로

$$
\begin{aligned}
\widehat{\theta}_t-\theta^*
&=
V_t^{-1}\left(S_t-\lambda\theta^*\right),\\
\|\widehat{\theta}_t-\theta^*\|_{V_t}
&=
\|S_t-\lambda\theta^*\|_{V_t^{-1}}\\
&\leq
\|S_t\|_{V_t^{-1}}
+\lambda\|\theta^*\|_{V_t^{-1}}\\
&\leq
\|S_t\|_{V_t^{-1}}
+\sqrt{\lambda}\|\theta^*\|_2.
\end{aligned}
$$

마지막 부등식은 $$V_t\succeq\lambda I$$, 즉 $$V_t^{-1}\preceq\lambda^{-1}I$$에서 따릅니다.

이를 통해 모든 $t \in \mathbb{N}$에 대해 최종적으로 원했던 confidence set을 얻을 수 있게 됩니다.

$$
\mathbb{P}\left(\|\hat{\theta}_t - \theta_*\|_{V_t} < \sqrt{\lambda} \|\theta_*\|_2 + \sqrt{2 \log \frac{1}{\delta} + \log \frac{\det V_t}{\lambda ^d}}\right) > 1 - \delta
$$

따라서 $$\|\theta^*\|_2\leq m_2$$라고 가정한다면, 아래의 confidence radius를 선택하면 됩니다.

$$
\sqrt{\beta_t}
=
\sqrt{\lambda}m_2
+
\sqrt{
2\log\frac{1}{\delta}
+\log\frac{\det(V_{t-1})}{\lambda^d}
}
$$

추가로, $$\|A_t\|_2\leq L$$일 때 아래 식을 이용할 수도 있으며 증명은 Regret Bound part에서 확인할 수 있습니다.

$$
\frac{\det(V_t)}{\lambda^d}
\leq
\left(
1+\frac{tL^2}{d\lambda}
\right)^d
$$

#### LinUCB Algorithm

이제 UCB 알고리즘으로 빌드할 차례입니다. 알고리즘에 쓰일 $$\mathrm{UCB}_t(a)$$는 다음과 같이 구할 수 있습니다. 첫 부등식에서는 아래의 Weighted Cauchy-Schwarz Inequality가 쓰입니다.

$$
\begin{aligned}
\mathrm{UCB}_t(a)
&= \max_{\theta \in \mathcal{C}_t} \langle \theta, a \rangle\\
&= \langle \hat{\theta}_t, a \rangle + \max_{\theta \in \mathcal{C}_t} \langle \theta - \hat{\theta}_t, a \rangle\\
&\leq \langle \hat{\theta}_t, a \rangle + \max_{\theta \in \mathcal{C}_t} \|\theta - \hat{\theta}_t\|_{V_t}\|a\|_{V_t^{-1}}\\
&\leq \langle \hat{\theta}_t, a \rangle + \sqrt{\beta_t} \|a\|_{V_{t-1}^{-1}}
\end{aligned}
$$

<div class="math-box math-box--lemma" markdown="1">
<div class="math-box__title">Lemma (Weighted Cauchy-Schwarz Inequality)</div>

$$V\in\mathbb{R}^{d\times d}$$가 symmetric positive definite이면 모든 $$x,y\in\mathbb{R}^d$$에 대해

$$
|\langle x,y\rangle|
\leq
\|x\|_V\|y\|_{V^{-1}},
\qquad
\|x\|_V=\sqrt{x^\top Vx}.
$$

</div>

$UCB_t(a)$를 이용하여 finite action set에서를 가정하면 다음의 LinUCB 알고리즘을 구성하게 됩니다.

1. $V_0=\lambda I$으로 초기화합니다.
2. round $t$에서 $\widehat{\theta}_{t-1}$을 계산합니다.
3. 각 $a\in\mathcal{A}_t$에 대해

   $$
   \mathrm{UCB}_t(a)
   =
   \langle\widehat{\theta}_{t-1},a\rangle
   +\sqrt{\beta_t}\sqrt{a^\top V_{t-1}^{-1}a}
   $$

   를 계산합니다.
4. $\mathrm{UCB}_t(a)$가 가장 큰 action $A_t$를 선택하고 reward $X_t$를 관측합니다.
5. 다음과 같이 갱신합니다.

   $$
   V_t=V_{t-1}+A_tA_t^\top,
   \qquad
   \hat{\theta}_t=V_t^{-1} \sum_{s=1} ^t X_s A_s.
   $$

드디어 UCB 알고리즘과 유사하게 Stochastic Linear Bandit에서도 LinUCB 알고리즘을 완성하였습니다! 이제 Regret을 분석할 차례입니다.

#### Regret Analysis

Regret 분석을 위해 다음을 가정할 수 있습니다.

1. $1\leq\beta_1\leq\beta_2\leq\cdots\leq\beta_n$.
2. 모든 round에서 action vector의 norm은 $$\|a\|_2\leq L$$로 bounded.
3. 한 round에서 발생할 수 있는 reward gap은 최대 $$1$$.
4. Probability at least $$1-\delta$$로 모든 $$t\in[n]$$에 대해 $$\theta^*\in\mathcal{C}_t$$.

마지막 가정은 앞서 논의한 $$\beta_t$$를 선택하면 만족합니다.

Round $$t$$의 optimal action을 $$A_t^*\in\operatorname*{argmax}_{a\in\mathcal{A}_t}\langle\theta^*,a\rangle$$ 라고 하겠습니다. 또한 LinUCB가 선택한 $$A_t$$에 대해 upper confidence bound를 달성하는 parameter를 $$\widetilde{\theta}_t\in\operatorname*{argmax}_{\theta\in\mathcal{C}_t}\langle\theta,A_t\rangle$$ 라고 두겠습니다. $$\theta^*\in\mathcal{C}_t$$이므로 optimal action의 true expected reward는 그 action의 UCB보다 작거나 같습니다. LinUCB는 UCB가 가장 큰 action을 선택하므로 아래가 성립합니다.

$$
\langle\theta^*,A_t^*\rangle
\leq
\mathrm{UCB}_t(A_t^*)
\leq
\mathrm{UCB}_t(A_t)
=
\langle\widetilde{\theta}_t,A_t\rangle
$$

따라서 매 round에서의 regret $$r_t$$는 다음과 같이 선택한 action 방향의 estimation error로 바꿀 수 있습니다.

$$
\begin{aligned}
r_t
&=
\langle\theta^*,A_t^*-A_t\rangle\\
&\leq
\langle\widetilde{\theta}_t-\theta^*,A_t\rangle\\
&\leq
\|\widetilde{\theta}_t-\theta^*\|_{V_{t-1}}
\|A_t\|_{V_{t-1}^{-1}}.
\end{aligned}
$$

한편 $$\widetilde{\theta}_t$$와 $$\theta^*$$는 모두 $$\mathcal{C}_t$$ 안에 있으므로, confidence ellipsoid의 중심인 $$\widehat{\theta}_{t-1}$$를 거쳐 triangle inequality를 적용하면 다음을 얻습니다.

$$
\|\widetilde{\theta}_t-\theta^*\|_{V_{t-1}}
\leq
\|\widetilde{\theta}_t-\widehat{\theta}_{t-1}\|_{V_{t-1}}
+
\|\widehat{\theta}_{t-1}-\theta^*\|_{V_{t-1}}
\leq 2\sqrt{\beta_t}.
$$

따라서

$$
r_t
\leq
2\sqrt{\beta_t}\|A_t\|_{V_{t-1}^{-1}}
$$

를 얻습니다. 즉 한 round의 regret은 선택한 action 방향에 남아 있는 uncertainty로 제어됩니다. 이미 많이 관측한 방향은 $$V_{t-1}$$가 커져서 $$\|A_t\|_{V_{t-1}^{-1}}$$가 작고, 아직 충분히 관측하지 않은 방향은 이 값이 크게 나타납니다.

이제 각 시점 $$t$$에서의 regret을 모두 더해야 합니다. Reward gap이 최대 $$1$$이고 $$\beta_t\leq\beta_n$$이므로

$$
r_t^2
\leq
4\beta_n
\min\left\{
1,\|A_t\|_{V_{t-1}^{-1}}^2
\right\}.
$$

Cauchy-Schwarz inequality를 통해

$$
\begin{aligned}
\widehat{R}_n
=\sum_{t=1}^{n}r_t
&\leq
\sqrt{n\sum_{t=1}^{n}r_t^2}\\
&\leq
\sqrt{
4n\beta_n
\sum_{t=1}^{n}
\min\left\{
1,\|A_t\|_{V_{t-1}^{-1}}^2
\right\}
}.
\end{aligned}
$$

따라서 남은 문제는 uncertainty의 제곱합을 bound하는 것입니다. 이를 위해 elliptical potential lemma를 사용합니다.

<div class="math-box math-box--lemma" markdown="1">
<div class="math-box__title">Lemma (Elliptical Potential Lemma)</div>

$$V_0\in\mathbb{R}^{d\times d}$$가 positive definite이고 $$V_t=V_0+\sum_{s=1}^{t}A_sA_s^\top$$ 라고 하면

$$
\sum_{t=1}^{n}
\min\left\{
1,\|A_t\|_{V_{t-1}^{-1}}^2
\right\}
\leq
2\log\frac{\det(V_n)}{\det(V_0)}.
$$

</div>

간단한 증명은 다음과 같습니다.

$$V_t=V_{t-1}+A_tA_t^\top$$에 determinant를 적용하면

$$
\det(V_t)
=
\det(V_{t-1})
\left(
1+\|A_t\|_{V_{t-1}^{-1}}^2
\right)
$$

를 얻습니다. 이를 모든 round에 대해 곱하면

$$
\frac{\det(V_n)}{\det(V_0)}
=
\prod_{t=1}^{n}
\left(
1+\|A_t\|_{V_{t-1}^{-1}}^2
\right).
$$

마지막으로 모든 $$u\geq0$$에 대해

$$
\min\{1,u\}\leq2\log(1+u)
$$

가 항상 성립하므로, 양변에 log를 취한 뒤 합하면 elliptical potential lemma가 얻어집니다.

Elliptical potential lemma를 앞의 regret 식에 대입하면 probability at least $$1-\delta$$로

$$
\widehat{R}_n
\leq
\sqrt{
8n\beta_n
\log\frac{\det(V_n)}{\det(V_0)}
}
$$

를 얻습니다.

이것이 바로 LinUCB의 기본 high-probability regret bound입니다!

추가적으로 determinant를 $$n,d,L,\lambda$$로 표현해보겠습니다. $$V_0=\lambda I$$이고 $$\|A_t\|_2\leq L$$이므로

$$
\operatorname{tr}(V_n)
=
d\lambda+\sum_{t=1}^{n}\|A_t\|_2^2
\leq
d\lambda+nL^2.
$$

산술-기하 부등식을 eigenvalue들에 적용하면

$$
\det(V_n)
\leq
\left(
\frac{\operatorname{tr}(V_n)}{d}
\right)^d
\leq
\left(
\lambda+\frac{nL^2}{d}
\right)^d.
$$

그리고 $$\det(V_0)=\lambda^d$$이므로

$$
\log\frac{\det(V_n)}{\det(V_0)}
\leq
d\log\left(
1+\frac{nL^2}{d\lambda}
\right).
$$

결국

$$
\widehat{R}_n
\leq
\sqrt{
8dn\beta_n
\log\left(
1+\frac{nL^2}{d\lambda}
\right)
}
$$

를 얻습니다. 앞에서 구한 confidence radius $$\beta_t$$를 사용하고 $$\delta=1/n$$으로 선택하면 bad event에서 발생하는 regret까지 expectation에 포함할 수 있습니다. 이때 $$\beta_n$$은 logarithmic factor를 제외하면 dimension $$d$$에 비례하므로

$$
R_n=\widetilde{O}(d\sqrt{n})
$$

이 됩니다.

#### Implementation Notes

$V_t^{-1}$을 매 round 처음부터 계산하면 $O(d^3)$의 시간이 필요합니다. $V_t$는 rank-one update이므로 Sherman-Morrison formula를 사용할 수 있습니다.

$$
V_t^{-1}
=
V_{t-1}^{-1}
-
\frac{
V_{t-1}^{-1}A_tA_t^\top V_{t-1}^{-1}
}{
1+A_t^\top V_{t-1}^{-1}A_t
}.
$$

이를 사용하면 inverse update는 $O(d^2)$에 가능합니다. Action의 수가 $k$이면 각 action의 score 계산까지 포함하여 한 round의 시간 복잡도는 $O(kd^2)$입니다.

#### LinUCB vs UCB Experiment

LinUCB와 UCB를 구현하여 expected regret을 비교한 결과입니다.
왼쪽은 두 개의 action을 서로 독립인 one-hot feature로 두었을 때, 오른쪽은 dimension을 $$d=5$$로 고정하고 action의 수 $$k$$를 늘린 linear bandit입니다.

<div class="experiment-figures">
  <img src="{{ site.baseurl }}/assets/images/19_8_a.png" alt="LinUCB와 UCB의 delta에 따른 expected regret 비교">
  <img src="{{ site.baseurl }}/assets/images/19_8_b.png" alt="LinUCB와 UCB의 action 수에 따른 expected regret 비교">
</div>

앞에서 살펴본 UCB의 regret bound는

$$
R_n
\leq
8\sqrt{nk\log n}
+3\sum_{i=1}^{k}\Delta_i
=
\widetilde{O}(\sqrt{nk})
$$

입니다. UCB는 각 action을 독립된 arm으로 다루기 때문에 leading term이 arm의 수 $$k$$에 의존합니다. 반면 LinUCB는 action의 feature를 통해 하나의 parameter를 함께 학습하며, 앞에서 구한 regret bound는 $$\widetilde{O}(d\sqrt{n})$$입니다. 두 알고리즘의 구현은 [`algorithms.hpp`](https://github.com/danielhO9/Bandit-Algorithms/blob/main/include/algorithms.hpp)에서 확인할 수 있습니다.

#### Reference

Tor Lattimore and Csaba Szepesvári, *Bandit Algorithms*, Chapters 19–20.
