---
layout: post
title:  "Stochastic Linear Bandits와 LinUCB"
date:   2026-07-31 16:00:00 +0900
tags: [Bandit Algorithms]
published: true
last_modified_at: 2026-08-11 12:30:08 +0900
---

본 포스트에서는 stochastic linear bandit problem과 이를 해결하는 LinUCB 알고리즘을 다룹니다.

##### stochastic k-armed $1$-subgaussian bandit

본격적인 진행에 앞서, stochastic k-armed $1$-subgaussian bandit에서의 UCB 알고리즘을 복습해보겠습니다. ( $\sigma$-subgaussian으로의 확장은 크게 다르지 않습니다.) arm의 개수를 $k$, horizon을 $n$으로 두며, 시점 $t$에서 선택한 arm (action) 의 reward random variable이 $X_t$ 입니다. arm $i$ 의 reward는 평균이 $\mu_i$인 $1$-subgaussian random variable입니다.

<div class="math-box math-box--definition" markdown="1">
<div class="math-box__title">Definition (σ-subgaussian Random Variable)</div>

Random variable $$X$$가 모든 $$\lambda\in\mathbb{R}$$에 대해 아래를 만족하면 $$X$$는 $$\sigma$$-subgaussian이다.

$$
\mathbb{E}\left[
\exp\left(\lambda X\right)
\right]
\leq
\exp\left(\frac{\lambda^2\sigma^2}{2}\right)
$$

</div>

$X_t$가 $1$-subgaussian이므로 $\hat{\mu}=\frac{1}{n}\sum_{t=1}^nX_t$은 $\frac{1}{\sqrt{n}}$-subgaussian이며 Cramer-Chernoff method를 사용하여 다음과 같이 $\hat{\mu}$의 confidence region을 구할 수 있습니다.

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

특히 $$X$$가 $$\sigma$$-subgaussian random variable이면

$$
\mathbb{P}(X\geq a)
\leq
\inf_{\lambda>0}
\exp(-\lambda a)\,
\exp\left(\frac{\lambda^2\sigma^2}{2}\right)
\leq
\exp\left(-\frac{a^2}{2\sigma^2}\right).\\
\mathbb{P}\left(X\geq \sqrt{2 \sigma ^2 \log (1/\delta)}\right)
\leq \delta \quad \forall \delta \in (0,1).
$$

</div>

위의 정리에서,

$$
\mathbb{P}\left(\mu  - \hat{\mu}\geq \sqrt{\frac{2 \log (1/\delta)}{n}}\right) \leq \delta \quad \forall \delta \in (0,1).
$$

이를 이용하여 자연스럽게 $\mathrm{UCB}_i$을 정의하고, 이를 최대로 하는 arm을 고르면 regret $R_n$을 아래와 같이 bound 할 수 있게 됩니다! (자세한 증명 과정은 생략합니다.)

$$
A_t=\underset{i \in [k]}{\operatorname{argmax}} \mathrm{UCB}_i(t-1, \delta)=\underset{i \in [k]}{\operatorname{argmax}} \left(\hat{\mu}_i(t-1)+\sqrt{\frac{2 \log (1/\delta)}{T_i(t-1)}}\right) \\
R_n \leq 8\sqrt{nk \log n} + 3 \sum_{i=1}^k \Delta_i
$$

이번 포스트의 주제인 Stochastic Linear Bandit에서도 이와 유사하게 진행이 되지만 confidence region을 구하는 과정이 이보다 어렵습니다.

#### Stochastic Linear Bandits

Stochastic linear bandit에서는 매 round $t$마다 learner에게 decision set $\mathcal{A}_t\subset\mathbb{R}^d$가 주어지며 action $A_t\in\mathcal{A}_t$를 고를 때 reward $X_t$는 아래 식과 같이 주어집니다. 이때 $\theta^* \in \mathbb{R}^d$는 unknown parameter를, $\eta_t$는 noise를 의미하며 과거의 관측과 현재 action이 주어졌을 때 conditionally $1$-subgaussian이라 가정합니다.

$$
\mathcal{F}_{t-1}=\sigma(\mathcal{A}_1, A_1, X_1,\dots,\mathcal{A}_{t-1}, A_{t-1}, X_{t-1},\mathcal{A}_{t}, A_{t})\\
\forall \lambda \geq 0,\quad \mathbb{E}[\exp(\lambda \eta_t)\mid\mathcal{F}_{t-1}] \leq \exp\left(\frac{\lambda^2}{2}\right) \quad \mathrm{a.s.}\\
X_t = \langle \theta^*,A_t\rangle+\eta_t
$$

또한 random pseudo regret과 regret은 다음과 같이 정의합니다.

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

#### LinUCB Algorithm Sketch

LinUCB 알고리즘 또한 UCB 알고리즘과 유사하며 각 시점 $t$에서의 과정은 아래와 같습니다.

1. $\theta^*$에 대한 confidence region $\mathcal{C}_t$을 구한다.
2. $$\operatorname*{argmax}_{a\in\mathcal{A}_t} \max_{\theta \in \mathcal{C}_t} \langle \theta, a \rangle$$를 선택한다.

기존의 UCB 알고리즘과는 다르게 unknwon vector $\theta^*$에 대한 confidence region $$\mathcal{C}_t$$를 구하는 과정이 추가되었습니다.

#### Regularized Least Squares

우선 $X_t$를 설명변수, $A_t$를 반응변수로 보아 $$X_t = \langle \theta^*,A_t\rangle+\eta_t$$ 에서  Ridge Regression을 통해 각 round마다 $\theta^*$에 대한 estimator를 구할 수 있습니다. $\lambda>0$은 regularization parameter입니다.

$$
\widehat{\theta}_t
=
\operatorname*{argmin}_{\theta\in\mathbb{R}^d}
\left\{
\sum_{s=1}^{t}
(X_s-\langle\theta,A_s\rangle)^2
+\lambda\|\theta\|_2^2
\right\}\\
V_t(\lambda)=\lambda I+\sum_{s=1}^{t}A_sA_s^\top,
\quad
\hat{\theta}_t=V_t(\lambda)^{-1}\sum_{s=1}^t A_s X_s
$$

위의 식에서 $\hat{\theta}_t$가 $$\theta^*$$에 대한 시점 $t$ 에서의 estimator 입니다.

다음 단계로 Confidence region을 구할 차례입니다.

#### Confidence Ellipsoid

$$V_t(\lambda)$$는 symmetric positive defitnie이고 직교대각화 가능하므로 eigen-decomposition을 통해 다음을 얻을 수 있습니다. $$\lambda_i$$는 $$V_t(\lambda)$$의 eigen value입니다.

$$
V_t(\lambda)=\lambda_1 v_1 v_1^\top + \cdots + \lambda_d v_d v_d^\top\\
\lambda_i= v_i^\top V_t(\lambda) v_i= v_i^\top (\lambda I+\sum_{s=1}^{t}A_sA_s^\top) v_i
= \lambda I + \sum_{s=1}^{t} \langle A_s, v_i \rangle^2
$$

고윳값 $$\lambda_i$$가 크다는 뜻은 $$t$$까지의 action들이 $$v_i$$와 유사한 방향으로 많이 선택되었다는 뜻입니다. 따라서 $V_t(\lambda)$의 eigenvalue가 큰 방향은 이미 많은 정보를 얻었음을 의미합니다. 따라서 아래와 같은 confidence ellipsoid를 구성하면 거의 관측하지 않은 방향에서는 ellipsoid의 폭이 크고, 관측이 일어났던 방향에서는 폭이 작은 confidence region을 만들 수 있게 됩니다. 이 region에서 폭이 큰 방향을 통해 선택하고 탐색하면 폭이 감소하게 될 것입니다.

$$
\mathcal{C}_t
=
\left\{
\theta\in\mathbb{R}^d:
\|\theta-\widehat{\theta}_{t-1}\|_{V_{t-1}(\lambda)}^2
\leq \beta_t
\right\}
$$

이제 $$\mathbb{P} \left(\forall t:\, \theta^* \in \mathcal{C}_t\right) \geq 1 - \delta$$를 만족시키는(높은 확률로 unknown parameter $$\theta^*$$를 포함하는) confidence radius $$\beta_t$$를 구해봅시다.

#### Martingale

$$\|S_t\|_{V_t(0)^{-1}}$$와 관련하여 아래의 첫번째 등식이 만족합니다. Fenchel duality의 특별한 케이스이며, 우변의 $$\max$$ 내부를 미분해 확인해보면 만족한다는 것을 확인할 수 있습니다. 식의 우변에 exponential을 취한 process $$M_t(x)$$를 생각하면, nonnegative supermatringale이 됨을 증명할 수 있습니다.

$$
\frac{1}{2} \|S_t\|_{V_t(0)^{-1}}^2=\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle
-\frac{1}{2}\|x\|_{V_t(0)}^2\right) \\
M_t(x)
=
\exp\left(
\langle x,S_t\rangle
-\frac{1}{2}\|x\|_{V_t(0)}^2
\right)
$$

<div class="math-box math-box--lemma" markdown="1">
<div class="math-box__title">Lemma</div>

$$M_t(x)=\exp\left(\langle x,S_t\rangle-\frac{1}{2}\|x\|_{V_t(\lambda)}^2\right)$$은 $$\mathbb{F}=(\mathcal{F}_t)_{t=1}^n$$-adapted nonnegative supermartingale이다.

<div class="math-box__title">Proof</div>

정의에서 $$M_t(x)$$는 $$\mathcal{F}_t$$-measurable.

$$\eta_t$$의 conditional 1-sub-gaussian 가정에서 $$\lambda=\langle x,A_t\rangle$$를 대입하면

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

$$M_{t-1}(x)$$가 $$\mathcal{F}_{t-1}$$-measurable이므로

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

추가로 $$S_0=0$$이고 $$V_0(\lambda)=\lambda I$$이므로 $$M_0(x)=\exp\left(-\frac{\lambda}{2}\|x\|_2^2\right)\leq1.$$

따라서 $$\mathbb{E}[M_t(x)]\leq\mathbb{E}[M_0(x)]\leq1$$이며 모든 고정된 $$x$$에 대해 $$M_t(x)$$는 nonnegative supermartingale.

</div>

$$M_t(x)$$는 $$V_t(0)$$으로 정의했지만, 위의 Lemma는 더 일반적인 $$V_t(\lambda)$$에서도 성립합니다. Cramer-Chernoff method로 아래와 같이 전개할 수 있습니다.

$$
\begin{aligned}
\mathbb{P}\left(\frac{1}{2} \|\hat{\theta}_t-\theta^*\|_{V_t(0)}^2 \geq \log (1/\delta)\right)
&=\mathbb{P}\left(\exp\left(\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle-\frac{1}{2}\|x\|_{V_t(0)}^2\right)\right) \geq 1/\delta \right)\\
&\leq \delta \mathbb{E}\left[\exp\left(\max_{x \in \mathbb{R}^d}\left(\langle x,S_t\rangle-\frac{1}{2}\|x\|_{V_t(0)}^2\right)\right)\right]\\
&=\delta \mathbb{E} \left[\max_{x \in \mathbb{R}^d} M_t(x) \right]
\end{aligned}
$$

고정된 $$x$$에서 $$ \mathbb{E}[M_t(x)] \leq \mathbb{E}[M_0(x)] \leq1 $$임을 이용하여 bound하고 싶지만, 아쉽게도 우리가 필요한 것은 고정된 $x$에서의 bound가 아닙니다. $$\mathbb{E} \left[\max_{x \in \mathbb{R}^d} M_t(x) \right] \geq \max_{x \in \mathbb{R}^d} \mathbb{E} \left[ M_t(x) \right]$$이기 때문에 우리가 원하던 것과 부등식 방향이 반대입니다.

#### Method of Mixtures

이 문제를 해결하는 방법이 method of mixtures입니다. $x$를 Gaussian distribution에서 뽑는다고 생각하고 $M_t(x)$를 평균내면 여전히 supermartingale의 성질은 유지하면서 closed form으로 계산할 수 있게 됩니다!

<div class="math-box math-box--lemma" markdown="1">
<div class="math-box__title">Lemma</div>

$$h$$가 $$\mathbb{R}^d$$ 위의 probability measure일 때, $$\overline{M}_t=\int_{\mathbb{R}^d}M_t(x)\,dh(x)$$는 $$\mathcal{F}_t$$-adapted nonnegative supermartingale이다.

<div class="math-box__title">Proof</div>

$$M_t(w, x)=\exp \left(\langle x, S_t(w)\rangle-\frac{1}{2} \|x\|_{V_t(0)(w)}^2\right)$$에서 $$\exp$$ 내부의 각 component가 $$\mathcal{F}_t \otimes \mathbb{R}^d$$-measurable함을 보일 수 있고, $$y \mapsto \exp(y)$$가 continous 하므로 $$M_t(w, x)$$는 $$\mathcal{F}_t \otimes \mathbb{R}^d$$-measurable.

sections lemma를 이용하면 $$\overline{M}_t=\int_{\mathbb{R}^d}M_t(x)\,dh(x)$$는 $$\mathcal{F}_t$$-measurable.

$$
\mathbb{E}[\overline{M}_t]
=
\int_{\mathbb{R}^d}\mathbb{E}[M_t(x)]\,dh(x)
\leq
\int_{\mathbb{R}^d}1\,dh(x)
=1.
$$

따라서 $$\overline{M}_t$$는 integrable.

$$\overline{M}_t$$의 supermartingale property가 성립하지 않는다고 가정하자. 어떤 $$\epsilon>0$$에 대해 아래를 만족하는 event $$A$$가 존재한다.

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

conditional expectation의 정의와 Fubini-Tonelli theorem을 통해

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

마지막 부등식은 각 $$M_t(x)$$가 supermartingale이며 $$A\in\mathcal{F}_{t-1}$$이기 때문에 성립한다.

이는 모순이므로

$$
\mathbb{E}[\overline{M}_t\mid\mathcal{F}_{t-1}]
\leq
\overline{M}_{t-1}
\quad\mathrm{a.s.}
$$

따라서 $$\overline{M}_t$$는 nonnegative supermartingale.

</div>

위의 lemma에 따라 $$\overline{M}_t$$는 nonnegative supermartingale이며 $$h(x)$$를 gaussian $$\mathcal{N}(0, H^{-1}=\lambda^{-1} I_d)$$ 로 두고 $$\overline{M}_t$$를 계산하면 다음과 같습니다.

$$
\begin{aligned}
\overline{M}_t
&=\int_{\mathbb{R}^d}M_t(x)\,dh(x)\\
&=\frac{1}{\sqrt{(2\pi)^d \det (H^{-1})}} \int_{\mathbb{R}^d} \exp \left( \langle x, S_t \rangle - \frac{1}{2} \|x\|_{V_t(0)}^2 - \frac{1}{2} \|x\|_{H}^2 \right)\,dx\\
&=\left(\frac{\det(H)}{\det (H + V_t(0))}\right)^{1/2} \exp \left(\frac{1}{2} \|S_t\|^2_{(H+V_t(0))^{-1}}\right)\\
&=\left(\frac{\lambda^d}{\det V_t(\lambda)}\right)^{1/2} \exp \left(\frac{1}{2} \|S_t\|^2_{V_t(\lambda)^{-1}}\right)
\end{aligned}
$$

<div class="math-box math-box--theorem" markdown="1">
<div class="math-box__title">Theorem (Nonnegative Supermartingale Maximal Inequality)</div>

$$M_t$$가 nonnegative supermartingale이면 모든 $$c>0$$에 대해

$$
\mathbb{P}\left(
\sup_{t\geq0}M_t\geq c
\right)
\leq
\frac{\mathbb{E}[M_0]}{c}.
$$

</div>

Maximal Inequality를 이용하여 $$c=1/\delta$$를 대입하면 다음을 얻게 됩니다.

$$
\mathbb{P}\left(\sup_{t\geq0} \overline{M}_t \geq \frac{1}{\delta}\right)
=\mathbb{P}\left(\sup_{t\geq0} \left(\frac{\lambda^d}{\det V_t(\lambda)}\right)^{1/2} \exp \left(\frac{1}{2} \|S_t\|^2_{V_t(\lambda)^{-1}}\right) \geq \frac{1}{\delta}\right)
\leq\delta\mathbb{E}[\overline{M}_0] \leq \delta.\\
\mathbb{P}\left(
\exists t \in \mathbb{N}:
\|S_t\|_{V_t^{-1}}^2
\geq
2\log\frac{1}{\delta}
+\log\frac{\det(V_t)}{\lambda^d}
\right)
\leq\delta.
$$

이제 $$\|\widehat{\theta}_t-\theta^*\|_{V_t}$$ 를 정리합니다.

$$
\begin{aligned}
\|\widehat{\theta}_t-\theta^*\|_{V_t}
&= \|V_t^{-1}\left(S_t-\lambda\theta^*\right)\|_{V_t}\\
&= \|S_t-\lambda\theta^*\|_{V_t^{-1}}\\
&\leq
\|S_t\|_{V_t^{-1}}
+\lambda\|\theta^*\|_{V_t^{-1}}\\
&\leq
\|S_t\|_{V_t^{-1}}
+\sqrt{\lambda}\|\theta^*\|_2.
\end{aligned}
$$

첫 부등식에서는 삼각부등식이 쓰였으며, 두 번째 부등식에서는 $$V_t^{-1} \preceq I$$가 사용되었습니다. $$\|\theta^*\|_2 \leq m_2$$임을 알고 있다 가정하면, 최종적으로 confidence radius $$\beta_t$$를 구할 수 있으며 이때의 $$\mathcal{C}_t$$에 속할 확률을 bound할 수 있습니다.

$$
\sqrt{\beta_t}
=
\sqrt{\lambda}m_2
+
\sqrt{
2\log\frac{1}{\delta}
+\log\frac{\det(V_{t-1})}{\lambda^d}
}.\\
\mathcal{C}_t
=
\left\{
\theta\in\mathbb{R}^d:
\|\theta-\widehat{\theta}_{t-1}\|_{V_{t-1}(\lambda)}^2
\leq \beta_t
\right\}.\\
\begin{aligned}
\mathbb{P}\left(\exists t \in \mathbb{N}: \theta^* \notin \mathcal{C}_t\right)
&=\mathbb{P}\left(\exists t \in \mathbb{N}: \|\theta^*-\widehat{\theta}_{t-1}\|_{V_{t-1}(\lambda)}^2 \leq \beta_t\right)\\
&\leq\mathbb{P}\left(\exists t \in \mathbb{N}: \|S_t\|_{V_t^{-1}} + \sqrt{\lambda} m_2 \leq \sqrt{\beta_t}\right)\\
&\leq \delta.
\end{aligned}
$$

이로써 $$\theta^*$$를 높은 확률로 포함하는 confidence region을 구했습니다!

#### LinUCB Algorithm

이제 LinUCB 알고리즘으로 빌드할 차례입니다. 각 시점에 action을 취할 때에 $$\operatorname*{argmax}_{a\in\mathcal{A}_t} \max_{\theta \in \mathcal{C}_t} \langle \theta, a \rangle$$를 구해야합니다. 이를 위해 $$\max_{\theta \in \mathcal{C}_t} \langle \theta, a \rangle$$를 정리합니다. 첫 부등식에서는 Weighted Cauchy-Schwarz Inequality가 쓰입니다. 앞으로 $$V_t(\lambda)$$를 $$V_t$$로 표기하겠습니다.

$$
\begin{aligned}
\max_{\theta \in \mathcal{C}_t} \langle \theta, a \rangle
&= \langle \hat{\theta}_t, a \rangle + \max_{\theta \in \mathcal{C}_t} \langle \theta - \hat{\theta}_t, a \rangle\\
&\leq \langle \hat{\theta}_t, a \rangle + \max_{\theta \in \mathcal{C}_t} \|\theta - \hat{\theta}_t\|_{V_t}\|a\|_{V_t^{-1}}\\
&\leq \langle \hat{\theta}_t, a \rangle + \sqrt{\beta_t} \|a\|_{V_{t-1}^{-1}}
\end{aligned}
$$

따라서 finite action set에서 다음의 LinUCB 알고리즘을 구성하게 됩니다.

1. $V_0=\lambda I$으로 초기화한다.
2. round $t$에서 $\widehat{\theta}_{t-1}$을 계산한다.
3. action $$A_t=\operatorname*{argmax}_{a\in\mathcal{A}_t} (\langle \hat{\theta}_t, a \rangle + \sqrt{\beta_t} \|a\|_{V_{t-1}^{-1}})$$를 선택하고 reward $$X_t$$를 관측한다.
4. $$V_t=V_{t-1}+A_tA_t^\top,\,\hat{\theta}_t=V_t^{-1} \sum_{s=1} ^t X_s A_s$$로 갱신한다.

드디어 Stochastic Linear Bandit에서의 LinUCB 알고리즘을 완성하였습니다! 이제 Regret을 분석할 차례입니다.

#### Regret Analysis

Regret 분석을 위해 다음을 가정합니다.

1. $1\leq\beta_1\leq\beta_2\leq\cdots\leq\beta_n$.
2. 모든 action vector에 대해 $$\|a\|_2\leq L$$.
3. 각 round의 reward gap은 최대 $$1$$.
4. Probability at least $$1-\delta$$로 모든 $$t\in[n]$$에 대해 $$\theta^*\in\mathcal{C}_t$$.

마지막 가정은 앞서 논의한 $$\beta_t$$를 선택하면 만족합니다.

Round $$t$$에서의 optimal action $$A_t^*=\operatorname*{argmax}_{a\in\mathcal{A}_t}\langle\theta^*,a\rangle$$, LinUCB가 고른 $$A_t$$에 대해 upper confidence bound를 달성하는 parameter를 $$\widetilde{\theta}_t=\operatorname*{argmax}_{\theta\in\mathcal{C}_t}\langle\theta,A_t\rangle$$ 라고 두면 $$\theta^*\in\mathcal{C}_t$$이므로 아래가 성립합니다. 따라서 매 round에서의 random pseudo regret $$r_t$$의 upper bound를 구할 수 있습니다.

$$
\langle\theta^*,A_t^*\rangle
\leq
\max_{\theta \in \mathcal{C}_t} \langle \theta, A_t^* \rangle
\leq
\max_{\theta \in \mathcal{C}_t} \langle \theta, A_t \rangle
=
\langle\widetilde{\theta}_t,A_t\rangle\\
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

한편 $$\widetilde{\theta}_t$$와 $$\theta^*$$는 모두 $$\mathcal{C}_t$$ 안에 있으므로, confidence ellipsoid의 중심인 $$\widehat{\theta}_{t-1}$$를 거쳐 triangle inequality를 적용하여 다음을 얻습니다.

$$
\|\widetilde{\theta}_t-\theta^*\|_{V_{t-1}}
\leq
\|\widetilde{\theta}_t-\widehat{\theta}_{t-1}\|_{V_{t-1}}
+
\|\widehat{\theta}_{t-1}-\theta^*\|_{V_{t-1}}
\leq 2\sqrt{\beta_t}.\\
r_t
\leq
2\sqrt{\beta_t}\|A_t\|_{V_{t-1}^{-1}}
$$

Reward gap이 최대 $$1$$이고 $$\beta_t\leq\beta_n$$이므로

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

따라서 남은 문제는 위의 제곱합을 bound하는 것입니다. 이를 위해 elliptical potential lemma를 사용합니다.

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

<div class="math-box__title">Proof</div>

$$V_t=V_{t-1}+A_tA_t^\top$$에 determinant를 적용하면

$$
\det(V_t)
=
\det(V_{t-1})
\left(
1+\|A_t\|_{V_{t-1}^{-1}}^2
\right).
$$

이를 모든 round에 대해 곱하면

$$
\frac{\det(V_n)}{\det(V_0)}
=
\prod_{t=1}^{n}
\left(
1+\|A_t\|_{V_{t-1}^{-1}}^2
\right).
$$

모든 $$u\geq0$$에 대해 $$\min\{1,u\}\leq2\log(1+u)$$ 가 항상 성립하므로, 양변에 log를 취한 뒤 합하면

$$
\det(V_t)
=
\det(V_{t-1})
\left(
1+\|A_t\|_{V_{t-1}^{-1}}^2
\right).
$$

</div>

Elliptical potential lemma를 앞의 regret 식에 대입하면 probability at least $$1-\delta$$로 아래를 얻습니다.

$$
\widehat{R}_n
\leq
\sqrt{
8n\beta_n
\log\frac{\det(V_n)}{\det(V_0)}
}
$$

이것이 바로 LinUCB의 기본 high-probability regret bound입니다!

추가로 앞서 두었던 벡터의 norm에 대한 bound를 이용해 regret을 표현해보겠습니다. $$V_0=\lambda I$$이고 $$\|A_t\|_2\leq L$$이므로

$$
\operatorname{tr}(V_n)
=
d\lambda+\sum_{t=1}^{n}\|A_t\|_2^2
\leq
d\lambda+nL^2.
$$

$$\det(V_0)=\lambda^d$$이고, 산술-기하 부등식을 통해

$$
\det(V_n)
\leq
\left(
\frac{\operatorname{tr}(V_n)}{d}
\right)^d
\leq
\left(
\lambda+\frac{nL^2}{d}
\right)^d.\\
\log\frac{\det(V_n)}{\det(V_0)}
\leq
d\log\left(
1+\frac{nL^2}{d\lambda}
\right).
$$

결국 아래와 같이 표현할 수 있습니다. 앞에서 구한 confidence radius $$\beta_t$$를 사용하고 $$\delta=1/n$$으로 선택하면 $$R_n=\tilde{O}(d \sqrt{n})$$임을 확인할 수 있습니다.

$$
\widehat{R}_n
\leq
\sqrt{
8dn\beta_n
\log\left(
1+\frac{nL^2}{d\lambda}
\right)
}.\\
\begin{aligned}
\sqrt{\beta_n}
&=
\sqrt{\lambda}m_2
+
\sqrt{
2\log\frac{1}{\delta}
+\log\frac{\det(V_{n-1})}{\lambda^d}
}\\
&\leq
\sqrt{\lambda}m_2
+
\sqrt{
2\log\frac{1}{\delta}
+d\log\frac{d \lambda + nL^2}{d \lambda}
}
\end{aligned}
.
$$

#### Implementation Notes

$V_t^{-1}$을 매 round 처음부터 계산하면 $O(d^3)$의 시간이 필요합니다. 이에 Sherman-Morrison formula를 사용하면 inverse update는 $O(d^2)$에 가능합니다. Action의 수가 $k$이면 각 action의 score 계산까지 포함하여 한 round의 시간 복잡도는 $O(kd^2)$입니다.

#### LinUCB vs UCB Experiment

LinUCB와 UCB를 구현하여 expected regret을 비교한 결과입니다.
왼쪽은 두 개의 action을 서로 독립인 one-hot feature로 두었을 때 두 action의 reward 차이에 따른 Expected regret을, 오른쪽은 dimension을 $$d=5$$로 고정하고 action의 수 $$k$$에 따른 Expected regret을 나타낸 그래프입니다.

<div class="experiment-figures">
  <img src="{{ site.baseurl }}/assets/images/19_8_a.png" alt="LinUCB와 UCB의 delta에 따른 expected regret 비교">
  <img src="{{ site.baseurl }}/assets/images/19_8_b.png" alt="LinUCB와 UCB의 action 수에 따른 expected regret 비교">
</div>

앞에서 살펴본 UCB의 regret bound는 $$R_n=\widetilde{O}(\sqrt{nk})$$입니다. UCB는 각 action을 독립된 arm으로 다루기 때문에 leading term이 arm의 수 $$k$$에 의존합니다. 반면 LinUCB는 action의 feature를 통해 하나의 parameter를 함께 학습하며, 앞에서 구한 regret bound는 $$\widetilde{O}(d\sqrt{n})$$입니다. 두 알고리즘의 구현은 [`algorithms.hpp`](https://github.com/danielhO9/Bandit-Algorithms/blob/main/include/algorithms.hpp)에서 확인할 수 있습니다.

#### Reference

Tor Lattimore and Csaba Szepesvári, *Bandit Algorithms*, Chapters 19–20.
