---
title: Notes on neural networks
numbersections: true
toc: true
papersize: a4
header-includes:
- \newcommand{\ce}{\mathrm{e}}
- \newcommand{\ol}{\overline}
---

# [Q-learning](https://en.wikipedia.org/wiki/Q-learning)

In Q-learning, we have a function $Q(s,a)$ which depends on the current state
$s$ and the action $a$ to perform and returns a real. We update it when we play
an action $a$ in a state $s$ with
$$
Q_{n+1}(s,a) = Q_n(s,a)+\alpha(r+\gamma\times\max_b Q_n(s\cdot a,b)-Q_n(s,a))
$$
where $s\cdot a$ is the state reached after performing the action $a$ and the
parameters are

- $\alpha$ is the _learning rate_ (how fast we perform the gradient descent,
  between $0$ and $1$, typically $0.1$),
- $\gamma$ is the _discount rate_ (whether we want it now or later, typically
  $0.9$),
- $r$ is the _reward_.

# [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

## For one neuron

A _neuron_ gives an output $y$ depending on a bunch of inputs $x_i$. The formula
is $$y=\phi(\sum_i w_ix_i)=\phi(x)$$ where $\phi$ is basically a function which
returns $0$ or $1$ depending on whether its input is below or above $0.5$ (some
people also add an offset called _bias_ to the sum, see below) and
$$x=\sum_iw_ix_i$$ is a weighted sum of the inputs. However, we want the
function $\phi$ to be continuous and differentiable, so a good approximation is
the _logistic_ or _sigmoid_ function $$\phi(x)=\frac 1{1+\ce^{-x}}$$ which maps
$\mathbb{R}$ to $[0,1]$ and satisfies $$\phi'(x)=\phi(x)(1-\phi(x))$$ We can add
a coefficient to the exponential, i.e. take $\phi(kx)$, to make the step sharper
or smoother, and other choices are popular such as $$\psi(x)=\tanh(x)$$ whose
derivative is $$\psi'(x)=1-\tanh^2(x)$$ This is basically the same since
$$\phi(x)=(1+\psi(x/2))/2$$ On a given input, suppose that we have an output $y$
whereas the expected output was $\ol y$. The error is $$E=L(y,\ol y)$$ where $L$
is the used norm. We typically use $$E=\frac 12(y-\ol y)^2$$ which is the square
of the euclidean norm up to a factor $1/2$.

We want to minimize this error, by tweaking the coefficients $w_i$. We thus
compute $$\frac{\partial E}{\partial w_i}=\frac{\partial E}{\partial
y}\frac{\partial y}{\partial x}\frac{\partial x}{\partial w_i}$$ With the usual
functions, we have

- $\frac{\partial E}{\partial y}=(y-\ol y)$ (the derivative of the square)
- $\frac{\partial y}{\partial x}=y(1-y)$ (the derivative of the sigmoid function)
- $\frac{\partial x}{\partial w_i}=x_i$

so that $$\frac{\partial E}{\partial w_i}=(y-\ol y)y(1-y)x_i$$

In order to minimize the error, we use a gradient descent and add, to each $w_i$
the quantity $$\Delta w_i=-\eta\frac{\partial E}{\partial w_i}=-\eta (y-\ol
y)y(1-y)x_i$$ where $\eta$ is the _learning rate_.

## For a neural network

Now suppose that we have $n$ layers of neurons: we write $y^k_j$ for the output
of the $j$-th neuron at round $k$, which is an input for round $k+1$:
$$y^{k+1}_j=\phi(\sum_i w^k_{ij}y^k_i)$$

As expected, we write $w^k_{ij}$ for the weight at round $k$ of the $i$-th input
in the $j$-th output and $$x^k_j=\sum_iw^k_{ij}y^k_i$$

The error we want to minimize is $$E=\frac 12\sum_j(y^n_j-\ol y^n_j)^2$$

We compute $$\frac{\partial E}{\partial y^k_j}=\sum_j\frac{\partial E}{\partial
y^{k+1}_j}\frac{\partial y^{k+1}_j}{\partial y^k_j}=\sum_j\frac{\partial
E}{\partial y^{k+1}_j}\frac{\partial y^{k+1}_j}{\partial x^k_j}\frac{\partial
x^k_j}{\partial y^k_j}=\sum_j\delta^k_jw^k_j$$ with $$\delta^k_j=\frac{\partial
E}{\partial y^{k+1}_j}\frac{\partial y^{k+1}_j}{\partial x^k_j}$$ where
$$\delta^{n-1}_j=(y^n_j-\ol y^n_j)y^n_j(1-y^n_j)$$ and
$$\delta^k_i=\sum_j\delta^{k+1}_j\frac{\partial x^{k+1}_j}{\partial
x^k_i}=\sum_j\delta^{k+1}_jw^{k+1}_{ij}\frac{\partial y^{k+1}_i}{\partial
x^{k+1}_i}$$ with $$\frac{\partial y^{k+1}_i}{\partial
x^{k+1}_i}=y^{k+1}_i(1-y^{k+1}_i)$$ Once the $y^k_i$ are computed, we can thus
compute the $\delta^k_i$ by "propagating backwards", i.e. computing the
$\delta^n_i$, the $\delta^{n-1}_i$, ..., up to the $\delta^0_i$.

We finally have $$\frac{\partial E}{\partial w^k_{ij}}=\sum_{j'}\frac{\partial
E}{\partial y^{k+1}_{j'}}\frac{\partial y^{k+1}_{j'}}{\partial x^k_{j'}}\frac{\partial
x^k_{j'}}{\partial w^k_{ij}}=\delta^k_jy^k_i$$

The change in $w^k_{ij}$ should be $$\Delta w^k_{ij}=-\eta\frac{\partial
E}{\partial w^k_{ij}}=-\eta\delta^k_jy^k_i$$

## Bias

If we consider neural networks with bias $$y^{k+1}_j=\phi(\sum_i
w^k_{ij}y^k_i+b^k_j)$$ we need to optimize those. This can be achieved in the
same way, computing that $$\frac{\partial E}{\partial b^k_i}=\delta^k_i$$ and
thus $$\Delta b^k_i=-\eta\delta^k_i$$
