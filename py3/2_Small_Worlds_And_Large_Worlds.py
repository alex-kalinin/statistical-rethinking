# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Chapter 2 | Small Worlds and Large Worlds

import numpy as np
class Object(object): pass
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy.stats import binom
from scipy import stats
import pandas as pd
from typing import List
import pymc3 as pm

# ## 2.3
#
# Define Grid:

NUM = 20
p_grid = np.linspace(start=0, stop=1, num=NUM)
p_grid

# Define prior:

prior = np.repeat(1, NUM)
prior

likelihood = binom.pmf(k=6, n=9, p=p_grid)
likelihood

plt.plot(p_grid, likelihood)

unstd = Object()
unstd.posterior = likelihood * prior
unstd.posterior

posterior = unstd.posterior / sum(unstd.posterior)
posterior

plt.plot(p_grid, posterior)


# ## 2.5 - Different Priors
#
# ### Step

def post(prior_, num_):
    p_grid = np.linspace(start=0, stop=1, num=num_)
    likelihood = binom.pmf(k=6, n=9, p=p_grid)
    unst_posterior = likelihood * prior_
    posterior = unst_posterior / unst_posterior.sum()
    return posterior


step = Object()
step.prior = (p_grid >= 0.5).astype(np.int)
plt.plot(p_grid, step.prior)

step.post = post(step.prior, NUM)
plt.plot(p_grid, step.post)

# ### Peak

peak = Object()
peak.prior = np.exp(-5 * np.abs(p_grid - 0.5))
plt.plot(p_grid, peak.prior)

peak.likelihood = binom.pmf(k=6, n=9, p=p_grid)
peak.unstd_post = peak.likelihood * peak.prior
peak.post = peak.unstd_post / np.sum(peak.unstd_post)
plt.plot(p_grid, peak.post)

# # Quadratic Approximation

from scipy.stats import norm
normal = Object()
normal.grid = np.linspace(-3, 3, 100)
normal.norm = norm.pdf(normal.grid)
plt.plot(normal.grid, normal.norm)

plt.plot(normal.grid, np.log(normal.norm))

# ## MAP

# See the problem in https://www.probabilitycourse.com/chapter9/9_1_2_MAP_estimation.php

_m = Object()
_m.x = np.linspace(-0.1, 1.1, 100)
_m.f_x = 2 * _m.x**2 * (1 - _m.x)**2
plt.plot(_m.x, _m.f_x)

# As the analytical solution shows we have three points where the derivative is zero: $x=0$, $x=1$, and $x=\frac{1}{2}$. The value $x=\frac{1}{2}$ maximizes the function, and is the answer.

# # 2.6

# +
_26 = Object()
_26.data = np.repeat((0, 1), (3, 6))
with pm.Model() as _26.na:
    _26.p = pm.Uniform('p', 0, 1)
    _26.w = pm.Binomial('w', n=len(_26.data), p=_26.p, observed=_26.data.sum())
    _26.mean_p = pm.find_MAP()
    _26.std_q = ((1/pm.find_hessian(_26.mean_p, vars=[_26.p]))**0.5)[0]
    
_26.mean_p['p'], _26.std_q
# -

# Assuming the posterior is Gaussian, it's maximized at $0.67$ and its standard deviation is $0.16$.

# 89% confidence interval:

_26.norm_dist = stats.norm(_26.mean_p['p'], _26.std_q)
_26.z = stats.norm.ppf([(1 - .89) / 2, 1 - (1 - 0.89) / 2])
print("89% confidence interval:", _26.mean_p['p'] + _26.std_q * _26.z)

# # Medium
# ## 2M1

_2m1 = Object()
_2m1.NUM = 100
_2m1.p_grid = np.linspace(0, 1, _2m1.NUM)
_2m1.prior = np.repeat(1, _2m1.NUM)

# ### Item 1 | W, W, W

_2m1.item_1 = Object()
_2m1.item_1.likelihood = binom.pmf(k=3, n=3, p=_2m1.p_grid)
_2m1.item_1.unstd_post = _2m1.prior * _2m1.item_1.likelihood
_2m1.item_1.post = _2m1.item_1.unstd_post / _2m1.item_1.unstd_post.sum()

_2m1.item_1.fig = plt.figure(figsize=(20, 4))
_2m1.item_1.axes = _2m1.item_1.fig.add_subplot(131)
_2m1.item_1.axes.set_title("Prior")
_2m1.item_1.axes.set_ylim([0, 1.1])
_2m1.item_1.axes.plot(_2m1.p_grid, _2m1.prior)
_2m1.item_1.axes = _2m1.item_1.fig.add_subplot(132)
_2m1.item_1.axes.set_title("Likelihood")
_2m1.item_1.axes.set_ylim([0, 1.1])
_2m1.item_1.axes.plot(_2m1.p_grid, _2m1.item_1.likelihood)
_2m1.item_1.axes = _2m1.item_1.fig.add_subplot(133)
_2m1.item_1.axes.set_title("Posterior")
_2m1.item_1.axes.plot(_2m1.p_grid, _2m1.item_1.post)
print("Confirm Posterior sums up to 1:", _2m1.item_1.post.sum())

# ### Item 2 | W, W, W, L

# +
_2m1.item_2 = Object()
_2m1.item_2.likelihood = binom.pmf(k=3, n=4, p=_2m1.p_grid)
_2m1.item_2.unstd_post = _2m1.prior * _2m1.item_2.likelihood
_2m1.item_2.post = _2m1.item_2.unstd_post / _2m1.item_2.unstd_post.sum()

_2m1.item_2.fig = plt.figure(figsize=(20, 4))
_2m1.item_2.axes = _2m1.item_2.fig.add_subplot(1, 3, 1)
_2m1.item_2.axes.set_title("Prior")
_2m1.item_2.axes.set_ylim([0, 1.1])
_2m1.item_2.axes.plot(_2m1.p_grid, _2m1.prior)
_2m1.item_2.axes = _2m1.item_2.fig.add_subplot(1, 3, 2)
_2m1.item_2.axes.set_title("Likelihood")
_2m1.item_2.axes.plot(_2m1.p_grid, _2m1.item_2.likelihood)
_2m1.item_2.axes = _2m1.item_2.fig.add_subplot(1, 3, 3)
_2m1.item_2.axes.set_title("Posterior")
_2m1.item_2.axes.plot(_2m1.p_grid, _2m1.item_2.post)
# -

# ### Item 3 | L, W, W, L, W, W, W

# +
_2m1.item_3 = Object()
_2m1.item_3.likelihood = binom.pmf(k=5, n=7, p=_2m1.p_grid)
_2m1.item_3.unstd_post = _2m1.prior * _2m1.item_3.likelihood
_2m1.item_3.post = _2m1.item_3.unstd_post / _2m1.item_3.unstd_post.sum()

_2m1.item_3.fig = plt.figure(figsize=(20, 4))
_2m1.item_3.axes = _2m1.item_3.fig.add_subplot(1, 3, 1)
_2m1.item_3.axes.set_title("Prior")
_2m1.item_3.axes.set_ylim([0, 1.1])
_2m1.item_3.axes.plot(_2m1.p_grid, _2m1.prior)

_2m1.item_3.axes = _2m1.item_3.fig.add_subplot(1, 3, 2)
_2m1.item_3.axes.set_title("Likelihood")
_2m1.item_3.axes.plot(_2m1.p_grid, _2m1.item_3.likelihood)

_2m1.item_3.axes = _2m1.item_3.fig.add_subplot(1, 3, 3)
_2m1.item_3.axes.set_title("Posterior")
_2m1.item_3.axes.plot(_2m1.p_grid, _2m1.item_3.post)
# -

# ## 2M2

_2m2 = Object()
_2m2.p_grid = _2m1.p_grid
# 0 if p < 0.5, and a const if p >= 0.5
_2m2.prior = (_2m2.p_grid >= 0.5).astype(np.int)


# +
def _2m2_plot(fig, index, title, x, y, ylim=None):
    axes = fig.add_subplot(1, 3, index)
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(ylim)
    axes.plot(x, y)

def _2m2_posterior(w, n, obj):
    likelihood = binom.pmf(k=w, n=n, p=obj.p_grid)
    unstd_post = obj.prior * likelihood
    post = unstd_post / unstd_post.sum()
    
    fig = plt.figure(figsize=(20, 4))
    _2m2_plot(fig, 1, "Prior", _2m2.p_grid, _2m2.prior)
    _2m2_plot(fig, 2, "Likelihood", _2m2.p_grid, likelihood)
    _2m2_plot(fig, 3, "Likelihood", _2m2.p_grid, post)


# -

# ### Item 1 | W, W, W

_2m2_posterior(w=3, n=3, obj=_2m2)

# ### Item 2 | W, W, W, L

_2m2_posterior(w=3, n=4, obj=_2m2)

# ### Item 3 | L, W, W, L, W, W, W

_2m2_posterior(w=5, n=7, obj=_2m2)


# ## 2M4 
# Simulate the result: $\frac{2}{3}$

# +
class Card:
    def __init__(self, sides_black):
        self.sides_black = sides_black
        
_2m4 = Object()
_2m4.cards = [Card([True, True]), Card([False, True]), Card([False, False])]

def _2m4_simulate(obj):
    obj.total_trials = 0
    obj.back_is_black = 0
    for i in range(20000):
        card = np.random.choice(obj.cards)
        # Choose random side, 0 or 1
        side = np.random.randint(0, 2)
        # Need a Black as a top side
        if not card.sides_black[side]:
            continue
        obj.total_trials += 1
        other_side = (side + 1) % 2
        if card.sides_black[other_side]:
            obj.back_is_black += 1
_2m4_simulate(_2m4)
print(_2m4.back_is_black / _2m4.total_trials, "vs", 2/3)
# -

# ## 2M5
#
# The correct answer is $\frac{4}{5} = 0.80$

_2m5 = Object()
_2m5.cards = [Card([True, True]), Card([False, True]), Card([False, False]), Card([True, True])]
_2m4_simulate(_2m5)
print(_2m5.back_is_black / _2m5.total_trials, "vs", 4/5)


