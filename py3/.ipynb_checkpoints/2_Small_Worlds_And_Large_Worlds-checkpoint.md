---
jupyter:
  jupytext:
    formats: ipynb,md,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
class Object(object): pass
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import binom
from scipy import stats
import pandas as pd
from typing import List
import pymc3 as pm
```

## 2.3

Define Grid:

```python
NUM = 20
p_grid = np.linspace(start=0, stop=1, num=NUM)
p_grid
```

Define prior:

```python
prior = np.repeat(1, NUM)
prior
```

```python
likelihood = binom.pmf(k=6, n=9, p=p_grid)
likelihood
```

```python
plt.plot(p_grid, likelihood)
```

```python
unstd = Object()
unstd.posterior = likelihood * prior
unstd.posterior
```

```python
posterior = unstd.posterior / sum(unstd.posterior)
posterior
```

```python
plt.plot(p_grid, posterior)
```

## 2.5 - Different Priors

### Step

```python
def post(prior_, num_):
    p_grid = np.linspace(start=0, stop=1, num=num_)
    likelihood = binom.pmf(k=6, n=9, p=p_grid)
    unst_posterior = likelihood * prior_
    posterior = unst_posterior / sum(unst_posterior)
    return posterior
```

```python
step = Object()
step.prior = (p_grid >= 0.5).astype(np.int)
plt.plot(p_grid, step.prior)
```

```python
step.post = post(step.prior, NUM)
plt.plot(p_grid, step.post)
```

### Peak

```python
peak = Object()
peak.prior = np.exp(-5 * np.abs(p_grid - 0.5))
plt.plot(p_grid, peak.prior)
```

```python
peak.likelihood = binom.pmf(k=6, n=9, p=p_grid)
peak.unstd_post = peak.likelihood * peak.prior
peak.post = peak.unstd_post / np.sum(peak.unstd_post)
plt.plot(p_grid, peak.post)
```

# Quadratic Approximation

```python
from scipy.stats import norm
normal = Object()
normal.grid = np.linspace(-3, 3, 100)
normal.norm = norm.pdf(normal.grid)
plt.plot(normal.grid, normal.norm)
```

```python
plt.plot(normal.grid, np.log(normal.norm))
```

## MAP


See the problem in https://www.probabilitycourse.com/chapter9/9_1_2_MAP_estimation.php

```python
_m = Object()
_m.x = np.linspace(-0.1, 1.1, 100)
_m.f_x = 2 * _m.x**2 * (1 - _m.x)**2
plt.plot(_m.x, _m.f_x)
```

As the analytical solution shows we have three points where the derivative is zero: $x=0$, $x=1$, and $x=\frac{1}{2}$. The value $x=\frac{1}{2}$ maximizes the function, and is the answer.


# 2.6

```python
_26 = Object()
_26.data = np.repeat((0, 1), (3, 6))
with pm.Model() as _26.na:
    _26.p = pm.Uniform('p', 0, 1)
    _26.w = pm.Binomial('w', n=len(_26.data), p=_26.p, observed=_26.data.sum())
    _26.mean_p = pm.find_MAP()
    _26.std_q = ((1/pm.find_hessian(_26.mean_p, vars=[_26.p]))**0.5)[0]
    
_26.mean_p['p'], _26.std_q
```

Assuming the posterior is Gaussian, it's maximized at $0.67$ and its standard deviation is $0.16$.


89% confidence interval:

```python
_26.norm_dist = stats.norm(_26.mean_p['p'], _26.std_q)
_26.z = stats.norm.ppf([(1 - .89) / 2, 1 - (1 - 0.89) / 2])
print("89% confidence interval:", _26.mean_p['p'] + _26.std_q * _26.z)
```

```python

```
