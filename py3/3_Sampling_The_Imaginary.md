---
jupyter:
  jupytext:
    formats: ipynb,md,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%matplotlib inline
import numpy as np
from scipy import stats
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

class Object(object): pass
```

# 3.1 Sampling from a grid-approximate posterior


## Code 3.2

```python
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, p_grid.shape[0])
likelihood = stats.binom.pmf(n=9, k=6, p=p_grid)

posterior = prior * likelihood
posterior /= posterior.sum()

_32 = Object()
_32.fig = plt.figure(figsize=(20, 4))
_32.axes = _32.fig.add_subplot(131)
_32.axes.plot(p_grid, prior)
_32.axes.set_title("Prior")

_32.axes = _32.fig.add_subplot(132)
_32.axes.plot(p_grid, likelihood)
_32.axes.set_title("Likelihood")

_32.axes = _32.fig.add_subplot(133)
_32.axes.plot(p_grid, posterior)
_32.axes.set_title("Posterior")
```

## Code 3.3

```python
samples = np.random.choice(p_grid, size=1000, p=posterior, replace=True)
```

```python
plt.plot(samples, marker='o', linestyle='None')
```

## Kernel Density Estimation

https://en.wikipedia.org/wiki/Kernel_density_estimation

```python
def dens(x, samples, smoothing_h=None):
    phi = lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    # Rule of thumb smoothing, see https://en.wikipedia.org/wiki/Kernel_density_estimation
    h = smoothing_h if smoothing_h is not None \
        else np.std(samples) * (4 / 3 / samples.shape[0])**(1/5)
    result = []
    for x_i in x:
        result.append(np.mean(phi((x_i - samples) / h)) / h)
    return np.array(result)
```

Smoothing too low--$0.01$:

```python
plt.plot(p_grid, dens(p_grid, samples, 0.01))
```

Smoothing too high--$1.0$:

```python
plt.plot(p_grid, dens(p_grid, samples, 1))
```

Rule of thumb: $(\frac{4}{3n})^\frac{1}{5}\hat\sigma \approx 1.06\hat\sigma n^{-1/5}$, where $\hat\sigma$ is the sample standard error.

```python
plt.plot(p_grid, dens(p_grid, samples))
```

Increase number of samples:

```python
%%time
plt.plot(p_grid, dens(p_grid, np.random.choice(p_grid, size=int(1e5), p=posterior, replace=True)))
```

# 3.2. Sampling to Summarize


### Code 3.6

```python
# Add up posterior probability where  p < 0.5
np.sum(posterior[p_grid < 0.5])
```

Same result, but use sample:


### Code 3.7

```python
np.sum(samples < 0.5), np.sum(samples < 0.5) / samples.shape[0]
```

### Code 3.8

```python
np.sum((samples > 0.5) & (samples < 0.75)) / samples.shape[0]
```

### Code 3.9 - Percentile

```python
def quantile(arr, perc_list):
    def single_quantile(arr, perc):
        k = (arr.shape[0] - 1) * perc
        f = np.floor(k)
        c = np.ceil(k)
        result = None
        if f == c:
            result = arr[int(k)]
        else:
            d0 = arr[int(f)] * (c - k)
            d1 = arr[int(c)] * (k - f)
            result = d0 + d1
        return result
    
    arr = np.sort(arr)
    
    if isinstance(perc_list, list):
        perc_list = np.array(perc_list)
        
    if isinstance(perc_list, np.ndarray):
        result = np.zeros_like(perc_list)
        for i, p in enumerate(perc_list):
            result[i] = single_quantile(arr, p)
    else:
        result = single_quantile(arr, perc_list)
    return result
```

```python
quantile(samples, 0.8)
```

### Code 3.10

```python
quantile(samples, np.array([0.1, 0.9]))
```

### Code 3.11

```python
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(k=3, n=3, p=p_grid)
posterior = prior * likelihood
posterior /= np.sum(posterior)
samples = np.random.choice(p_grid, size=1000, replace=True, p=posterior)
```

```python
ret = quantile(samples, np.array([0.25, 0.75]))
print(ret)
```

```python
indices = np.where((p_grid >= ret[0]) & (p_grid <= ret[1]))
x = p_grid[indices]
f = posterior[indices]

fig = plt.figure(figsize=(14, 4))
axes = fig.add_subplot(121)
axes.plot(p_grid, posterior)
axes.fill_between(x, f)
axes.set_title("50% percentile")
axes.set_xlabel("proportion of water (p)")
axes.set_ylabel("Density")
fig.suptitle("Figure 3.3")

axes = fig.add_subplot(122)
axes.plot(p_grid, posterior)
axes.set_xlabel("proportion of water (p)")
axes.set_ylabel("Density")
axes.set_title("50% HPDI")
```

```python

```
