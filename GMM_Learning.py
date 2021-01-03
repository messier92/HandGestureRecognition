# https://relguzman.blogspot.com/2018/03/gaussian-mixture-models-explained.html

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd

sns.set_style("white")
# Return evenly spaced numbers over a specified interval
x = np.linspace(start=-10, stop=10, num=1000)
# A normal continuous random variable
# loc specifies the mean
# scale specifies specifies the standard dev 
y = stats.norm.pdf(x, loc=0, scale=1.5)

#plot it!
df = pd.read_csv("bimodal_example.csv")

# show the distribution of the data as a histogram
data = df.x.astype(float)
gfit_mean = np.mean(data)
gfit_sigma = np.std(data)

x = np.linspace(-6, 8, 200)
g_single = stats.norm(gfit_mean, gfit_sigma).pdf(x)
sns.distplot(data, bins=20, kde=False, norm_hist=True)
plt.plot(x, g_single, label='single gaussian')
plt.legend();
plt.show();
