import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, fisher_exact, power_divergence, chi2_contingency
from loguru import logger
from scipy.stats import expon, chi2
#import seaborn

# logger.info(chisquare([16, 18, 16, 14, 12, 12]))


##################################################################################
# Following code taken from: https://towardsdatascience.com/how-to-compare-two-or-more-distributions-9b06ee4d30bf
##################################################################################
# Init dataframe
df_bins = pd.DataFrame()
N = 512
n_control_group = 16 * 1024
q = 10
# np.random.seed(39)
comparison_ex = np.random.exponential(size=n_control_group)

mean = 1
stdev = 0.5

normal_distr_samples = np.random.normal(mean, stdev, N)
comparison_norm = np.random.normal(1, 0.52, n_control_group)

comparison = comparison_norm

# Generate bins from control group
_, bins = pd.qcut(comparison, q=q, retbins=True)
df_bins['bin'] = pd.cut(comparison, bins=bins).value_counts().index

# Apply bins to both groups
df_bins['comparison_observed'] = pd.cut(comparison, bins=bins).value_counts().values
df_bins['normal_distr_observed'] = pd.cut(normal_distr_samples, bins=bins).value_counts().values

# Compute expected frequency in the treatment group
df_bins['normal_distr_expected'] = df_bins['comparison_observed'] / np.sum(df_bins['comparison_observed']) * \
                                   np.sum(df_bins['normal_distr_observed'])

comp_series = pd.Series(comparison)
t1 = pd.DataFrame({'index':comp_series.index, 'values':comp_series.values, 'group': 'comparison'})
distr_series = pd.Series(normal_distr_samples)
t2 = pd.DataFrame({'index':distr_series.index, 'values':distr_series.values, 'group': 'distr'})
df_plotting = pd.concat([t1, t2])

#seaborn.boxplot(data=df_plotting, x='group', y='values')
plt.title("Boxplot")
plt.show()

#seaborn.histplot(data=df_plotting, x='values', hue='group', bins=50)
df_plotting.hist(bins=50)
plt.title("Histogram")
plt.show()

#seaborn.histplot(data=df_plotting, x='values', hue='group', bins=50, stat='density', common_norm=False)
#df_plotting.hist(bins=50)
#plt.title("Density Histogram")
#plt.show()

#seaborn.kdeplot(x='values', data=df_plotting, hue='group', common_norm=False)
#plt.title("Kernel Density Function")
#plt.show()

df_plotting.hist(bins=len(df_plotting))
#seaborn.histplot(x='values', data=df_plotting, hue='group', bins=len(df_plotting), stat="density",
 #            element="step", fill=False, cumulative=True, common_norm=False)
#plt.title("Cumulative distribution function")
#plt.show()

logger.info(f"\n{df_bins.to_markdown()}")

# stat, p_value = chisquare(df_bins['normal_distr_observed'])
# stat, p_value = chisquare(df_bins['normal_distr_observed'], df_bins['normal_distr_expected'])

res = chi2_contingency(df_bins[['normal_distr_observed', 'comparison_observed']], lambda_="log-likelihood")
# stat, p_value = power_divergence(df_bins['normal_distr_observed'],
#                                  df_bins['comparison_observed'],
#                                  lambda_="log-likelihood")

logger.info(f"Chi-squared Test: statistic={res.statistic:.4f} (What does this mean?)")
logger.info(f"Chi-squared Test: p-value={res.pvalue:.4f} - {100*res.pvalue:.2f}% ")
if res.pvalue < 0.05:
    logger.info(f"p-Value < 0.05 --> null hypothesis is rejected --> data is not correlated "
                f"(distributions are not the same)")
else:
    logger.info(f"p-Value > 0.05 --> null hypothesis can not be rejected --> data is correlated "
                f"(distributions are the same)")

##################################################################################
# code above taken from: https://towardsdatascience.com/how-to-compare-two-or-more-distributions-9b06ee4d30bf
##################################################################################
