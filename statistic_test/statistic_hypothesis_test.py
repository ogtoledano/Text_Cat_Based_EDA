import numpy as np
from scipy import stats

from scipy.stats import mannwhitneyu,wilcoxon

def t_test_giving_samples(a,b):

    ## Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(a, b)
    print("t-statistic = " + str(t2))
    print("p-value = " + str(p2))

    alpha = 0.05
    if p2 > alpha:
        print('Independent samples have identical average (Accept H0)')
    else:
        print('Independent samples have not identical average (Reject H0)')

    print("\n")

def wilcoxon_non_parametric_test(a,b):
    # compare samples

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    # std deviation
    s = np.sqrt((var_a + var_b) / 2)

    print("mean for fist param:", a.mean())
    print("std dev for fist param: ", a.std())
    print("mean for second param", b.mean())
    print("std dev for second param: ", b.std())
    print("\n")

    stat, p = wilcoxon(a, b)
    print('Statistics= {}, p= {}'.format(stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Independent samples have identical average (Accept H0)')
    else:
        print('Independent samples have not identical average (Reject H0)')

    print("\n")

def mannwhitneyu_non_parametric_test(a,b):
    # compare samples
    stat, p = mannwhitneyu(a, b)
    print('Statistics= {}, p= {}'.format(stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Independent samples have identical average (Accept H0)')
    else:
        print('Independent samples have not identical average (Reject H0)')

    print("\n")


def normal_test(a,b):
    # Normal test using Shapiro-wilk
    tuple_a = stats.shapiro(a)
    tuple_b = stats.shapiro(b)
    print("Normal test using Shapiro-Wilk\nFor first sample: W-statistic={} and p-value={}".format(tuple_a[0], tuple_a[1]))
    print("For second sample: W-statistic={} and p-value={}".format(tuple_b[0], tuple_b[1]))
    print("\n")
    return tuple_a,tuple_b


def hypothesis_test(a,b):
    print("Basics statistic for two samples with equal lengths")
    N = len(a)
    print("N measures:", N)

    # Calculate the Standard Deviation
    # Calculate the variance to get the standard deviation

    # For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    # std deviation
    s = np.sqrt((var_a + var_b) / 2)

    print("mean for fist param:", a.mean())
    print("std dev for fist param: ", a.std())
    print("mean for second param", b.mean())
    print("std dev for second param: ", b.std())
    print("\n")

    tuple_a, tuple_b=normal_test(a,b)
    alpha = 0.05

    if tuple_a[1] >= alpha and tuple_b[1] >= alpha:
        print("Parametric test: T-Test")
        t_test_giving_samples(a,b)
    else:
        print("Non-parametric test: Mann-Whitney U")
        wilcoxon_non_parametric_test(a,b)

    print("-----------------------------------------------------------")
