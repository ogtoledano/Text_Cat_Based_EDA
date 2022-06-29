## Import the packages
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statistic_test.statistic_hypothesis_test import t_test_giving_samples,wilcoxon_non_parametric_test, mannwhitneyu_non_parametric_test,hypothesis_test


adam=[0.9835329341317365,0.9700598802395209,0.9775449101796407,0.9835329341317365,0.9730538922155688,0.9760479041916168,0.9760479041916168,0.9670658682634731,0.9745508982035929,0.9730538922155688,0.9745508982035929,0.9760479041916168,0.9820359281437125,0.9820359281437125,0.9700598802395209,0.9715568862275449,0.9805389221556886,0.9745508982035929,0.9775449101796407,0.9730538922155688,0.9760479041916168,0.9775449101796407,0.968562874251497,0.9715568862275449,0.9805389221556886,0.9790419161676647,0.9700598802395209,0.9745508982035929,0.9775449101796407,0.9760479041916168]
adam_emna=[0.9835329341317365,0.9760479041916168,0.9790419161676647,0.9835329341317365,0.9730538922155688,0.9775449101796407,0.9760479041916168,0.9715568862275449,0.9745508982035929,0.9730538922155688,0.9745508982035929,0.9805389221556886,0.9835329341317365,0.9835329341317365,0.9730538922155688,0.9730538922155688,0.9805389221556886,0.9745508982035929,0.9775449101796407,0.9745508982035929,0.9760479041916168,0.9775449101796407,0.9700598802395209,0.9730538922155688,0.9835329341317365,0.9790419161676647,0.9700598802395209,0.9745508982035929,0.9775449101796407,0.9760479041916168]

a=np.asarray(adam)
b=np.asarray(adam_emna)

print("Adam-emna Wilcoxon")
wilcoxon_non_parametric_test(a,b)

adam=[0.9775449101796407,0.9775449101796407,0.9745508982035929,0.9700598802395209,0.9730538922155688,0.9760479041916168,0.9850299401197605,0.9745508982035929,0.9805389221556886,0.9715568862275449,0.9745508982035929,0.9745508982035929,0.968562874251497,0.9775449101796407,0.9700598802395209,0.9730538922155688,0.9670658682634731,0.968562874251497,0.9775449101796407,0.9775449101796407,0.9760479041916168,0.9730538922155688,0.9730538922155688,0.9775449101796407,0.9835329341317365,0.9760479041916168,0.9775449101796407,0.9775449101796407,0.9760479041916168,0.9760479041916168]
adam_cumda=[0.9760479041916168,0.9790419161676647,0.9715568862275449,0.9730538922155688,0.9745508982035929,0.9745508982035929,0.9850299401197605,0.9760479041916168,0.9790419161676647,0.9715568862275449,0.9745508982035929,0.9745508982035929,0.9700598802395209,0.9775449101796407,0.968562874251497,0.9760479041916168,0.9745508982035929,0.9715568862275449,0.9745508982035929,0.9745508982035929,0.9790419161676647,0.968562874251497,0.9790419161676647,0.9775449101796407,0.9850299401197605,0.9760479041916168,0.9775449101796407,0.9760479041916168,0.9760479041916168,0.9760479041916168]
g=np.asarray(adam)
h=np.asarray(adam_cumda)

print("Adam-cumda Wilcoxon")
wilcoxon_non_parametric_test(g,h)

# Plot results

plt.boxplot([a,h,b],labels=["Adam","Variant 3", "Variant 4"])
plt.title('Comparison of measurements for the BBCN dataset')
plt.ylabel("Accuracy")
plt.savefig('bbcnewsAdam.eps', format='eps')
plt.show()


sgd=[0.968562874251497,0.9760479041916168,0.968562874251497,0.9670658682634731,0.9565868263473054,0.9610778443113772,0.9700598802395209,0.9745508982035929,0.9625748502994012,0.9775449101796407,0.9700598802395209,0.9730538922155688,0.9730538922155688,0.9745508982035929,0.9730538922155688,0.9655688622754491,0.9700598802395209,0.9700598802395209,0.9790419161676647,0.9775449101796407,0.9700598802395209,0.9745508982035929,0.9775449101796407,0.9790419161676647,0.9775449101796407,0.9745508982035929,0.9715568862275449,0.9730538922155688,0.9730538922155688,0.968562874251497]
sgd_emna=[0.968562874251497,0.9760479041916168,0.968562874251497,0.9670658682634731,0.9580838323353293,0.9610778443113772,0.9700598802395209,0.9760479041916168,0.9640718562874252,0.9775449101796407,0.9715568862275449,0.9730538922155688,0.9730538922155688,0.9745508982035929,0.9745508982035929,0.968562874251497,0.9700598802395209,0.9715568862275449,0.9835329341317365,0.9775449101796407,0.9730538922155688,0.9745508982035929,0.9790419161676647,0.9775449101796407,0.9775449101796407,0.9745508982035929,0.9715568862275449,0.9745508982035929,0.9730538922155688,0.9655688622754491]


c=np.asarray(sgd)
d=np.asarray(sgd_emna)

print("SGD-emna Wilcoxon")
wilcoxon_non_parametric_test(c,d)


sgd=[0.9655688622754491,0.9760479041916168,0.9655688622754491,0.9745508982035929,0.9670658682634731,0.9640718562874252,0.9700598802395209,0.9700598802395209,0.9760479041916168,0.9625748502994012,0.9760479041916168,0.9745508982035929,0.9670658682634731,0.9715568862275449,0.9730538922155688,0.9745508982035929,0.9565868263473054,0.9640718562874252,0.9715568862275449,0.968562874251497,0.9775449101796407,0.9715568862275449,0.9760479041916168,0.968562874251497,0.9760479041916168,0.9760479041916168,0.9775449101796407,0.9775449101796407,0.9775449101796407,0.9745508982035929]
sgd_cumda=[0.9640718562874252,0.9745508982035929,0.9730538922155688,0.9715568862275449,0.9715568862275449,0.9655688622754491,0.9715568862275449,0.9700598802395209,0.9760479041916168,0.9640718562874252,0.9730538922155688,0.9790419161676647,0.968562874251497,0.9715568862275449,0.9760479041916168,0.9730538922155688,0.9625748502994012,0.9640718562874252,0.9700598802395209,0.968562874251497,0.9775449101796407,0.9745508982035929,0.9775449101796407,0.968562874251497,0.9775449101796407,0.9730538922155688,0.9790419161676647,0.9775449101796407,0.9775449101796407,0.9730538922155688]

e=np.asarray(sgd)
f=np.asarray(sgd_cumda)

print("SGD-cumda Wilcoxon")
wilcoxon_non_parametric_test(e,f)

# Plot results

plt.boxplot([c,f,d],labels=["SGD","Variant 1", "Variant 2"])
plt.title('Comparison of measurements for the BBCN dataset')
plt.ylabel("Accuracy")
plt.savefig('bbcnewsSGD.eps', format='eps')
plt.show()