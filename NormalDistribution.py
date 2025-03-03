import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(42)
mu,sigma=5,1.5
feature_vector=np.random(mu,sigma,size=1000)

# compute PDF(Probability distribution function )& CDF (cumuative dist)
x_values=np.linspace(min(feature_vector),max(feature_vector),100)
pdf_values=stats.norm.pdf(x_values,mu,sigma)#pdf
cdf_values=stats.norm.cdf(x_values,mu,sigma)#cdf

#generate bivariate random variables(2 correlated features)
mean_vector=[5,10]
cov_matrix=[[2.0,1.5],
            [1.5,3.0]]
bivariate_data=np.random.multivariate_normal(mean_vector,cov_matrix,size=1000)
df_bivariate=pd.DataFrame(bivariate_data,columns=["Feature_1","Feature_2"])