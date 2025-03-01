# This is a sample Python script.
import numpy as np
import pandas as pd
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

np.random.seed(42)
data=np.random.randint(1,10,size=(4,3))

columns=["Feature_1","Feature_2","Feature_3"]
df=pd.DataFrame(data,columns=columns)
#mean
mean_values=df.mean()
#variance
variance_values=df.var()
#covariance
cov_matrix=df.cov()

print("Dataset:\n", df)
print("\nMean Values:\n", mean_values)
print("\nVariance Values:\n", variance_values)
print("\nCovariance Matrix:\n", cov_matrix)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
