# This is a sample Python script.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
data=np.random.randint(1,10,size=(4,3))

columns=["Feature_1","Feature_2","Feature_3"]
df=pd.DataFrame(data,columns=columns)
#mean,variance,standard dev
mean_values=df.mean()
variance_values=df.var()
std_dev_values=df.std()

#covariance
cov_matrix=df.cov()
corr_matrix=df.corr()
eig_values,eig_vectors=np.linalg.eig(cov_matrix)

print("Dataset:\n", df)
print("\nMean Values:\n", mean_values)
print("\nVariance Values:\n", variance_values)
print("\nStandard Deviation:\n", std_dev_values)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nCorrelation Matrix:\n", corr_matrix)
print("\nEigenvalues:\n", eig_values)
print("\nEigenvectors:\n", eig_vectors)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
df.hist()
plt.suptitle("Feature Distribution")


plt.subplot(1,2,2)
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f")
plt.show()
