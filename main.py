import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import KernelPCA

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
data = np.random.randint(1, 10, size=(4, 3))
columns = ["Feature_1", "Feature_2", "Feature_3"]
df = pd.DataFrame(data, columns=columns)

# Mean, Variance, Standard Deviation
mean_values = df.mean()
variance_values = df.var()
std_dev_values = df.std()

# Covariance and Correlation matrices
cov_matrix = df.cov()
corr_matrix = df.corr()

# Eigenvalues and Eigenvectors
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

# Print results
print("Dataset:\n", df)
print("\nMean Values:\n", mean_values)
print("\nVariance Values:\n", variance_values)
print("\nStandard Deviation:\n", std_dev_values)
print("\nCovariance Matrix:\n", cov_matrix)
print("\nCorrelation Matrix:\n", corr_matrix)
print("\nEigenvalues:\n", eig_values)
print("\nEigenvectors:\n", eig_vectors)

# Apply KernelPCA to transform the data into a higher dimension
# Using RBF kernel (Radial Basis Function) for non-linear transformation
kernel_pca = KernelPCA(kernel="rbf", gamma=1)  # Adjust gamma for different transformations
df_kernel = kernel_pca.fit_transform(df)

# Print the transformed data
print("\nTransformed Data (higher dimension via Kernel PCA):\n", df_kernel)

# Plot the feature distribution of the original dataset
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df.hist()
plt.suptitle("Feature Distribution")

# Plot the correlation heatmap
plt.subplot(1, 2, 2)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
