import numpy as np
from sklearn.decomposition import PCA

data = np.loadtxt('train_data.csv' , delimiter=',',)
X = data[:,:-2]
'''
#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=5)
pca.fit(X[:,:-2])

print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
print(pca.components_)
#pca = PCA(n_components=1, svd_solver='arpack')
#pca.fit(X[:,:-2])
#print(pca.explained_variance_ratio_)  

#print(pca.singular_values_)  
X1 = pca.transform(X[:,:-2])
'''
n_samples = X.shape[0]
pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)
    
X = np.concatenate((X_transformed,data[:,-2:]),axis = 1)
np.savetxt("train_data_pca.csv", X, delimiter=",")