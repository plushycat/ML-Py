import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data and perform PCA
iris = load_iris()
data_reduced = PCA(n_components=2).fit_transform(iris.data)

# Create DataFrame with reduced dimensions and labels
df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])
df['Label'] = iris.target

# Plot results
plt.figure(figsize=(8, 6))
for i, label_name in enumerate(iris.target_names):
    mask = df['Label'] == i
    plt.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], 
                label=label_name, color=['r', 'g', 'b'][i])
    
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()