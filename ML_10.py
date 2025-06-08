import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# Load and preprocess data
X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y, y_kmeans))
print("\nClassification Report:\n", classification_report(y, y_kmeans))

# PCA for visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)
centers_pca = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)

# Create dataframe
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = y

# Plot settings
def plot_clusters(hue, title, palette, add_centroids=False):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=hue,
                    palette=palette, s=100, edgecolor='black', alpha=0.7)
    if add_centroids:
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                    s=200, c='black', marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=hue)
    plt.tight_layout()
    plt.show()

# Visualizations
plot_clusters('Cluster', 'K-Means Clustering', 'Set1')
plot_clusters('True Label', 'True Labels of Breast Cancer Dataset', 'coolwarm')
plot_clusters('Cluster', 'K-Means Clustering with Centroids', 'Set1', add_centroids=True)