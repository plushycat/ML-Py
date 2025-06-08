import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset and select numerical features
housing_df = fetch_california_housing(as_frame=True).frame
numerical_features = housing_df.select_dtypes(include=[np.number]).columns

# Plot histograms and boxplots using functions
def plot_distributions(data, features, plot_type, colors, figsize=(15, 10)):
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for i, feature in enumerate(features):
        ax = axes.flat[i] if hasattr(axes, 'flat') else axes
        if plot_type == 'hist':
            sns.histplot(data[feature], kde=True, bins=30, color=colors, ax=ax)
            ax.set_title(f'Distribution of {feature}')
        else:
            sns.boxplot(x=data[feature], color=colors, ax=ax)
            ax.set_title(f'Box Plot of {feature}')
    plt.tight_layout()
    plt.show()

# Create plots
plot_distributions(housing_df, numerical_features, 'hist', 'blue')
plot_distributions(housing_df, numerical_features, 'box', 'orange')

# Identify outliers
print("Outliers Detection:")
for feature in numerical_features:
    Q1, Q3 = housing_df[feature].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = sum((housing_df[feature] < Q1 - 1.5 * IQR) | (housing_df[feature] > Q3 + 1.5 * IQR))
    print(f"{feature}: {outliers} outliers")

# Print summary
print("\nDataset Summary:")
print(housing_df.describe())