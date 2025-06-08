import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load and split data
df = load_breast_cancer()
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and predict
clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Predict a new sample
sample = X_test[0].reshape(1, -1)
cls = ["Malignant", "Benign"][clf.predict(sample)[0]]
print(f"Predicted Class for the new sample: {cls}")

# Plot tree
plt.figure(figsize=(20, 12), dpi=150)
plot_tree(clf, filled=True, feature_names=df.feature_names,
        class_names=df.target_names, fontsize=9)
plt.title("Decision Tree - Breast Cancer Dataset", fontsize=14)
plt.show()
