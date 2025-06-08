import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and split data
X, y = fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f'\nCross-validation accuracy: {cross_val_score(model, X, y, cv=5).mean() * 100:.2f}%')

# Visualization
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f'True: {true}\nPred: {pred}', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()