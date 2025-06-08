import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_and_plot(X_test, y_test, y_pred, xlabel, ylabel, title, is_poly=False):
    plt.scatter(X_test, y_test, color="blue", label="Actual", alpha=0.7)
    if is_poly:
        plt.scatter(X_test, y_pred, color="red", label="Predicted", alpha=0.7)
    else:
        plt.plot(X_test, y_pred, color="red", label="Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(title)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}\n")

def linear_regression_boston():
    url1="https://raw.githubusercontent.com/plushycat/ML-Datasets/main/Boston.csv"
    df = pd.read_csv(url1)
    X = df[["rm"]]
    y = df["medv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    evaluate_and_plot(X_test, y_test, model.predict(X_test),
                    "Average number of rooms per dwelling (RM)",
                    "Median value of homes ($1000s)",
                    "Linear Regression - Boston Housing")

def polynomial_regression_auto_mpg():
    url2 = "https://raw.githubusercontent.com/plushycat/ML-Datasets/main/MPG.csv"
    df = pd.read_csv(url2).dropna()
    X = df[["displacement"]]
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()).fit(X_train, y_train)
    evaluate_and_plot(X_test, y_test, model.predict(X_test),
                    "Displacement", "Miles per gallon (MPG)",
                    "Polynomial Regression - Auto MPG", is_poly=True)

linear_regression_boston()
polynomial_regression_auto_mpg()