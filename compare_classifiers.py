import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib.colors import ListedColormap

# ==== LOAD DATA ====
df = pd.read_csv("dataset.csv")

# ==== ENCODE CATEGORICAL FEATURES ====
label_encoders = {}
for col in ["breed", "color", "target"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("target", axis=1)
y = df["target"]

# ==== SCALE NUMERICAL FEATURES ====
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==== TRAIN TEST SPLIT ====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==== CLASSIFIERS ====
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
os.makedirs("visualizations", exist_ok=True)

# ==== TRAIN & EVALUATE ====
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

# ==== ACCURACY BAR CHART ====
plt.figure()
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/accuracy_bar.png")
plt.close()

# ==== CONFUSION MATRIX FOR RANDOM FOREST ====
rf_model = models["Random Forest"]
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("visualizations/confusion_matrix_rf.png")
plt.close()

# ==== DECISION TREE PLOT ====
plt.figure(figsize=(12, 8))
plot_tree(models["Decision Tree"],
          feature_names=X.columns,
          class_names=[str(c) for c in np.unique(y)],
          filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("visualizations/decision_tree.png")
plt.close()

# ==== DECISION BOUNDARY (for first 2 features) ====
X_2d = X_scaled.iloc[:, :2]  # Only first 2 features for visualization
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train_2d, y_train_2d)

h = 0.02
x_min, x_max = X_2d.iloc[:, 0].min() - 1, X_2d.iloc[:, 0].max() + 1
y_min, y_max = X_2d.iloc[:, 1].min() - 1, X_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ListedColormap(["red", "green"])

plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_2d.iloc[:, 0], X_2d.iloc[:, 1], c=y,
            edgecolor="k", cmap=cmap_bold)
plt.title("Decision Boundary (KNN, First 2 Features)")
plt.xlabel(X_2d.columns[0])
plt.ylabel(X_2d.columns[1])
plt.savefig("visualizations/decision_boundary_knn.png")
plt.close()

# ==== LOGISTIC REGRESSION PROBABILITY PLOT ====
feature_name = "weight"
X_feature = X_scaled[[feature_name]]
y_binary = y

log_reg = LogisticRegression()
log_reg.fit(X_feature, y_binary)

X_test_range = np.linspace(X_feature.min(), X_feature.max(), 300).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_test_range)[:, 1]  # Probability of class 1

plt.figure()
plt.scatter(X_feature, y_binary, c=y_binary, cmap=ListedColormap(["red", "green"]), edgecolors="k")
plt.plot(X_test_range, y_proba, color="blue", linewidth=2)
plt.title(f"Logistic Regression Probability Curve ({feature_name})")
plt.xlabel(feature_name)
plt.ylabel("Probability of Class 1")
plt.savefig("visualizations/logistic_regression_probability.png")
plt.close()

# ==== LINEAR REGRESSION PLOT ====
X_lin = X_scaled[["weight"]]
y_lin = X_scaled["height"]

lin_reg = LinearRegression()
lin_reg.fit(X_lin, y_lin)

X_plot = np.linspace(X_lin.min(), X_lin.max(), 300).reshape(-1, 1)
y_pred = lin_reg.predict(X_plot)

plt.figure()
plt.scatter(X_lin, y_lin, color="blue", alpha=0.6, edgecolors="k")
plt.plot(X_plot, y_pred, color="red", linewidth=2)
plt.title("Linear Regression: Height vs Weight")
plt.xlabel("Weight (scaled)")
plt.ylabel("Height (scaled)")
plt.savefig("visualizations/linear_regression_plot.png")
plt.close()

print("\n✅ All visualizations saved in 'visualizations/' folder.")
