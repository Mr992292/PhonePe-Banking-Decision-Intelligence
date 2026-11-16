
import numpy as np
import csv
import os

# Load dataset
data = np.genfromtxt(
    "data/records.csv",
    delimiter=",",
    skip_header=1,
    dtype=str
)

X = data[:, 1:8].astype(float)
y = data[:, 8].astype(int)

# Stratified training split
train_indices = []

for cls in [0, 1]:
    indices = np.where(y == cls)[0]
    split_point = (7 * len(indices)) // 10
    train_indices.extend(indices[:split_point])

train_indices = np.array(sorted(train_indices))

X_train = X[train_indices]
y_train = y[train_indices]

# Entropy
def entropy(labels):
    if len(labels) == 0:
        return 0

    probabilities = [
        np.mean(labels == 0),
        np.mean(labels == 1)
    ]

    return -sum(
        p * np.log2(p) for p in probabilities if p > 0
    )

# Gini
def gini(labels):
    if len(labels) == 0:
        return 0

    p1 = np.mean(labels == 1)
    p0 = 1 - p1

    return 1 - p0**2 - p1**2

root_entropy = entropy(y_train)
root_gini = gini(y_train)

conditions = {
    "S1": X_train[:, 5] == 1,
    "S2": X_train[:, 6] == 1,
    "S3": X_train[:, 2] > 0.5,
    "S4": X_train[:, 0] <= 24
}

results = []

for name, condition in conditions.items():

    yes_y = y_train[condition]
    no_y = y_train[~condition]

    n_yes = len(yes_y)
    n_no = len(no_y)

    positives_yes = np.sum(yes_y == 1)
    positives_no = np.sum(no_y == 1)

    weighted_entropy = (
        n_yes / len(y_train) * entropy(yes_y)
        + n_no / len(y_train) * entropy(no_y)
    )

    weighted_gini = (
        n_yes / len(y_train) * gini(yes_y)
        + n_no / len(y_train) * gini(no_y)
    )

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    results.append([
        name, n_yes, positives_yes,
        n_no, positives_no, info_gain, gini_gain
    ])

best_info = max(results, key=lambda x: x[5])
best_gini = max(results, key=lambda x: x[6])

os.makedirs("outputs", exist_ok=True)

with open("outputs/part4_splits.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "split", "n_yes", "positives_yes",
        "n_no", "positives_no", "info_gain", "gini_gain"
    ])

    for row in results:
        writer.writerow([
            *row[:5],
            f"{row[5]:.4f}",
            f"{row[6]:.4f}"
        ])

with open("outputs/part4_results.txt", "w") as f:
    f.write(f"root_entropy = {root_entropy:.4f}\n")
    f.write(f"root_gini = {root_gini:.4f}\n")
    f.write(f"best_split_by_info_gain = {best_info[0]}\n")
    f.write(f"best_split_by_gini_gain = {best_gini[0]}\n")

print("Root entropy:", round(root_entropy, 4))
print("Root gini:", round(root_gini, 4))
print("Best information gain split:", best_info[0])
print("Best Gini gain split:", best_gini[0])
print("Task 4.1 completed successfully!")


# ==========================================

# Create test data using the same stratified split
test_indices = []

for cls in [0, 1]:
    indices = np.where(y == cls)[0]
    cutoff = (7 * len(indices)) // 10
    test_indices.extend(indices[cutoff:])

test_indices = np.array(sorted(test_indices))

X_test = X[test_indices]
y_test = y[test_indices]

# TASK 4.2 - MODEL VOTING
# ==========================================

# Standardise using training data
mean = X_train.mean(axis=0)
sd = X_train.std(axis=0, ddof=1)

X_train_std = (X_train - mean) / sd
X_test_std = (X_test - mean) / sd

# Logistic Regression from Part 2
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

weights = np.zeros(7)
bias = 0.0

for epoch in range(1000):
    p = sigmoid(X_train_std @ weights + bias)
    error = p - y_train
    weights -= 0.1 * (X_train_std.T @ error) / len(y_train)
    bias -= 0.1 * np.mean(error)

logistic_prob = sigmoid(X_test_std @ weights + bias)
logistic_pred = (logistic_prob >= 0.3).astype(int)

# KNN with best K = 5
K = 5

knn_prob = []

for query in X_test_std:
    distances = np.sqrt(
        np.sum((X_train_std - query) ** 2, axis=1)
    )

    nearest = np.argsort(distances, kind="stable")[:K]
    positive_count = np.sum(y_train[nearest])

    knn_prob.append(positive_count / K)

knn_prob = np.array(knn_prob)
knn_pred = (knn_prob >= (K + 1) / (2 * K)).astype(int)

# Decision Tree Stump - Best split S4
condition_train = X_train[:, 0] <= 24
condition_test = X_test[:, 0] <= 24

stump_rate_yes = np.mean(y_train[condition_train])
stump_rate_no = np.mean(y_train[~condition_train])

stump_prob = np.where(
    condition_test,
    stump_rate_yes,
    stump_rate_no
)

stump_pred = (stump_prob >= 0.3).astype(int)

# Hard voting
hard_vote = (
    logistic_pred + knn_pred + stump_pred >= 2
).astype(int)

# Soft voting
soft_prob = (
    logistic_prob + knn_prob + stump_prob
) / 3

soft_vote = (soft_prob >= 0.3).astype(int)

# Evaluation function
def evaluate_model(predictions):
    tp = int(np.sum((predictions == 1) & (y_test == 1)))
    fp = int(np.sum((predictions == 1) & (y_test == 0)))
    tn = int(np.sum((predictions == 0) & (y_test == 0)))
    fn = int(np.sum((predictions == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return tp, fp, tn, fn, precision, recall, f1

models = {
    "logistic": logistic_pred,
    "knn": knn_pred,
    "stump": stump_pred,
    "hard_vote": hard_vote,
    "soft_vote": soft_vote
}

with open("outputs/part4_voting.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "model", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1"
    ])

    for name, predictions in models.items():
        metrics = evaluate_model(predictions)

        writer.writerow([
            name,
            *metrics[:4],
            *[f"{v:.4f}" for v in metrics[4:]]
        ])

print("\nTASK 4.2 COMPLETED")
print("Stump yes rate:", round(stump_rate_yes, 4))
print("Stump no rate:", round(stump_rate_no, 4))

for name, predictions in models.items():
    print(name, evaluate_model(predictions))

print("Saved outputs/part4_voting.csv")

# TASK 4.3 - ADABOOST

y_boost = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

weights = np.ones(8) / 8
boost_results = []
alphas = []
errors = []

for round_no, h in enumerate([h1, h2], start=1):
    before = weights.copy()
    error = np.sum(weights[h != y_boost])
    alpha = 0.5 * np.log((1-error)/error)

    unnormalised = weights * np.exp(-alpha*y_boost*h)
    weights = unnormalised / np.sum(unnormalised)

    errors.append(error)
    alphas.append(alpha)

    for i in range(8):
        boost_results.append([
            round_no, i+1, before[i],
            unnormalised[i], weights[i]
        ])

ensemble_score = alphas[0]*h1 + alphas[1]*h2
ensemble_prediction = np.where(ensemble_score >= 0, 1, -1)
ensemble_accuracy = np.mean(ensemble_prediction == y_boost)

with open("outputs/part4_adaboost.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "round", "point", "weight_before",
        "weight_unnormalised", "weight_after"
    ])
    for row in boost_results:
        writer.writerow([
            row[0], row[1],
            *[f"{v:.4f}" for v in row[2:]]
        ])

print("Task 4.3 saved successfully!")
