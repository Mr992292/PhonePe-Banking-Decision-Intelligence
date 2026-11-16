
import csv
import numpy as np

# Load dataset
with open("data/records.csv", "r") as f:
    rows = list(csv.DictReader(f))

columns = [
    "account_age_months",
    "failed_txn_count_90d",
    "credit_utilisation_ratio",
    "avg_monthly_inflow_inr_thousands",
    "repayment_score",
    "irregular_salary_flag",
    "autopay_flag"
]

X = np.array([[float(r[c]) for c in columns] for r in rows])
y = np.array([int(r["loan_default"]) for r in rows])

# Stratified split
train_idx, test_idx = [], []

for outcome in [0, 1]:
    idx = np.where(y == outcome)[0]
    cutoff = 7 * len(idx) // 10
    train_idx.extend(idx[:cutoff])
    test_idx.extend(idx[cutoff:])

train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standardisation
mean = X_train.mean(axis=0)
sd = X_train.std(axis=0, ddof=1)

X_train = (X_train - mean) / sd
X_test = (X_test - mean) / sd

# Logistic regression
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

weights = np.zeros(7)
bias = 0.0
learning_rate = 0.1

checkpoints = {}

for epoch in range(1001):
    probabilities = sigmoid(X_train @ weights + bias)

    if epoch in [0, 1, 10, 100, 1000]:
        checkpoints[epoch] = bce(y_train, probabilities)

    if epoch < 1000:
        error = probabilities - y_train
        weights -= learning_rate * (X_train.T @ error) / len(y_train)
        bias -= learning_rate * np.mean(error)

# Test predictions
test_probabilities = sigmoid(X_test @ weights + bias)

print("PART 2 - TASK 2.2")
print("--------------------------")

for epoch, loss in checkpoints.items():
    print(f"Loss epoch {epoch}: {loss:.4f}")

for i, weight in enumerate(weights, 1):
    print(f"Weight X{i}: {weight:.4f}")

print(f"Bias: {bias:.4f}")
print(f"Test BCE: {bce(y_test, test_probabilities):.4f}")

print("\nTraining rows:", len(y_train))
print("Training positives:", np.sum(y_train))
print("Testing rows:", len(y_test))
print("Testing positives:", np.sum(y_test))
