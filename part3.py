import csv
import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

# Load data
with open("data/records.csv", newline="") as f:
    rows = list(csv.DictReader(f))

X = np.array([
    [float(r[c]) for c in [
        "account_age_months",
        "failed_txn_count_90d",
        "credit_utilisation_ratio",
        "avg_monthly_inflow_inr_thousands",
        "repayment_score"
    ]] for r in rows
])

y = np.array([int(r["loan_default"]) for r in rows])
value = np.array([float(r["monthly_spend_inr_thousands"]) for r in rows])

n = len(X)
mean = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)
Z = (X - mean) / sd

results = {}

# TASK 3.1 - K MEANS
def kmeans(data, k):
    step = len(data) // k
    centroids = data[np.arange(k) * step].copy()

    while True:
        distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for j in range(k):
            members = data[labels == j]
            if len(members):
                new_centroids[j] = members.mean(axis=0)

        if np.array_equal(labels, getattr(kmeans, "previous", np.full(len(data), -1))):
            centroids = new_centroids
            break

        kmeans.previous = labels.copy()
        centroids = new_centroids

    kmeans.previous = np.full(len(data), -1)

    distances = ((data - centroids[labels]) ** 2).sum(axis=1)
    return labels, centroids, distances.sum()

wcss = {}

for k in range(2, 7):
    labels, centroids, score = kmeans(Z, k)
    wcss[k] = score
    results[f"wcss_k{k}"] = score

elbow = 5

for k in range(3, 6):
    if wcss[k] - wcss[k+1] < 0.5 * (wcss[k-1] - wcss[k]):
        elbow = k
        break

results["elbow_k"] = elbow

# TASK 3.3 - THREE SEGMENTS
labels, centroids, _ = kmeans(Z, 3)

clusters = []

for j in range(3):
    indices = np.where(labels == j)[0]
    clusters.append((X[indices, 2].mean(), indices))

clusters.sort(key=lambda item: item[0])

segment_rows = []

with open("outputs/part3_segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "segment", "size", "mean_X1", "mean_X2", "mean_X3",
        "mean_X4", "mean_X5", "outcome_rate", "mean_value"
    ])

    for name, (_, indices) in zip(["A", "B", "C"], clusters):
        subset = X[indices]
        size = len(indices)

        results[f"size_segment_{name}"] = size

        row = [
            name, size,
            *subset.mean(axis=0),
            y[indices].mean(),
            value[indices].mean()
        ]

        segment_rows.append((name, indices))
        writer.writerow([
            row[0], row[1],
            *[f"{v:.4f}" for v in row[2:]]
        ])

# TASK 3.4 - HIERARCHICAL CLUSTERING
names = ["E1", "E2", "E3", "E4", "E5", "E6"]

D = np.array([
    [0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0]
], dtype=float)

hierarchy = []

for method in ["single", "complete", "average"]:
    groups = [[i] for i in range(6)]
    heights = []

    for merge_no in range(1, 6):
        candidates = []

        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                distances = [D[a,b] for a in groups[i] for b in groups[j]]

                if method == "single":
                    height = min(distances)
                elif method == "complete":
                    height = max(distances)
                else:
                    height = np.mean(distances)

                candidates.append((height, i, j))

        height, i, j = min(candidates)
        a = "+".join(names[x] for x in groups[i])
        b = "+".join(names[x] for x in groups[j])

        hierarchy.append([method, merge_no, a, b, height])
        heights.append(height)

        groups[i] = sorted(groups[i] + groups[j])
        del groups[j]

    gaps = np.diff(heights)
    largest_gap = int(np.argmax(gaps))
    results[f"{method}_clusters_at_cut"] = 5 - largest_gap

with open("outputs/part3_hierarchical.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])

    for row in hierarchy:
        writer.writerow([
            *row[:4], f"{row[4]:.4f}"
        ])

# TASK 3.5 - PCA
cov = Z.T @ Z / (n - 1)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

for j in range(5):
    idx = np.argmax(np.abs(eigenvectors[:, j]))
    if eigenvectors[idx, j] < 0:
        eigenvectors[:, j] *= -1

total = eigenvalues.sum()
scores = Z @ eigenvectors

results["eigenvalue_sum"] = total

cumulative = 0

for j in range(5):
    results[f"eigenvalue_{j+1}"] = eigenvalues[j]
    results[f"explained_ratio_{j+1}"] = eigenvalues[j] / total
    cumulative += eigenvalues[j] / total
    results[f"cumulative_ratio_{j+1}"] = cumulative

results["components_for_80pct"] = int(np.argmax(np.cumsum(eigenvalues / total) >= 0.8) + 1)

for j in range(5):
    results[f"pc1_loading_X{j+1}"] = eigenvectors[j, 0]
    results[f"pc2_loading_X{j+1}"] = eigenvectors[j, 1]

for i in range(3):
    results[f"row{i+1}_pc1"] = scores[i, 0]
    results[f"row{i+1}_pc2"] = scores[i, 1]

for name, indices in segment_rows:
    results[f"mean_pc1_segment_{name}"] = scores[indices, 0].mean()

# SAVE RESULTS
with open("outputs/part3_results.txt", "w") as f:
    for key, val in results.items():
        if isinstance(val, (int, np.integer)):
            f.write(f"{key} = {val}\n")
        else:
            f.write(f"{key} = {val:.4f}\n")

print("PART 3 COMPLETED")
print("WCSS:", wcss)
print("Elbow K:", elbow)
print("Segment sizes:", [len(x[1]) for x in clusters])
print("PCA eigenvalue sum:", total)
