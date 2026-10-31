"""
Store purchase classifier using K-Nearest Neighbors (KNN).

This script:
1. Builds a synthetic dataset of 150 shoppers (age, salary, purchase).
2. Trains a KNN classifier to predict whether a customer purchases (1) or not (0).
3. Saves the trained model and the feature scaler as binary files.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# All outputs are written next to this script so the files stay together.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "store_purchase_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "knn_purchase_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "purchase_scaler.pkl")

# Fixed seed so the same 150 rows and the same train/test split are produced
# every time the script is run.
RANDOM_SEED = 42
N_RECORDS = 150


def generate_purchase_data(n_records=N_RECORDS, seed=RANDOM_SEED):
    """
    Create synthetic store-purchase records.

    Age and salary are drawn from realistic ranges. Purchase (0 or 1) is
    generated from a simple rule that depends on both features, plus a little
    noise, so the label is related to the inputs but not perfectly predictable.
    """
    rng = np.random.default_rng(seed)

    # Age in years, between 18 and 70.
    age = rng.integers(18, 71, size=n_records)

    # Annual salary in currency units, between 20,000 and 150,000.
    salary = rng.integers(20_000, 150_001, size=n_records)

    # A score that rises with salary and falls slightly with age.
    # Higher scores are more likely to become a purchase (1).
    score = (salary / 10_000) - (age / 25)

    # Convert the score into a probability with a logistic curve, then
    # sample a hard 0/1 label. This keeps the column binary, as requested.
    probability = 1 / (1 + np.exp(-(score - 6.5)))
    purchase = (rng.random(n_records) < probability).astype(int)

    data = pd.DataFrame(
        {
            "age": age,
            "salary": salary,
            "purchase_probability": purchase,
        }
    )
    return data


def main():
    # -----------------------------------------------------------------------
    # Block 1: Build and save the dataset
    # 150 rows, three columns: age, salary, and purchase_probability.
    # purchase_probability is 0 (did not purchase) or 1 (purchased).
    # -----------------------------------------------------------------------
    data = generate_purchase_data()
    data.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(data)} records to {CSV_PATH}")
    print(data["purchase_probability"].value_counts().sort_index().to_string())
    print()

    # -----------------------------------------------------------------------
    # Block 2: Separate features from the label
    # X holds the inputs the model will see (age and salary).
    # y holds the target the model must predict (0 or 1).
    # -----------------------------------------------------------------------
    feature_columns = ["age", "salary"]
    X = data[feature_columns]
    y = data["purchase_probability"]

    # -----------------------------------------------------------------------
    # Block 3: Train / test split
    # 80% of the rows are used to fit the model.
    # 20% are held out so we can measure accuracy on unseen customers.
    # stratify keeps the 0/1 mix similar in both sets.
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # -----------------------------------------------------------------------
    # Block 4: Scale the features
    # KNN measures distance. Salary values (tens of thousands) would otherwise
    # dominate age values (tens). StandardScaler puts both columns on a
    # comparable scale (mean 0, standard deviation 1) using the training set
    # only, then applies that same scaling to the test set.
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # Block 5: Train the KNN classifier
    # k=5 means each new customer is labeled by the majority purchase decision
    # among the 5 closest customers in the training set.
    # weights="distance" lets nearer neighbors count more than farther ones.
    # -----------------------------------------------------------------------
    model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    model.fit(X_train_scaled, y_train)

    # -----------------------------------------------------------------------
    # Block 6: Evaluate on the held-out customers
    # Accuracy is the share of test rows predicted correctly.
    # The confusion matrix and classification report show how well the model
    # separates purchasers (1) from non-purchasers (0).
    # -----------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, digits=3))

    # -----------------------------------------------------------------------
    # Block 7: Save the model and the scaler in binary format
    # joblib writes Python objects to a binary .pkl file.
    # Both files are required later: new ages and salaries must be scaled
    # with the same scaler before they are passed to the model.
    # -----------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")

    # -----------------------------------------------------------------------
    # Block 8: Show how to score one new customer
    # Load the binary files, scale the new row, then predict 0 or 1.
    # -----------------------------------------------------------------------
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    new_customer = pd.DataFrame([{"age": 34, "salary": 82000}])
    new_customer_scaled = loaded_scaler.transform(new_customer)
    prediction = loaded_model.predict(new_customer_scaled)[0]
    print()
    print("Example customer: age=34, salary=82000")
    print(f"Predicted purchase (0=no, 1=yes): {int(prediction)}")


if __name__ == "__main__":
    main()
