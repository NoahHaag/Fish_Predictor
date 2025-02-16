import os
import time

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import uniform, randint, loguniform
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, make_scorer, balanced_accuracy_score, top_k_accuracy_score, \
    confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight

# Load pre-processed data
X, y, feature_names, feature_sizes, groups = joblib.load('sklearn_model/processed_data.joblib')
print("Using pre-processed data")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'sklearn_model/label_encoder.joblib')
print("Images loaded and labels encoded.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# File to save the best model
model_save_path = 'sklearn_model/best_hist_model.joblib'

# Load the best existing model (if it exists)
best_model = None
best_accuracy = 0.0

if os.path.exists(model_save_path):
    best_model = joblib.load(model_save_path)
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)
    print(f"Loaded pre-trained model with test accuracy: {best_accuracy:.4f}")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),  # Replace with your classes
    y=y_train  # Replace with your target
)
class_weight_dict = dict(enumerate(class_weights))


# Define parameter distributions (using refined params if available)
param_dist = {
    'learning_rate': uniform(0.01, 0.2),
    'max_iter': randint(300, 1200),
    'max_leaf_nodes': randint(10, 20),
    'l2_regularization': loguniform(1e-6, 0.1).rvs(size=10),
    'min_samples_leaf': randint(5, 50),
    'max_bins': randint(32, 200),  # Reduce the number of bins for histogram approximation
}

start_time = time.time()
hgbc = HistGradientBoostingClassifier(random_state=42,
                                      early_stopping=True,
                                      n_iter_no_change=5,
                                      validation_fraction=0.4,
                                      class_weight=class_weight_dict)

StratKFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

balanced_accuracy_scorer = make_scorer(balanced_accuracy_score, adjusted=True)

# Initialize randomized search
search = RandomizedSearchCV(
    estimator=hgbc,
    param_distributions=param_dist,
    n_iter=5,  # Number of random samples
    scoring=balanced_accuracy_scorer,
    cv=StratKFold,
    n_jobs=-1,
    verbose=3,
    random_state=42,
    refit=True
)

# Perform randomized search
# Train initial model
search.fit(X_train, y_train)

# Extract only valid parameters
valid_params = {k: v for k, v in search.best_params_.items() if k in hgbc.get_params()}

# Initialize and train best model with valid parameters
hgbc_best = HistGradientBoostingClassifier(**valid_params)
hgbc_best.fit(X_train, y_train)

# Track training and validation accuracy at each boosting iteration
train_accuracies = []
test_accuracies = []

for y_pred_train, y_pred_test in zip(
        hgbc_best.staged_predict(X_train),  # Predictions on training set after each iteration
        hgbc_best.staged_predict(X_test)  # Predictions on validation set after each iteration
):
    train_accuracies.append(accuracy_score(y_train, y_pred_train))
    test_accuracies.append(accuracy_score(y_test, y_pred_test))

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy", marker='o')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Validation Accuracy", marker='o')

plt.title("HistGradientBoostingClassifier Training Progress")
plt.xlabel("Number of Estimators (Boosting Iterations)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig('Training Progress.jpeg')
plt.show()

best_params = search.best_params_
print(f"HistGradientBoostingClassifier best Parameters: {best_params}")

# Save processed data
# Evaluate the new model
new_model = search.best_estimator_
y_pred = new_model.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)

print(f"New model test accuracy: {new_accuracy:.4f}")

# Compare accuracies and save the model only if it outperforms the previous one
if best_model is not None:
    y_pred_saved = best_model.predict(X_test)
    saved_model_accuracy = accuracy_score(y_test, y_pred_saved)
    print(f"Previously saved HistGradientBoostingClassifier model test accuracy: {saved_model_accuracy:.4f}")
else:
    print("No previous HistGradientBoostingClassifier model available for comparison.")
    saved_model_accuracy = 0.0

if new_accuracy > saved_model_accuracy:
    joblib.dump(new_model, model_save_path)
    print(f"New HistGradientBoostingClassifier model saved with test accuracy: {new_accuracy:.4f}")
else:
    print(
        f"New HistGradientBoostingClassifier model not saved. Best model accuracy remains: {saved_model_accuracy:.4f}")

best_model = joblib.load('sklearn_model/best_hist_model.joblib')
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"Training took {minutes} minutes and {seconds} seconds.")

print("Pre-trained HistGradientBoostingClassifier saved.")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"HistGradientBoostingClassifier test accuracy: {accuracy:.4f}")

y_pred_proba = best_model.predict_proba(X_test)
top_k_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
print(f"HistGradientBoostingClassifier top-3 Accuracy: {top_k_acc * 100:.2f}%")

# further analysis

# Confusion Matrix
# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Get the class labels (fish species names)
class_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))

# Create a ConfusionMatrixDisplay object
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
disp.plot(cmap='OrRd', values_format='d', xticks_rotation='vertical')
plt.title('HistGradientBoostingClassifier Confusion Matrix')
plt.grid()
plt.savefig('HistGradientBoostingClassifier model confusion matrix.jpeg')
plt.show()

clf_report1 = classification_report(y_test, y_pred,
                                    target_names=class_labels)
print(clf_report1)

clf_report2 = classification_report(y_test, y_pred,
                                    target_names=class_labels,
                                    output_dict=True)
sns.heatmap(pd.DataFrame(clf_report2).iloc[:-1, :].T, annot=True)
plt.show()
