import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from scipy.special import softmax

def SVM(dataset, type, size):
    # Load dataset directly
    data = pd.read_csv(f'path_to/datasets/{dataset}', skiprows=1)

    # Sample 'size' rows from the dataset
    sampled_data = data.dropna().sample(n=size, random_state=42)
    X = sampled_data.iloc[:, :-1].values
    y = sampled_data.iloc[:, -1].values

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode the labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model architecture with probability=True to get probability estimates
    svm_classifier = SVC(kernel='linear', random_state=42, probability=True, class_weight="balanced")

    # Number of models for ensemble approach
    n_models = 1 if type == "base" else 10

    # Define models list
    models = []

    # Create n models with bootstrapping
    for _ in range(n_models):
        X_resampled, y_resampled = resample(X_train, y_train, replace=True, n_samples=len(X_train))
        model = svm_classifier
        model.fit(X_resampled, y_resampled)
        models.append(model)

    # Initialize lists to collect predictions and true labels
    all_probabilities = []
    all_true_labels = []

    # Track time for predictions
    start_time = time.time()

    # Predict with all models and aggregate probability outputs
    for model in models:
        probabilities = model.predict_proba(X_test)  # Get probability estimates
        all_probabilities.append(probabilities)
        all_true_labels = y_test

    end_time = time.time()

    # Calculate the average classification time per sample
    total_time = end_time - start_time
    avg_classification_time = total_time / len(X_test)

    # Averaging the probabilities across models and applying softmax
    avg_probabilities = np.mean(all_probabilities, axis=0)
    softmaxed_probabilities = softmax(avg_probabilities, axis=1)

    # Final predictions are the class with the highest softmax probability
    final_predictions = np.argmax(softmaxed_probabilities, axis=1)

    # Compute performance metrics
    precision = precision_score(all_true_labels, final_predictions, average='macro')
    recall = recall_score(all_true_labels, final_predictions, average='macro')
    f1 = f1_score(all_true_labels, final_predictions, average='macro')
    accuracy = accuracy_score(all_true_labels, final_predictions)
    conf_matrix = confusion_matrix(all_true_labels, final_predictions)

    # Initialize FPR and FNR for each class
    fpr_list = []
    fnr_list = []

    # Calculate FPR and FNR for each class (one-vs-rest approach)
    for i in range(conf_matrix.shape[0]):
        # TP: diagonal element for class i
        tp = conf_matrix[i, i]

        # FN: sum of the i-th row minus TP
        fn = np.sum(conf_matrix[i, :]) - tp

        # FP: sum of the i-th column minus TP
        fp = np.sum(conf_matrix[:, i]) - tp

        # TN: sum of all elements minus the elements in the i-th row and i-th column
        tn = np.sum(conf_matrix) - (tp + fn + fp)

        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        fpr_list.append(fpr)
        fnr_list.append(fnr)

    # Average FPR and FNR across all classes
    avg_fpr = np.mean(fpr_list)
    avg_fnr = np.mean(fnr_list)

    # Final results including all softmax predictions
    final_results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'false_positive_rate': avg_fpr,
        'false_negative_rate': avg_fnr,
        'average_classification_time_per_sample': avg_classification_time,
        'softmax_predictions': softmaxed_probabilities  # Softmaxed probabilities for each sample
    }

    return final_results
