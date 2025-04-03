import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import joblib

def NN(dataset, type, size):
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
    y_one_hot = to_categorical(y)

    # Calculate class weights based on the labels (to mimic class_weight="balanced")
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(enumerate(class_weights))

    # Define model architecture
    def create_nn_model():
        model = Sequential([
            Dense(128, input_dim=X.shape[1], activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(y_one_hot.shape[1], activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Number of models for ensemble approach
    n_models = 1 if type == "base" else 10

    # Define models list
    models = []

    # Measure training time
    training_start_time = time.time()

    # Create n models with bootstrapping
    for _ in range(n_models):
        X_resampled, y_resampled = resample(X, y_one_hot, replace=True, n_samples=len(X))
        model = create_nn_model()
        # Train model using class weights
        model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, verbose=1, class_weight=class_weights_dict)
        models.append(model)

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save models and scaler
    for i, model in enumerate(models):
        model.save(f'nn_model_{i}.h5')
        joblib.dump(scaler, 'scaler.pkl')

    # Split the data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Initialize lists to collect predictions and true labels
    all_predictions = []
    all_true_labels = []
    all_softmax_probabilities = []

    # Measure prediction time
    prediction_start_time = time.time()

    # Predict with all models and aggregate predictions
    for model in models:
        predictions = model.predict(X_test)  # Get softmax predictions (probabilities)
        predicted_labels = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_labels)
        all_true_labels.extend(np.argmax(y_test, axis=1))
        all_softmax_probabilities.append(predictions)  # Collect softmax probabilities

    prediction_end_time = time.time()

    # Calculate the average classification time per sample
    total_prediction_time = prediction_end_time - prediction_start_time
    avg_classification_time = total_prediction_time / len(X_test)

    # Compute performance metrics
    precision = precision_score(all_true_labels, all_predictions, average='macro')
    recall = recall_score(all_true_labels, all_predictions, average='macro')
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_true_labels, all_predictions)
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)

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
        'softmax_predictions': np.vstack(all_softmax_probabilities)  # Softmax probabilities for each sample
    }

    return final_results
