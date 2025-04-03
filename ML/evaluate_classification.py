import os
import csv
from ML.CNN.CNN import CNN
from ML.NN.NN import NN
from ML.RF.RF import RF
from ML.SVM.SVM import SVM

types = ["base", "bagging"]
models = ["SVM", "RF", "NN", "CNN"]

for type in types:
    for model in models:
        for dataset in os.listdir("datasets"):
            # Initialize an empty dictionary to store cumulative results for averaging
            cumulative_results = {}
            all_softmax_predictions = {}  # Dictionary to store softmax predictions

            print(f"Processing dataset: {dataset} with model: {model} and type: {type}")

            # Loop 10 times to get multiple runs
            for i in range(10):
                # Set j to desired sample size
                j = 3000
                if model == "SVM":
                    result = SVM(dataset, type, j)
                elif model == "RF":
                    result = RF(dataset, type, j)
                elif model == "NN":
                    result = NN(dataset, type, j)
                elif model == "CNN":
                    result = CNN(dataset, type, j)

                # Accumulate the results for each sample size `j`
                if j not in cumulative_results:
                    # Initialize empty result dict for each sample size
                    cumulative_results[j] = {
                        'precision': 0,
                        'recall': 0,
                        'f1': 0,
                        'accuracy': 0,
                        'false_positive_rate': 0,
                        'false_negative_rate': 0,
                        'average_classification_time_per_sample': 0,
                        'count': 0  # To keep track of the number of iterations
                    }
                    all_softmax_predictions[j] = []  # Initialize list to collect softmax predictions

                # Accumulate results for this iteration
                cumulative_results[j]['precision'] += result['precision']
                cumulative_results[j]['recall'] += result['recall']
                cumulative_results[j]['f1'] += result['f1']
                cumulative_results[j]['accuracy'] += result['accuracy']
                cumulative_results[j]['false_positive_rate'] += result.get('false_positive_rate', 0)
                cumulative_results[j]['false_negative_rate'] += result.get('false_negative_rate', 0)
                cumulative_results[j]['average_classification_time_per_sample'] += result.get('average_classification_time_per_sample', 0)
                cumulative_results[j]['count'] += 1

                # Collect softmax predictions for this iteration
                all_softmax_predictions[j].append(result.get('softmax_predictions', []))

            # Now average the results for each sample size after 10 iterations
            averaged_results = {}
            for sample_size, values in cumulative_results.items():
                averaged_results[sample_size] = {
                    'precision': values['precision'] / values['count'],
                    'recall': values['recall'] / values['count'],
                    'f1': values['f1'] / values['count'],
                    'accuracy': values['accuracy'] / values['count'],
                    'false_positive_rate': values['false_positive_rate'] / values['count'],
                    'false_negative_rate': values['false_negative_rate'] / values['count'],
                    'average_classification_time_per_sample': values['average_classification_time_per_sample'] / values['count'],
                }

            # Ensure that the results directory exists
            os.makedirs(f'results/{dataset}', exist_ok=True)

            # Define the CSV file path and fieldnames for performance metrics
            csv_file_path = f'results/{dataset}/{dataset}_{model}_{type}.csv'
            fieldnames = ['samples', 'precision', 'recall', 'f1', 'accuracy', 'false_positive_rate', 'false_negative_rate', 'average_classification_time_per_sample']

            # Write the averaged results for each sample size
            with open(csv_file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                # Write averaged results for each sample size `j`
                for sample_size, result in averaged_results.items():
                    row = {
                        'samples': sample_size,
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'f1': result['f1'],
                        'accuracy': result['accuracy'],
                        'false_positive_rate': result['false_positive_rate'],
                        'false_negative_rate': result['false_negative_rate'],
                        'average_classification_time_per_sample': result['average_classification_time_per_sample'],
                    }
                    writer.writerow(row)

            print(f"Averaged results for {model} ({type}) saved to {csv_file_path}")

            # Optionally, save softmax predictions to separate CSV files
            softmax_file_path = f'results/{dataset}/{dataset}_{model}_{type}_softmax.csv'
            with open(softmax_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['samples', 'softmax_predictions'])

                # Write softmax predictions for each sample size `j`
                for sample_size, predictions in all_softmax_predictions.items():
                    # Flatten the list of softmax predictions for saving
                    flattened_predictions = [str(pred) for run_predictions in predictions for pred in run_predictions]
                    writer.writerow([sample_size] + flattened_predictions)

            print(f"Softmax predictions for {model} ({type}) saved to {softmax_file_path}")
