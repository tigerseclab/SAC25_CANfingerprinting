import os
import csv
import shutil
import glob
import random
import numpy as np
from feature_management.parse_raw_data import parse_raw_data
from data_management.open_file import openFile
from data_management.check_high_low import checkHighLow
from ringing_management.find_ringing_limit import find_ringing_limit
from data_management.filter import filter

def extract_features(directory):
    # Waveform, signal, and data features to be analyzed
    waveform_types = ["raw", "differential"]
    signal_types = ["combined", "rising", "falling", "steady", "ringing"]
    feature_types = ["combined", "voltage", "frequency"]

    # Extract ringing limits  
    for directory_name in os.listdir(directory):
        sub_directory = os.path.join(directory, directory_name)
        temp_limits = []
        used_files = []
        while len(temp_limits) < 10:
            file_name = random.choice(glob.glob(f"{sub_directory}/*.csv"))
            if file_name in used_files:
                continue
            else:
                file_path = os.path.join(sub_directory, file_name)

                # Open file
                data_frame = openFile(file_path)

                if data_frame is None:
                    continue  # Skip to the next file if file loading fails

                # Determine CAN High / Low columns
                volt_high_col, volt_low_col = checkHighLow(data_frame)

                # Apply butterworth filter
                data_frame = filter(data_frame, volt_high_col, volt_low_col)

                # Find limit from bit in frame
                ringing_limit = find_ringing_limit(data_frame)

                if ringing_limit is not None:  # Skip None values
                    temp_limits.append(ringing_limit)

                # Add file to list to prevent resampling
                used_files.append(file_name)

        # Find average limits for ecu   
        ringing_limits.append({
            'high_ringing_start': np.nanmean([p['high_ringing_start'] for p in temp_limits if p and p['high_ringing_start'] is not None]),
            'high_ringing_end': np.nanmean([p['high_ringing_end'] for p in temp_limits if p and p['high_ringing_end'] is not None]),
            'low_ringing_start': np.nanmean([p['low_ringing_start'] for p in temp_limits if p and p['low_ringing_start'] is not None]),
            'low_ringing_end': np.nanmean([p['low_ringing_end'] for p in temp_limits if p and p['low_ringing_end'] is not None])
        })
        temp_limits.clear()

    # Find average limits for all ecus
    ringing_limits = [
        np.nanmean([p['high_ringing_start'] for p in ringing_limits if p['high_ringing_start'] is not None]),
        np.nanmean([p['high_ringing_end'] for p in ringing_limits if p['high_ringing_end'] is not None]),
        np.nanmean([p['low_ringing_start'] for p in ringing_limits if p['low_ringing_start'] is not None]),
        np.nanmean([p['low_ringing_end'] for p in ringing_limits if p['low_ringing_end'] is not None])
    ]

    # Extract all features
    for w_type in waveform_types:
        for s_type in signal_types:
            for f_type in feature_types:
                print(f'{w_type}_{s_type}_{f_type}')

                for directory_name in os.listdir(directory):
                    label = str(directory_name).split('_')[1]

                    # Create the output directory if it doesn't exist
                    output_dir = f'./extracted_data/ecu_{label}'
                    os.makedirs(output_dir, exist_ok=True)

                    sub_directory = os.path.join(directory, directory_name)

                    parse_raw_data(output_dir, sub_directory, label, w_type, s_type, f_type, ringing_limits)

                # Concatenate all processed files into a single CSV
                all_data = []
                for directory_name in os.listdir("extracted_data"):
                    sub_directory = os.path.join("extracted_data", directory_name)
                    first_file = True
                    for file_name in os.listdir(sub_directory):
                        file_path = os.path.join(sub_directory, file_name)
                        with open(file_path, 'r') as file:
                            csv_reader = csv.reader(file)

                            if first_file == False:
                                next(csv_reader)

                            for row in csv_reader:
                                if any(field.strip() for field in row):
                                    all_data.append(row)

                            first_file = False

                with open(f"path_to/ML/datasets/dataset_{w_type}_{s_type}_{f_type}.csv", 'w') as f:
                    write = csv.writer(f)
                    write.writerows(all_data)

                shutil.rmtree("./extracted_data")


directory = input("Input directory: ")
extract_features(directory)
