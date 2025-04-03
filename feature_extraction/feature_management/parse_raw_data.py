import os
import pandas as pd
from data_management.open_file import openFile
from data_management.check_high_low import checkHighLow
from data_management.filter import filter
from feature_management.find_signals import findSignals
from feature_management.find_signal_features import find_signal_features
from feature_management.find_bits import find_bits
from ringing_management.find_ringing import find_ringing

def parse_raw_data(output_dir, sub_directory, label, w_type, s_type, f_type, ringing_limits):
    for file_name in os.listdir(sub_directory):
            if file_name.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(sub_directory, file_name)
                
                # Open file
                data_frame = openFile(file_path)

                if data_frame is None:
                    continue  # Skip to the next file if file loading fails

                # Determine CAN High / Low columns
                volt_high_col, volt_low_col = checkHighLow(data_frame)

                # Apply butterworth filter
                data_frame = filter(data_frame, volt_high_col, volt_low_col)

                # Locate signal sections
                if s_type == "ringing":
                    signals = findSignals(data_frame, ringing_limits)
                else:
                    signals = findSignals(data_frame)

                # finds features
                features, columns = find_signal_features(signals, w_type, s_type, f_type)

                features_df = pd.DataFrame(features, columns=columns)
                
                # Add the Label column
                features_df['Label'] = label

                # Define the output file path
                output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_processed.csv")
                
                # Save the DataFrame to a CSV file
                features_df.to_csv(output_file_path, index=False)

    