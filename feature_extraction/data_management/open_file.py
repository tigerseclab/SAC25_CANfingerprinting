import pandas as pd

def openFile(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=2)  # Skip the first two rows to get the data
        df.columns = ['Time', 'CAN_H', 'CAN_L']  # Rename columns to match expected names
        return df
    except Exception as e:
        print(f"Error in loading file {file_path}: {e}")
        return None
