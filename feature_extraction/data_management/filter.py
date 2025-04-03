from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt

def filter(df, vH, vL):
    try:
        # Ensure the DataFrame index is reset
        df = df.reset_index(drop=True)
        
        if 'Time' not in df.columns or len(df['Time']) < 2:
            raise ValueError("The 'Time' column is missing or doesn't have enough data.")
        
        # Ensure data is numeric
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df['CAN_H'] = pd.to_numeric(df['CAN_H'], errors='coerce')
        df['CAN_L'] = pd.to_numeric(df['CAN_L'], errors='coerce')
        
        # Define the sample rate and cutoff frequency
        sample_rate = 1 / (df['Time'].iloc[1] - df['Time'].iloc[0])
        cutoff_frequency = 3

        # Create a butterworth filter
        b, a = butter(1, cutoff_frequency / (sample_rate / 2), btype='low')

        # Apply the filter to the voltage data
        df['CAN_H'] = filtfilt(b, a, df[vH])
        df['CAN_L'] = filtfilt(b, a, df[vL])

    except Exception as e:
        print(f"Error in filtering: {e}")
        return None

    return df
