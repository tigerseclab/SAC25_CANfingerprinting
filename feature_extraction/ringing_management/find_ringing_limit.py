import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
import random

# Define the exponential decay function for curve_high using a lambda function
exp_decay = lambda x, a, b, c: a * np.exp(-b * x) + c

# Define the logarithmic function for curve_low using a lambda function
log_func = lambda x, a, b, c: a * np.log(b * x) + c

def normalize_data(series):
    """ Normalize the data to the range [0, 1]. """
    return (series - series.min()) / (series.max() - series.min()), series.min(), series.max()

def find_ringing_limit(df):
    bits = []
    onBit = False
    start_time = None

    # Loop through dataframe and find differential
    for i in range(len(df) - 1):  
        voltage_diff = abs(df['CAN_H'][i] - df['CAN_L'][i])
        next_voltage_diff = abs(df['CAN_H'][i+1] - df['CAN_L'][i+1])
        
        # If voltage differential is more than .08 volts and derivative is positive, it's on a new bit
        if voltage_diff > 0.08:
            if not onBit and (next_voltage_diff - voltage_diff) > 0:
                start_time = df['Time'][i]
                onBit = True
        # Else if differential is less than .08 volts, and derivative is negative, it's at the end of a bit
        elif voltage_diff < 0.08:
            if onBit and (next_voltage_diff - voltage_diff) < 0:
                bit_end_time = df['Time'][i]
                # Make a pair from bit start and end as new bit and append to bits, reset bit detection
                if start_time is not None:
                    bits.append((start_time, bit_end_time))
                    start_time = None
                onBit = False

    bits = bits[:-1]  # Remove the last bit if necessary

    # Check if bits were detected
    if not bits:
        print("No bits detected.")
        return None

    # Randomly select one bit
    random_bit = random.choice(bits)
    start, end = random_bit
    bit_df = df[(df['Time'] >= start) & (df['Time'] <= end)]
    bit_length = len(bit_df)

    # Define the portion of the DataFrame to analyze (first 15%)
    bit_beginning_index = int(len(bit_df) * 0.15)

    # Get the subset of the DataFrame that represents the first 15%
    bit_df_beginning = bit_df.iloc[:bit_beginning_index]

    # Get the relative position (index) within the subset of the sub-dataframe
    curve_start_high = bit_df_beginning["CAN_H"].idxmax() - bit_df.index[0]
    curve_start_low = bit_df_beginning["CAN_L"].idxmin() - bit_df.index[0]

    # Compute the end indices relative to the sub-dataframe
    curve_end = bit_length - int(0.5 * bit_length)

    # Extract data for curve fitting using the relative indices
    high_curve_df = bit_df.iloc[curve_start_high:curve_end]
    low_curve_df = bit_df.iloc[curve_start_low:curve_end]

    # Now fit an exponential decay to high_curve_dfs and a logarithmic curve to low_curve_dfs
    if len(high_curve_df) > 0 and len(low_curve_df) > 0:
        try:
            # Normalize the CAN_H and CAN_L columns
            high_norm, high_min, high_max = normalize_data(high_curve_df['CAN_H'])
            low_norm, low_min, low_max = normalize_data(low_curve_df['CAN_L'])

            # Dynamic initial guesses for high curve fitting (exponential decay)
            a_high_init = high_norm.max() - high_norm.min()
            b_high_init = 1 / (high_curve_df['Time'].max() - high_curve_df['Time'].min())  # rough estimate
            c_high_init = high_norm.min()

            # Dynamic initial guesses for low curve fitting (logarithmic function)
            a_low_init = low_norm.max() - low_norm.min()
            b_low_init = 1e-2  # small positive number to avoid log(0)
            c_low_init = low_norm.min()

            # Define the bounds for the parameters
            param_bounds_high = ([0, 0, 0], [np.inf, np.inf, 1])  # Normalized to [0, 1]
            param_bounds_low = ([0, 1e-6, 0], [np.inf, np.inf, 1])  # Normalized to [0, 1]

            # Fit the exponential decay to high_curve_df
            popt_high, _ = curve_fit(
                exp_decay, 
                high_curve_df['Time'], 
                high_norm,
                p0=(a_high_init, b_high_init, c_high_init), 
                bounds=param_bounds_high,
                max_nfev=100000,
                method='trf'
            )
           
            # Fit the logarithmic curve to low_curve_df
            popt_low, _ = curve_fit(
                log_func, 
                low_curve_df['Time'] - low_curve_df['Time'].min() + 1e-6, 
                low_norm,
                p0=(a_low_init, b_low_init, c_low_init), 
                bounds=param_bounds_low,
                max_nfev=100000,
                method='trf'
            )

            # Generate x values for plotting
            x_high = np.linspace(high_curve_df['Time'].min(), high_curve_df['Time'].max(), 1000)
            x_low = np.linspace(low_curve_df['Time'].min(), low_curve_df['Time'].max(), 1000)

            # Calculate the fitted curves (denormalizing the output)
            y_high_fit = exp_decay(x_high, *popt_high) * (high_max - high_min) + high_min
            y_low_fit = log_func(x_low - x_low.min() + 1e-6, *popt_low) * (low_max - low_min) + low_min

            # Calculate the derivative of the fitted curves
            high_diff = np.diff(y_high_fit) / np.diff(x_high)
            low_diff = np.diff(y_low_fit) / np.diff(x_low)

            # Find the points where the derivative meets the criteria
            high_ringing_limit_indices = np.where(np.abs(high_diff) < .01)[0]
            low_ringing_limit_indices = np.where(np.abs(low_diff) < .005)[0]

            # Ringing limits in terms of actual time
            high_ringing_time = x_high[high_ringing_limit_indices[0]] if high_ringing_limit_indices.size > 0 else None
            low_ringing_time = x_low[low_ringing_limit_indices[0]] if low_ringing_limit_indices.size > 0 else None

            # Start of the curve in the context of the full bit
            high_ringing_start_time = high_curve_df['Time'].min()
            low_ringing_start_time = low_curve_df['Time'].min()

            # Calculate percentages relative to the entire bit
            if high_ringing_start_time:
                high_ringing_start_percentage = (high_ringing_start_time - bit_df['Time'].min()) / (bit_df['Time'].max() - bit_df['Time'].min())
            else:
                high_ringing_start_percentage = None

            if high_ringing_time:
                high_ringing_end_percentage = (high_ringing_time - bit_df['Time'].min()) / (bit_df['Time'].max() - bit_df['Time'].min())
            else:
                high_ringing_end_percentage = None

            if low_ringing_start_time:
                low_ringing_start_percentage = (low_ringing_start_time - bit_df['Time'].min()) / (bit_df['Time'].max() - bit_df['Time'].min())
            else:
                low_ringing_start_percentage = None

            if low_ringing_time:
                low_ringing_end_percentage = (low_ringing_time - bit_df['Time'].min()) / (bit_df['Time'].max() - bit_df['Time'].min())
            else:
                low_ringing_end_percentage = None

            ringing_percentages = {
                'high_ringing_start': high_ringing_start_percentage,
                'high_ringing_end': high_ringing_end_percentage,
                'low_ringing_start': low_ringing_start_percentage,
                'low_ringing_end': low_ringing_end_percentage
            }

            return ringing_percentages

        except RuntimeError as e:
            warnings.warn(f"Fitting failed for the selected bit: {e}", OptimizeWarning)
            return None
    else:
        print(f"Skipping bit due to insufficient data.")
        return None
