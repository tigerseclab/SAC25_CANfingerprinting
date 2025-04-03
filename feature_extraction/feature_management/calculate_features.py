import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import entropy  # for calculating Shannon entropy

# Helper function to calculate Shannon entropy for a signal
def shannon_entropy(signal):
    # Calculate the probability distribution
    probability_distribution, _ = np.histogram(signal, bins=256, density=True)
    # Filter out zero values for entropy calculation
    probability_distribution = probability_distribution[probability_distribution > 0]
    return entropy(probability_distribution)

# Function to calculate the features for each interval
def calculate_features(data, w_type, f_type):
    features = []
    columns = []

    if w_type == "differential":
        # find differential
        VoltageDifferential_values = data.iloc[:, 1]
        if f_type == "combined" or f_type == "voltage":
            # Calculate time-domain features for voltage
            VoltageDifferential_mean = VoltageDifferential_values.mean()
            VoltageDifferential_std = VoltageDifferential_values.std()
            VoltageDifferential_meandev = np.mean(np.abs(VoltageDifferential_values - VoltageDifferential_mean))
            VoltageDifferential_skewness = skew(VoltageDifferential_values)
            VoltageDifferential_kurtosis = kurtosis(VoltageDifferential_values)
            VoltageDifferential_rms = np.sqrt(np.mean(np.square(VoltageDifferential_values)))
            VoltageDifferential_highest = VoltageDifferential_values.max()
            VoltageDifferential_lowest = VoltageDifferential_values.min()

            # Append all features to the list
            features.extend([
                VoltageDifferential_mean,
                VoltageDifferential_std, VoltageDifferential_meandev,
                VoltageDifferential_skewness, VoltageDifferential_kurtosis,
                VoltageDifferential_rms, VoltageDifferential_highest,
                VoltageDifferential_lowest
            ])

            columns.extend([
                'differential_voltage_mean',
                'differential_voltage_std',
                'differential_voltage_meandev',
                'differential_voltage_skewness',
                'differential_voltage_kurtosis',
                'differential_voltage_rms',
                'differential_voltage_highest',
                'differential_voltage_lowest',
            ])

        if f_type == "combined" or f_type == "frequency":
            # Perform Fourier Transform (FFT) to get frequency components
            fft_values = np.fft.fft(VoltageDifferential_values)
            freq = np.fft.fftfreq(len(VoltageDifferential_values))

            # Calculate magnitude spectrum
            magnitude_spectrum = np.abs(fft_values)

            # Calculate frequency-domain features
            centroid = np.sum(freq * magnitude_spectrum) / np.sum(magnitude_spectrum)
            entropy_freq = -np.sum(magnitude_spectrum * np.log2(magnitude_spectrum + 1e-12))  # Avoid log(0)
            spread = np.sqrt(np.sum((freq - centroid) ** 2 * magnitude_spectrum) / np.sum(magnitude_spectrum))
            skewness_freq = skew(magnitude_spectrum)
            avg_spectrum = np.mean(magnitude_spectrum)
            variance = np.var(magnitude_spectrum)
            kurtosis_freq = kurtosis(magnitude_spectrum)
            irregularity = np.mean(np.abs(np.diff(magnitude_spectrum)))

            # Append all features to the list
            features.extend([
                centroid, entropy_freq, spread, skewness_freq, avg_spectrum,
                variance, kurtosis_freq, irregularity
            ])

            columns.extend([
                'differential_frequency_centroid',
                'differential_frequency_entropy', 
                'differential_frequency_spread',
                'differential_frequency_skewness_freq', 
                'differential_frequency_avg_spectrum',
                'differential_frequency_variance', 
                'differential_frequency_kurtosis_freq', 
                'differential_frequency_irregularity'
            ])

    else:
        if f_type == "combined" or f_type == "voltage":
            # Calculate time-domain features for voltage
            voltage_high_mean = data["VoltHigh"].mean()
            voltage_low_mean = data["VoltLow"].mean()
            voltage_high_std = data["VoltHigh"].std()
            voltage_low_std = data["VoltLow"].std()
            voltage_high_meandev = np.mean(np.abs(data["VoltHigh"] - voltage_high_mean))
            voltage_low_meandev = np.mean(np.abs(data["VoltLow"] - voltage_low_mean))
            voltage_high_skewness = skew(data["VoltHigh"])
            voltage_low_skewness = skew(data["VoltLow"])
            voltage_high_kurtosis = kurtosis(data["VoltHigh"])
            voltage_low_kurtosis = kurtosis(data["VoltLow"])
            voltage_high_rms = np.sqrt(np.mean(np.square(data["VoltHigh"])))
            voltage_low_rms = np.sqrt(np.mean(np.square(data["VoltLow"])))
            voltage_high_highest = data["VoltHigh"].max()
            voltage_low_highest = data["VoltLow"].max()
            voltage_high_lowest = data["VoltHigh"].min()
            voltage_low_lowest = data["VoltLow"].min()

            # Append all features to the list
            features.extend([
                voltage_high_mean, voltage_low_mean,
                voltage_high_std, voltage_low_std,
                voltage_high_meandev, voltage_low_meandev,
                voltage_high_skewness, voltage_low_skewness,
                voltage_high_kurtosis, voltage_low_kurtosis,
                voltage_high_rms, voltage_low_rms,
                voltage_high_highest, voltage_low_highest,
                voltage_high_lowest, voltage_low_lowest,
            ])

            columns.extend([
                'VH_voltage_mean', 'VL_voltage_mean',
                'VH_voltage_std', 'VL_voltage_std',
                'VH_voltage_meandev', 'VL_voltage_meandev',
                'VH_voltage_skewness', 'VL_voltage_skewness',
                'VH_voltage_kurtosis', 'VL_voltage_kurtosis',
                'VH_voltage_rms', 'VL_voltage_rms',
                'VH_voltage_highest', 'VL_voltage_highest',
                'VH_voltage_lowest', 'VL_voltage_lowest',
            ])
        
        if f_type == "combined" or f_type == "frequency":
            # Perform Fourier Transform (FFT) to get frequency components
            voltage_high_fft_values = np.fft.fft(data["VoltHigh"])
            voltage_high_freq = np.fft.fftfreq(len(data["VoltHigh"]))
            voltage_low_fft_values = np.fft.fft(data["VoltLow"])
            voltage_low_freq = np.fft.fftfreq(len(data["VoltLow"]))

            # Calculate magnitude spectrum
            voltage_high_magnitude_spectrum = np.abs(voltage_high_fft_values)
            voltage_low_magnitude_spectrum = np.abs(voltage_low_fft_values)

            # Calculate frequency-domain features
            voltage_high_centroid = np.sum(voltage_high_freq * voltage_high_magnitude_spectrum) / np.sum(voltage_high_magnitude_spectrum)
            voltage_low_centroid = np.sum(voltage_low_freq * voltage_low_magnitude_spectrum) / np.sum(voltage_low_magnitude_spectrum)
            voltage_high_entropy_freq = -np.sum(voltage_high_magnitude_spectrum * np.log2(voltage_high_magnitude_spectrum + 1e-12))  # Avoid log(0)
            voltage_low_entropy_freq = -np.sum(voltage_low_magnitude_spectrum * np.log2(voltage_low_magnitude_spectrum + 1e-12))
            voltage_high_spread = np.sqrt(np.sum((voltage_high_freq - voltage_high_centroid) ** 2 * voltage_high_magnitude_spectrum) / np.sum(voltage_high_magnitude_spectrum))
            voltage_low_spread = np.sqrt(np.sum((voltage_low_freq - voltage_low_centroid) ** 2 * voltage_low_magnitude_spectrum) / np.sum(voltage_low_magnitude_spectrum))
            voltage_high_skewness_freq = skew(voltage_high_magnitude_spectrum)
            voltage_low_skewness_freq = skew(voltage_low_magnitude_spectrum)
            voltage_high_avg_spectrum = np.mean(voltage_high_magnitude_spectrum)
            voltage_low_avg_spectrum = np.mean(voltage_low_magnitude_spectrum)
            voltage_high_variance = np.var(voltage_high_magnitude_spectrum)
            voltage_low_variance = np.var(voltage_low_magnitude_spectrum)
            voltage_high_kurtosis_freq = kurtosis(voltage_high_magnitude_spectrum)
            voltage_low_kurtosis_freq = kurtosis(voltage_low_magnitude_spectrum)
            voltage_high_irregularity = np.mean(np.abs(np.diff(voltage_high_magnitude_spectrum)))
            voltage_low_irregularity = np.mean(np.abs(np.diff(voltage_low_magnitude_spectrum)))

            # Append all features to the list
            features.extend([
                voltage_high_centroid, voltage_low_centroid,
                voltage_high_entropy_freq, voltage_low_entropy_freq,
                voltage_high_spread, voltage_low_spread, 
                voltage_high_skewness_freq, voltage_low_skewness_freq,
                voltage_high_avg_spectrum, voltage_low_avg_spectrum,
                voltage_high_variance, voltage_low_variance, 
                voltage_high_kurtosis_freq, voltage_low_kurtosis_freq,
                voltage_high_irregularity, voltage_low_irregularity
            ])

            columns.extend([
                'VH_frequency_centroid', 'VL_frequency_centroid',
                'VH_frequency_entropy', 'VL_frequency_entropy',
                'VH_frequency_spread', 'VL_frequency_spread', 
                'VH_frequency_skewness_freq', 'VL_frequency_skewness_freq',
                'VH_frequency_avg_spectrum', 'VL_frequency_avg_spectrum',
                'VH_frequency_variance', 'VL_frequency_variance', 
                'VH_frequency_kurtosis_freq', 'VL_frequency_kurtosis_freq',
                'VH_frequency_irregularity', 'VL_frequency_irregularity'
            ])

    return features, columns
