# feature_extraction
This module is used to extract features from waveform datasets.
The feature_extraction.py file takes in a path to a directory containing ECU waveform data
and extracts features from their signal sections.

First, the average ringing limits of the bus are extracted. Then features from all combinations
of features are extracted.

The data_management folder contains functions to streamline general data processing:
1) check_high_low.py: verifies CAN_H and CAN_L columns in waveform data
2) filter.py: applies butterworth filter to waveform data to minimize noise
3) open_file.py: opens and prepares waveform data for data processing.

The feature_management folder contains functions pertaining to the feature_extraction process itself:
1) calculate_features.py: calculates features for given section
2) find_bits.py: finds locations of bit signal sections
3) find_signal_features.py: groups signal section data based on which sections are being analyzed
4) find_signals.py: identifies the location of given signal sections to be analyzed
5) parse_raw_data.py: control function to manage feature extraction process

The ringing_management folder contains functions pertaining specifically to ringing feature extraction:
1) find_ringing_limits.py: finds average ringing section limits to streamline process
2) find_ringing.py: implements curve fitting to isolate the exact ringing section location of a given data frame

# ML
This module is used to evaluate the classification of processed features by various models.
The evaluate_classification.py file takes a dataset folder in the ML module and outputs a results
folder containing the classification results for each dataset.

There are four models in the ML module, though more can be added if desired:
1) CNN
2) NN
3) RF
4) SVM

# results and datasets
The datasets used to train and classify the models, the respective results of those classifications, and
all of the raw waveform datasets used for feature extraction are in datasets-master
