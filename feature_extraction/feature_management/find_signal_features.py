from feature_management.calculate_features import calculate_features

def find_signal_features(signals, w_type, s_type, f_type):
    all_features = []
    columns = []

    # Calculate features for each section, sorting by given types
    if s_type != "ringing":
        for signal in signals:
            rising_edge, dominant_state, falling_edge = signal

            if s_type == "combined":
                rising_edge_features, rising_edge_columns = calculate_features(rising_edge, w_type, f_type)
                dominant_state_features, dominant_state_columns = calculate_features(dominant_state, w_type, f_type)
                falling_edge_features, falling_edge_columns = calculate_features(falling_edge, w_type, f_type)

                for i in range(len(rising_edge_columns)):
                    rising_edge_columns[i] = 'rising_edge_' + rising_edge_columns[i]

                for i in range(len(dominant_state_columns)):
                    dominant_state_columns[i] = 'dominant_state_' + dominant_state_columns[i]
                
                for i in range(len(falling_edge_columns)):
                    falling_edge_columns[i] = 'falling_edge_' + falling_edge_columns[i]

                signal_features = rising_edge_features + dominant_state_features + falling_edge_features
                feature_columns = rising_edge_columns + dominant_state_columns + falling_edge_columns

                all_features.append(signal_features)
                

            elif s_type == "rising":
                signal_features, feature_columns = calculate_features(rising_edge, w_type, f_type)
                all_features.append(signal_features)
                
            
            elif s_type == "steady":
                signal_features, feature_columns = calculate_features(dominant_state, w_type, f_type)
                all_features.append(signal_features)
                
            
            elif s_type == "falling":
                signal_features, feature_columns = calculate_features(falling_edge, w_type, f_type)
                all_features.append(signal_features)
                
    else:
        for signal in signals:
            ringing_high, ringing_low = signal

            ringing_high_features, ringing_high_columns = calculate_features(ringing_high, w_type, f_type)
            ringing_low_features, ringing_low_columns = calculate_features(ringing_low, w_type, f_type)

            for i in range(len(ringing_high_columns)):
                ringing_high_columns[i] = 'ringing_high_' + ringing_high_columns[i]

            for i in range(len(ringing_low_columns)):
                ringing_low_columns[i] = 'ringing_low_' + ringing_low_columns[i]

            signal_features = ringing_high_features + ringing_low_features
            feature_columns = ringing_high_columns + ringing_low_columns     

            all_features.append(signal_features)

    columns.append(feature_columns)
    
    return all_features, columns
