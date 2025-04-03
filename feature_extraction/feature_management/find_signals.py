def findSignals(df, ringing_limits=None):
    bits = []
    onBit = False
    start_time = None

    # Loop through dataframe and find differential
    for i in range(len(df) - 1):  
        voltage_diff = abs(df['CAN_H'][i] - df['CAN_L'][i])
        next_voltage_diff = abs(df['CAN_H'][i+1] - df['CAN_L'][i+1])
        
        # If voltage differential is more than .08 volts and derivative is positive, its on a new bit
        if voltage_diff > 0.08:
            if not onBit and (next_voltage_diff - voltage_diff) > 0:
                start_time = df['Time'][i]
                onBit = True
        # Else if differential is less than .08 volts, and derivative is negative, its at the end of a bit
        elif voltage_diff < 0.08:
            if onBit and (next_voltage_diff - voltage_diff) < 0:
                bit_end_time = df['Time'][i]
                # Make a pair from bit start and end as new bit and append to bits, reset bit detection
                if start_time is not None:
                    bits.append((start_time, bit_end_time))
                    start_time = None
                onBit = False

    bits = bits[:-1]

    # Separate bits into rising, dominant, and falling
    signals = []

    for bit in bits:
        if ringing_limits == None:
            start_time, end_time = bit
            bit_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
            bit_length = len(bit_data)
            
            rising_edge_end = int(bit_length * 0.25)
            falling_edge_start = int(bit_length * 0.90)
            
            rising_edge = bit_data.iloc[:rising_edge_end]
            dominant_state = bit_data.iloc[rising_edge_end:falling_edge_start]
            falling_edge = bit_data.iloc[falling_edge_start:]
            
            signals.append((rising_edge, dominant_state, falling_edge)) 
        else:
            start_time, end_time = bit
            bit_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
            bit_length = len(bit_data)
            
            high_ringing_start, high_ringing_end, low_ringing_start, low_ringing_end = ringing_limits
            id_hrs = int(bit_length * high_ringing_start)
            id_hre = int(bit_length * high_ringing_end)
            id_lrs = int(bit_length * low_ringing_start)
            id_lre = int(bit_length * low_ringing_end)
            
            ringing_high = bit_data.iloc[id_hrs:id_hre]
            ringing_low = bit_data.iloc[id_lrs:id_lre]
            
            signals.append((ringing_high, ringing_low)) 

    return signals
