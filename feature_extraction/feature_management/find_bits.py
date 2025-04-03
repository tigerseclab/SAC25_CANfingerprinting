def find_bits(df):
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
    
    return bits[:-1]

