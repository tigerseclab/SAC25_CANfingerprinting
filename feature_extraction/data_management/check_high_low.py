# Determine which column has higher values
def checkHighLow(df):
    second_col = df.columns[1]
    third_col = df.columns[2]
    
    if df[second_col].mean() > df[third_col].mean():
        vH = second_col
        vL = third_col
    else:
        vH = third_col
        vL = second_col

    return vH, vL
