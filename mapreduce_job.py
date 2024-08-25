import pandas as pd
import sys

def calculate_kpis(df):
    kpis = {}
    
    # Ensure these columns exist in the DataFrame
    if 'PER' in df.columns:
        kpis['PER'] = df['PER'].mean()
    if 'TS%' in df.columns:
        kpis['TS%'] = df['TS%'].mean()
    if 'eFG%' in df.columns:
        kpis['eFG%'] = df['eFG%'].mean()
    if 'AST' in df.columns and 'TOV' in df.columns:
        kpis['AST/TO'] = (df['AST'] / df['TOV']).mean()
    if 'USG%' in df.columns:
        kpis['USG%'] = df['USG%'].mean()
    
    return pd.DataFrame([kpis])

if __name__ == "__main__":
    input_csv = sys.argv[2]
    output_csv = sys.argv[4]

    df = pd.read_csv(input_csv)
    kpis = calculate_kpis(df)
    kpis.to_csv(output_csv, index=False)

