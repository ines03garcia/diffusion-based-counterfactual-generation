import pandas as pd
import os
from config import ROOT, METADATA_ROOT

# Define paths
base_path = ROOT
input_file = os.path.join(METADATA_ROOT, "grouped_df.csv")
output_file = os.path.join(METADATA_ROOT, "grouped_df_with_anomaly.csv")

# Read the CSV file
print(f"Reading CSV file from: {input_file}")
df = pd.read_csv(input_file)

# Create the 'anomaly' column
# Set to 1 if breast_birads is not "BI-RADS 1", otherwise 0
df['anomaly'] = (df['breast_birads'] != "BI-RADS 1").astype(int)

# Save the new CSV file
print(f"Saving new CSV file to: {output_file}")
df.to_csv(output_file, index=False)

# Print statistics
print(f"\nStatistics:")
print(f"Total rows: {len(df)}")
print(f"Rows with anomaly=1: {df['anomaly'].sum()}")
print(f"Rows with anomaly=0: {(df['anomaly'] == 0).sum()}")
print(f"\nFile saved successfully!")
