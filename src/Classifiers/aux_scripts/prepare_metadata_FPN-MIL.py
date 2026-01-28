import pandas as pd
import os
from config import ROOT, METADATA_ROOT

# Define variable
with_counterfactuals = False

# Define paths
base_path = ROOT
input_file = os.path.join(METADATA_ROOT, "resized_df_has_counterfactual.csv")
if not with_counterfactuals:
    output_file = os.path.join(METADATA_ROOT, "resized_df_anomaly.csv")
else:
    output_file = os.path.join(METADATA_ROOT, "resized_df_counterfactuals_has_anomaly.csv")

# Read the CSV file
print(f"Reading CSV file from: {input_file}")
df = pd.read_csv(input_file)


if with_counterfactuals:
    for row in df.itertuples():
        if row.has_counterfactual:
            # Add new row with cf info
            cf_row = row._asdict().copy()
            cf_row.pop('Index', None)

            original_image_id = cf_row['image_id']
            if original_image_id.endswith('.png'):
                cf_row['image_id'] = original_image_id.replace('.png', 'cf.png')

            cf_row['finding_categories'] = "[""['No Finding']""]"
            cf_row['finding_birads'] = "[0]"
            cf_row['breast_birads'] = "BI-RADS 1"
            cf_row['Mass'] = 0
            cf_row['Suspicious_Calcification'] = 0
            cf_row['resized_xmin'] = "[0]"
            cf_row['resized_ymin'] = "[0]"
            cf_row['resized_xmax'] = "[0]"
            cf_row['resized_ymax'] = "[0]"
            
            df = pd.concat([df, pd.DataFrame([cf_row])], ignore_index=True)


# Drop the 'has_counterfactual' column
df = df.drop(columns=['has_counterfactual'])

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
