# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:06:14 2023

@author: lb945465
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_iaq_file(file_path, environment):
    # Create an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Read each worksheet in the Excel file and merge them into a single DataFrame
    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)

        # Check if the sheet contains the necessary columns
        if 'DateTime' not in df.columns:
            # If 'DateTime' column doesn't exist, combine 'Date' and 'Time' columns into one
            if 'Date' in df.columns and 'Time' in df.columns:
                df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
                df.drop(['Date', 'Time'], axis=1, inplace=True)
            else:
                # Handle the case where neither 'DateTime' nor 'Date' and 'Time' columns exist
                continue

        # Set the DateTime column as the index
        df.set_index('DateTime', inplace=True)

        # Append the DataFrame to the merged DataFrame if it's not empty
        if not df.empty:
            merged_df = pd.concat([merged_df, df])

    # Optionally, sort the merged DataFrame by DateTime
    merged_df.sort_index(inplace=True)

    # Filter the merged DataFrame to include only columns ending with "_BC"
    BC_columns = merged_df.filter(like='_BC')

    # Add the "Site" column to the filtered DataFrame
    BC_columns['Site'] = merged_df['Site']

    # Remove the "_BC" suffix from column names
    BC = BC_columns.rename(columns=lambda x: x.rstrip('_BC'))

    # Check if "_BrC" columns exist in the DataFrame before processing
    if '_BrC' in merged_df.columns:
        # Filter the merged DataFrame to include only columns ending with "_BrC"
        BrC_columns = merged_df.filter(like='_BrC')

        # Add the "Site" column to the filtered DataFrame
        BrC_columns['Site'] = merged_df['Site']

        # Remove the "_BrC" suffix from column names
        BrC = BrC_columns.rename(columns=lambda x: x.rstrip('_BrC'))
    else:
        BrC = None

    # Add the "Dataset" column to identify the environment
    BC['Dataset'] = environment
    if BrC is not None:
        BrC['Dataset'] = environment

    return BC, BrC

# Define file paths for indoor and outdoor data
indoor_file_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\IAQ\IAQ_Indoor.xlsx'
outdoor_file_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\IAQ\\IAQ_Outdoor.xlsx'  # Replace with the outdoor file path

# Process indoor and outdoor data
indoor_BC, indoor_BrC = process_iaq_file(indoor_file_path, 'Indoor')
outdoor_BC, outdoor_BrC = process_iaq_file(outdoor_file_path, 'Outdoor')

# Print the first few rows of the resulting DataFrames
print("Indoor BC:")
if indoor_BC is not None:
    print(indoor_BC.head())
else:
    print("No BC data for Indoor")
    
print("Indoor BrC:")
if indoor_BrC is not None:
    print(indoor_BrC.head())
else:
    print("No BrC data for Indoor")

print("Outdoor BC:")
if outdoor_BC is not None:
    print(outdoor_BC.head())
else:
    print("No BrC data for Outdoor")
    
print("Outdoor BrC:")
if outdoor_BrC is not None:
    print(outdoor_BrC.head())
else:
    print("No BrC data for Outdoor")

# Merge indoor and outdoor BC DataFrames
all_BC = pd.concat([indoor_BC, outdoor_BC])

# Drop rows with all NaN values
all_BC.dropna(how='all', inplace=True)

# Reset the index to convert it into a regular column
all_BC.reset_index(inplace=True)

# Define a function to replace outliers with NaNs based on IQR
def replace_outliers_with_nan_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column[(column < lower_bound) | (column > upper_bound)] = pd.NA

# Select columns for outlier processing
columns_to_process = all_BC.columns.difference(['DateTime', 'Site', 'Dataset'])

# Apply the function to selected columns one by one
for column_name in columns_to_process:
    replace_outliers_with_nan_iqr(all_BC[column_name])

all_BC

# Melt the DataFrame while keeping 'Site' and 'Dataset' columns
melted_BC = pd.melt(all_BC, id_vars=['DateTime', 'Site', 'Dataset'], var_name='Variable', value_name='Value')

# Replace values less than or equal to 0 with NaN in the melted_BC DataFrame
melted_BC['Value'] = melted_BC['Value'].where(melted_BC['Value'] > 0, other=float('nan'))

# Create the catplot with the desired figure size
plot = sns.catplot(
    data=melted_BC, x="Site", y="Value", hue="Dataset",
    kind="violin", bw_adjust=.5, cut=0, split=True, 
    inner="quartiles",
    height=5,  # Set the figure height
    aspect=2,  # Set the aspect ratio to control the width
)


# Create the figure and set the figure size
plt.figure(figsize=(12, 4))

# Define the color palette for the legend and violin plots
palette = {'Indoor': 'blue', 'Outdoor': 'orange'}

# Create the violin plots and specify the hue and palette
g=sns.violinplot(data=melted_BC, 
               x='Site', 
               y='Value', 
               hue='Dataset', 
               split=True, 
               inner=None, 
               palette=palette,
               )

# Create the box plots without legend
sns.boxplot(data=melted_BC, 
            x='Site', 
            y='Value', 
            hue='Dataset', 
            color='white', 
            width=0.3, 
            boxprops={'zorder': 2}, 
            showfliers=False,
            )

# Set the x-label and y-label
plt.xlabel("")
plt.ylabel("BC concentration (μg/m${^3}$)")

handles, labels = g.get_legend_handles_labels()
g.legend(handles[:2], labels[:2], title='Set')

plt.tight_layout()
plt.show()


