# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:59:00 2024

@author: lb945465
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def process_iaq_file(file_path, environment):
    merged_df = pd.DataFrame()

    excel_file = pd.ExcelFile(file_path)
    for sheet_name in excel_file.sheet_names:
        df = excel_file.parse(sheet_name)
        print(f"Processing sheet: {sheet_name}")

        # Check if DataFrame is empty
        if df.empty:
            print(f"Warning: DataFrame from sheet '{sheet_name}' is empty.")
            continue

        # Check if 'Datetime' column exists and set it as the index
        if 'Datetime' in df.columns:
            df.set_index('Datetime', inplace=True)
        else:
            print(f"'Datetime' column not found in '{sheet_name}'.")
            continue

        # Check if 'Site' is in df before concatenation
        if 'Site' not in df.columns:
            print(f"'Site' column missing in '{sheet_name}'.")
            continue

        merged_df = pd.concat([merged_df, df])

    if merged_df.empty:
        raise ValueError("Merged DataFrame is empty. No data was processed.")

    if 'Site' not in merged_df.columns:
        raise ValueError("Column 'Site' not found after merging sheets.")

    # Sort the merged DataFrame by index (Datetime)
    merged_df.sort_index(inplace=True)

    # Filter and process "_BC" columns
    BC_columns = merged_df.filter(like='_BC')
    BC_columns['Site'] = merged_df['Site']
    BC = BC_columns.rename(columns=lambda x: x.rstrip('_BC'))
    BC['Dataset'] = environment

    # Check and process "_BrC" columns
    if '_BrC' in merged_df.columns:
        BrC_columns = merged_df.filter(like='_BrC')
        BrC_columns['Site'] = merged_df['Site']
        BrC = BrC_columns.rename(columns=lambda x: x.rstrip('_BrC'))
        BrC['Dataset'] = environment
    else:
        BrC = None

    return BC, BrC

# Define file paths for indoor and outdoor data
indoor_file_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\IAQ\PM_Indoor.xlsx'
outdoor_file_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\IAQ\PM_Outdoor.xlsx' 

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
all_data = pd.concat([indoor_BC, outdoor_BC])
#all_data = pd.concat([indoor_BrC, outdoor_BrC])

# Drop rows with all NaN values
all_data.dropna(how='all', inplace=True)

# Reset the index to convert it into a regular column
all_data.reset_index(inplace=True)

# Define a function to replace outliers with NaNs based on IQR
def replace_outliers_with_nan_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column[(column < lower_bound) | (column > upper_bound)] = pd.NA

# Select columns for outlier processing
columns_to_process = all_data.columns.difference(['DateTime', 'Site', 'Dataset'])

# Apply the function to selected columns one by one
for column_name in columns_to_process:
    replace_outliers_with_nan_iqr(all_data[column_name])
    
# Melt the DataFrame while keeping 'Site' and 'Dataset' columns
melted_BC = pd.melt(all_data, id_vars=['Datetime', 'Site', 'Dataset'], var_name='Variable', value_name='Value')

# Replace values less than or equal to 0 with NaN in the melted_BC DataFrame
melted_BC['Value'] = melted_BC['Value'].where(melted_BC['Value'] > 0, other=float('nan'))

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

# Assuming you have the indoor_BC and outdoor_BC dataframes
dataframes = [indoor_BC, outdoor_BC]

# List to store the resulting dataframes
daily_mean_dfs = []

for df in dataframes:
    # Reset the index and drop 'Dataset' column
    df.drop(columns=['Dataset'], inplace=True)
    df.reset_index(inplace=True)

    # Melt the dataframe while keeping 'DateTime' and 'Site' columns
    melted_df = pd.melt(df, id_vars=['Datetime', 'Site'], var_name='Variable', value_name='Value')
    melted_df.set_index('Datetime', inplace=True)

    # Group the melted dataframe by 'Site' and resample to daily frequency, calculating the mean
    daily_mean_df = melted_df.groupby('Site').resample('D').mean()

    # Append the resulting dataframe to the list
    daily_mean_dfs.append(daily_mean_df)

# Unpack the list into separate dataframes for indoor and outdoor
daily_mean_indoor, daily_mean_outdoor = daily_mean_dfs

# Print the resulting dataframes
print("Daily Mean Indoor:")
print(daily_mean_indoor)

print("Daily Mean Outdoor:")
print(daily_mean_outdoor)

# Assuming you have daily_mean_outdoor and daily_mean_indoor dataframes

# Create an empty dictionary to store the correlation results by Site
correlation_results = {}

# Get the unique Site values
unique_sites = daily_mean_outdoor.index.get_level_values('Site').unique()

# Loop through each unique Site
for site in unique_sites:
    # Extract the data for the current Site
    outdoor_data = daily_mean_outdoor.loc[daily_mean_outdoor.index.get_level_values('Site') == site]
    indoor_data = daily_mean_indoor.loc[daily_mean_indoor.index.get_level_values('Site') == site]

    # Calculate the Spearman correlation between outdoor and indoor data for the current Site
    spearman_corr = outdoor_data.corrwith(indoor_data, method='spearman')

    # Store the correlation results in the dictionary
    correlation_results[site] = spearman_corr

# Create a DataFrame from the correlation results
correlation_df = pd.DataFrame(correlation_results)

# Print the resulting correlation DataFrame
print(correlation_df)

# Assuming you have the correlation_df DataFrame

# Create a heatmap
plt.figure(figsize=(10, 3), dpi=300)  # Adjust the figure size as needed
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1)

# Set the plot title and labels
plt.title('Spearman Correlation Heatmap', fontweight="bold")
plt.xlabel('')
plt.ylabel('r${_s}$', fontweight="bold")

# Show the heatmap
plt.show()


