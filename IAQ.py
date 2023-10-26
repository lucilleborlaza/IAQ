# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:06:14 2023

@author: lb945465
"""

import os
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Define the folder and file path
folder_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\IAQ'
file_name = 'IAQ_Database2.xlsx'
file_path = os.path.join(folder_path, file_name)

# Create an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# Read each worksheet in the Excel file and merge them into a single DataFrame
excel_file = pd.ExcelFile(file_path)
for sheet_name in excel_file.sheet_names:
    df = excel_file.parse(sheet_name)
    
    # Check if the sheet contains the necessary columns
    if 'Date' in df.columns and 'Time' in df.columns:
        # Combine 'Date' and 'Time' columns into a single datetime column
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        
        # Set the DateTime column as the index
        df.set_index('DateTime', inplace=True)
        
        # Drop the 'Date' and 'Time' columns
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        # Append the DataFrame to the merged DataFrame if it's not empty
        if not df.empty:
            merged_df = pd.concat([merged_df, df])

# Optionally, sort the merged DataFrame by DateTime
merged_df.sort_index(inplace=True)

# Filter the merged DataFrame to include only columns ending with "_BC"
BC_columns = merged_df.filter(like='_BC')

# Add the "Site" column to the filtered DataFrame
BC_columns['Site'] = merged_df['Site']

# The resulting DataFrame BC_dataframe contains the desired columns
print(BC_columns.head())

# Remove the "_BC" suffix from column names
BC = BC_columns.rename(columns=lambda x: x.rstrip('_BC'))

# Now, the column names in BC_dataframe do not have the "_BC" suffix
print(BC.head())

# Optionally, sort the merged DataFrame by DateTime
merged_df.sort_index(inplace=True)

# Filter the merged DataFrame to include only columns ending with "_BC"
BrC_columns = merged_df.filter(like='_BrC')

# Add the "Site" column to the filtered DataFrame
BrC_columns['Site'] = merged_df['Site']

# The resulting DataFrame BC_dataframe contains the desired columns
print(BrC_columns.head())

# Remove the "_BC" suffix from column names
BrC = BrC_columns.rename(columns=lambda x: x.rstrip('_BrC'))

# Now, the column names in BC_dataframe do not have the "_BC" suffix
print(BrC.head())

# Assuming you have the BC_dataframe with "Site" as a column
# Melt the BC_dataframe to long format
Specie="Brown carbon"
dataframe=BrC
dataframe.reset_index(inplace=True)

melted_df = pd.melt(dataframe, id_vars=["DateTime", "Site"], var_name="Column_Name")
melted_df.dropna(inplace=True)
melted_df = melted_df[melted_df['value'] > 0]

plt.figure(figsize=(20, 6), dpi=300)
ax = sns.boxenplot(x="Column_Name", 
                 y="value", 
                 data=melted_df, 
                 showfliers=False,
                 palette="Blues",
                 )
ylims=ax.get_ylim()
ax = sns.stripplot(x="Column_Name", 
                   y="value", 
                   hue="Site", 
                   data=melted_df,
                   alpha=0.3,
                   palette="tab10",
                   )
ax.set(ylim=ylims)
plt.ylabel(f"{Specie} mass conc. (mcg/m$^3$)")
plt.xlabel("")
plt.title(f"{Specie} by Sensor ID and Site", fontweight="bold")

# Move the legend to the right side outside of the plot
plt.legend(title="Site", bbox_to_anchor=(1, 0.5), loc='center left')
plt.show()

# Generate a list of 23 distinct colors using the 'tab20' color cycle
base_colors = plt.cm.tab20(np.linspace(0, 1, 23))

# Create a new colormap with the generated colors
custom_cmap = ListedColormap(base_colors)

# Convert the ListedColormap into a list of colors
custom_palette = [custom_cmap(i) for i in range(custom_cmap.N)]

# Set the custom palette using sns.set_palette()
sns.set_palette(custom_palette)

plt.figure(figsize=(10, 6), dpi=300)
ax = sns.boxenplot(x="Site", 
                 y="value", 
                 data=melted_df, 
                 showfliers=False,
                 palette="Blues",
                 )
ylims=ax.get_ylim()
ax = sns.stripplot(x="Site", 
                   y="value", 
                   hue="Column_Name", 
                   data=melted_df,
                   alpha=0.6,
                   #palette="custom_palette",
                   )
ax.set(ylim=ylims)
plt.ylabel(f"{Specie} mass conc. (mcg/m$^3$)")
plt.xlabel("")
plt.title(f"{Specie} by Sensor ID and Site", fontweight="bold")

# Move the legend to the right side outside of the plot
plt.legend(title="Site", bbox_to_anchor=(1, 0.5), loc='center left')
plt.show()