import pandas as pd
df = pd.read_csv('atlas_segmentations/segresnet/all_cases.csv')
print('All columns:', df.columns.tolist())
print('All unique region names:', sorted(df['atlas_region_name'].unique()))
print('Total unique regions:', df['atlas_region_name'].nunique())
