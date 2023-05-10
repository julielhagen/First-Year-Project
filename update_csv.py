# FILE ONLY TO UPDATE MASK COLUMN IN METADATA COPY

import pandas as pd
from os import listdir, remove

# Create pandas dataframe from metadata file
df = pd.read_csv('metadata_withmasks.csv')

# List of mask names
mask_ids = listdir('test_images_masks')

# Update dataframe from masks in mask_ids
for i, row in df.iterrows():
    im_id = row['img_id']
    mask_id = im_id[:-4] + '_mask.png'
    if mask_id in mask_ids:
        df.at[i, 'mask'] = 1

# Remove old metadata file and write new one
remove('metadata_withmasks.csv')
df.to_csv('metadata_withmasks.csv')
