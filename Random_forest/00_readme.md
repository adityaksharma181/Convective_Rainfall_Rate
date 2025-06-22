follow the steps to get results

**Data Reprocessing**
1. Use insat-resampling file to resample training data and save to external file
2. Use imerg-resampling file to resample label data and save to external file

**Model Training**
3. Use saved files to train the model by splitting the dataset into training and test sets through RF-model file

**Validation**
4. Use insat-resampling file to resample Validation data and save to external file
6. Use imerg-resampling file to resample Validation data and save to external file
7. Use validation resampled files to RF_validation file to validate
