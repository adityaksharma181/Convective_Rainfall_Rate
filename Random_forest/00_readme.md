follow the steps to get results

_For training I used file from 28th May and for validation i used 30th May file_

kaggle link: https://www.kaggle.com/code/aditxaks/rf-single-file-py

**Data Reprocessing**
1. Use insat-resampling file to resample training data and save to external file
2. Use imerg-resampling file to resample label data and save to external file

**Model Training**
1. Use saved files to train the model by splitting the dataset into training and test sets through RF-model file

**Validation**
1. Use insat-resampling file to resample Validation data and save to external file
2. Use imerg-resampling file to resample Validation data and save to external file
3. Use validation resampled files to RF_validation file to validate


