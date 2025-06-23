follow the steps to get results

For training I used file from 28th May and for validation i used 30th May file

Data Reprocessing

Use insat-resampling file to resample training data and save to external file
Use imerg-resampling file to resample label data and save to external file
Model Training

Use saved files to train the model by splitting the dataset into training and test sets through ANN-model file
Validation

Use insat-resampling file to resample Validation data and save to external file
Use imerg-resampling file to resample Validation data and save to external file
Use  ANN_validation file to predict precipitationand its comparasion to IMERG true precipitationz
