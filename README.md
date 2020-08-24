# Target predict
# Task description
 Task:Scope of this project is to predict as precisely as possible the values of the targets (Target1,Target2) relative to the validation set.
 
# Information about project
The project consists of 5 modules:
- only_function.py;
It's main module. This module stores all other modules and undergoes operations of writing the results to a file.
- pars_data.py
This module is responsible for the fact that it would qualitatively and completely open all files for their further work.
- preprocessing.py
This module is responsible for normalizing and processing data to a state where it can already be trained.
- test_data.py
Here the data is trained on different models.
- train_data.py
We have result on test data.

The main module to run is called only_function.py. It contains all the other modules. After you run the project, you will get 2 folders and 2 files (txt). The folders that you will see after launching they are called (target1_models and target2_models) these folders contain models but you do not need them. To view the result, you need files named (predicted_target1.txt and predicted_target2.txt). You will also see them after launch.


 
 # Download project
 In order for the project to be cloned to the current folder, you need to open the command line and run the commandу (git clone 
 https://github.com/edword1996/Project--1-predict-Target.git)
 
# Package Installation

In order to run the program, you must install Python 3.5.6

You can download and install python here https://www.python.org/downloads/

Installation Instructions https://www.howtogeek.com/197947/how-to-install-python-on-windows/

Requirements install:
```
cycler==0.10.0
h5py==2.10.0
imbalanced-learn==0.7.0
joblib==0.16.0
kiwisolver==1.2.0
matplotlib==3.2.2
numpy==1.19.0
pandas==1.0.5
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2020.1
PyYAML==5.3.1
scikit-learn==0.23.1
scipy==1.5.1
seaborn==0.10.1
six==1.15.0
threadpoolctl==2.1.0
xlrd==1.2.0
```
All the libraries above are installed using (```pip install -r requirements.txt```) the command line.

# Usage
To start a project, open a command prompt in the folder where the project is located.
```
python only_result.py
```
# Result
Regularisation
- regularization on train 1 0.029381506064767082
- regularization on test 1 -0.02625309831985545
- regularization on train 0.5 0.036217354495948095
- regularization on test 0.5 -0.03756384660277923

Plain random forest
 
- RF (train) score R2 target-1:  0.8447448502287112 
- RF (train) score R2 target-2:  0.8238899756668032
- RF (test) score R2 target-1:  -0.11845883266909829 
- RF (test) score R2 target-2:  -0.05342851189771025

Random forest +1 feature with cluster: 
- train=0.8587967267824484; 
- test=0.04600999234721792

Random forest clusterization:
- class 1 RF (train) R2 target-1:  0.8453296005471305 
- class 0 RF (train) score R2 target-2:  0.8297992080836059
- class 1 RF (test) score R2 target-1:  -0.12388571038685892 
- class 0 RF (test) score R2 target-2:  0.1575205882352929
 

Plain random forest
- RF (train) score R2 target-1:  0.847827046008671 
- RF (train) score R2 target-2:  0.8350441095953473
- RF (test) score R2 target-1:  -0.17701133485505527 
- RF (test) score R2 target-2:  0.04073105502109964


GradientBoostingRegressor
- (train) score R2 target-1:  0.9931652227254302 
- (train) score R2 target-2:  0.9626017198476138
- (test) score R2 target-1:  -0.22131367683256808 
- (test) score R2 target-2:  -0.38876850909475724
 

# Conclusion

Pay attention to the results. As mentioned earlier, we have a pretty good result on the training sample. That is, the model learns well.
But we see a bad even negative result on the test. This is the so-called retraining (overfitting).
As a conclusion, we have a very powerful model, aggressive learning rates, based on this, and a bad test.
