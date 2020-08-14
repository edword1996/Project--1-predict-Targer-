import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error
import scipy as sp
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
import xlrd
from scipy.interpolate import griddata
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)



NUMBER_OF_LETTERS=-12
#Делаем список файлов в тренировочной директории
filenames = []
for (_, _, fn) in os.walk('./Data_Andrew_181_Files/'):
    #f.extend(filenames)
    for file in fn:
        filenames.append(file[:NUMBER_OF_LETTERS])
target = pd.read_excel('Andrew_181_Files - копия.xlsx', index_col=False)


#Preprocessing:
dict_data = {}

for f in fn:
    if f[:NUMBER_OF_LETTERS] not in dict_data:
        dict_data.update({f[:NUMBER_OF_LETTERS] : [pd.read_csv('./Data_Andrew_181_Files/'+f)]})
    else:
        dict_data[f[:NUMBER_OF_LETTERS]].append(pd.read_csv('./Data_Andrew_181_Files/'+f))


data_features =[]
data_target =[]



for key, val in (dict_data).items():
    data_features.append(pd.concat(val,ignore_index=True))
    for targetIndex in range(target.shape[0]):
        if key[9:] == target.loc[targetIndex, 'ID'][9:]:
            data_target.append(target.loc[targetIndex])


data_target = pd.concat(data_target, 1).T

for i in range(len(data_features)):
    df = data_features[i]
    df = df[(df['x15'] == 0) & (df['x16'] == 0)]
    df = df.drop(['x15', 'x16'], 1)
    data_features[i] = df


#Drop tails
for i in range(len(data_features)):
    df = data_features[i]
    df = df[(df['x13'] < 1000) & (df['x6'] < 20)]
    data_features[i] = df


data_stats = {'Mean': []}
for df in data_features:
    data_stats['Mean'].append(df.mean())

print('Sorted data')
data_stats['Mean'] = pd.concat(data_stats['Mean'], 1).T

data_mean = pd.concat([data_stats['Mean'],data_target],1)
data_mean = data_mean.drop(['ID'],1)
data_mean['target1'] = data_mean['target1'].astype('float64')
data_mean['target2'] = data_mean['target2'].astype('float64')


##############SMOTE################################
X = data_mean.drop(['target1','target2'], 1)
y = data_mean['target1']

min_1 = X.min()
max_1 = X.max()


grid_X = np.zeros((min_1.shape[0], 100))
for i in range(min_1.shape[0]):
    grid_X[i] = np.random.random_sample(100) * (max_1.iloc[i] - min_1.iloc[i]) + min_1.iloc[i]
grid_X = grid_X.T
grid_y = griddata(X, y, grid_X, method='nearest')



data_interpol = np.hstack([grid_X,grid_y.reshape(-1,1)])




def scale(A):
    return (A - A.min())/(A.max() - A.min())
data_mean = scale(data_mean)


train, test = train_test_split(data_mean, test_size=0.2, random_state=47)


kmeans_1 = KMeans(n_clusters=2)
kmeans_1.fit(train.drop(['target1','target2'],1))

X_train = train.drop(['target1','target2'],1)
X_test = test.drop(['target1','target2'],1)
y_train = train['target1']
y_test = test['target1']




reg1 = Ridge(alpha=1.0)
reg1.fit(train.drop(['target1','target2'],1),train['target1'])
print('regularization on train 1', reg1.score(train.drop(['target1','target2'],1),train['target1']))
print('regularization on test 1', reg1.score(test.drop(['target1','target2'],1),test['target1']))


reg2 = Ridge(alpha=.5)
reg2.fit(train.drop(['target1','target2', ],1),train['target1'])
print('regularization on train 0.5', reg2.score(train.drop(['target1','target2'],1),train['target1']))
print('regularization on test 0.5', reg2.score(test.drop(['target1','target2'],1),test['target1']))




#################Plain random forest#######################
print('<Plain random forest>')
model_rf1 = RandomForestRegressor()
model_rf1.fit(train.drop(['target1','target2'],1),train['target1'])
model_rf2 = RandomForestRegressor()
model_rf2.fit(train.drop(['target1','target2'],1),train['target2'])
print('RF (train) score R2 target-1: ',model_rf1.score(train.drop(['target1','target2'],1),train['target1']),
      '\nRF (train) score R2 target-2: ',model_rf2.score(train.drop(['target1','target2'],1),train['target2']))
print('RF (test) score R2 target-1: ',model_rf1.score(test.drop(['target1','target2'],1),test['target1']),
      '\nRF (test) score R2 target-2: ',model_rf2.score(test.drop(['target1','target2'],1),test['target2']))
print('</Plain random forest>')
#################Plain random forest#######################
cluster1 = pd.Series(kmeans_1.predict(train.drop(['target1','target2'],1)), name='Cluster')
train_cluster = pd.concat([train.reset_index(), cluster1], 1)


cluster1 = pd.Series(kmeans_1.predict(test.drop(['target1','target2'],1)), name='Cluster')
test_cluster = pd.concat([test.reset_index(), cluster1], 1)



model_rf = RandomForestRegressor()
model_rf.fit(train_cluster.drop(['target1','target2'],1),train_cluster['target1'])
print('RF +1 feature with cluster: train={}; test={}'.format(model_rf.score(train_cluster.drop(['target1','target2'],1), train_cluster['target1']),
      model_rf.score(test_cluster.drop(['target1', 'target2'], 1), test_cluster['target1'])))



cluster_train = kmeans_1.predict(train.drop(['target1','target2'],1))

train1 = train[cluster_train==1]
train0 = train[cluster_train==0]


cluster_test = kmeans_1.predict(test.drop(['target1','target2'],1))

test1 = test[cluster_test==1]
test0 = test[cluster_test==0]

#################################Clusterization####################333####
print('<Random forest clusterization>:')
model1 = RandomForestRegressor()
model1.fit(train1.drop(['target1','target2'],1),train1['target1'])
model2 = RandomForestRegressor()
model2.fit(train0.drop(['target1','target2'],1),train0['target1'])
print('class 1 RF (train) R2 target-1: ',model1.score(train1.drop(['target1','target2'],1),train1['target1']),
      '\nclass 0 RF (train) score R2 target-2: ',model2.score(train0.drop(['target1','target2'],1),train0['target1']))
print('class 1 RF (test) score R2 target-1: ',model1.score(test1.drop(['target1','target2'],1),test1['target1']),
      '\nclass 0 RF (test) score R2 target-2: ',model2.score(test0.drop(['target1','target2'],1),test0['target1']))

print('</Random forest clusterization>')
#############################Clusterization##############################



print('<Plain random forest>')
model_rf1 = RandomForestRegressor()
model_rf1.fit(train.drop(['target1','target2'],1),train['target1'])
model_rf2 = RandomForestRegressor()
model_rf2.fit(train.drop(['target1','target2'],1),train['target2'])
print('RF (train) score R2 target-1: ',model_rf1.score(train.drop(['target1','target2'],1),train['target1']),
      '\nRF (train) score R2 target-2: ',model_rf2.score(train.drop(['target1','target2'],1),train['target2']))
print('RF (test) score R2 target-1: ',model_rf1.score(test.drop(['target1','target2'],1),test['target1']),
      '\nRF (test) score R2 target-2: ',model_rf2.score(test.drop(['target1','target2'],1),test['target2']))
print('</Plain random forest>')



#
print('<GradientBoostingRegressor>')
GBmodel_1 = GradientBoostingRegressor(min_samples_leaf= 4, learning_rate= 0.5, max_depth= 7)

GBmodel_1.fit(train.drop(['target1','target2'],1),train['target1'])
GBmodel_2 = GradientBoostingRegressor(min_samples_leaf= 4, learning_rate= 0.5, max_depth= 7)
GBmodel_2.fit(train.drop(['target1','target2'],1),train['target2'])

print('RF (train) score R2 target-1: ',GBmodel_1.score(train.drop(['target1','target2'],1),train['target1']),
      '\nRF (train) score R2 target-2: ',GBmodel_2.score(train.drop(['target1','target2'],1),train['target2']))
print('RF (test) score R2 target-1: ',GBmodel_1.score(test.drop(['target1','target2'],1),test['target1']),
      '\nRF (test) score R2 target-2: ',GBmodel_2.score(test.drop(['target1','target2'],1),test['target2']))


