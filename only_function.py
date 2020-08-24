import pandas as pd
from sklearn.model_selection import train_test_split
from new_data.pars_data import pars, pars_target
from new_data.preprocessing import drop_feature, drop_tails, mean, scale
from new_data.train_data import model_train
from new_data.test_data import model_test
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# variables for function
filesdir = ('D:\learning\predict_value\\new_data\Data_Andrew_181_Files\\')
NUMBER_OF_LETTERS = -12
filenames, fn, target = pars(filesdir, 'D:\learning\predict_value\\new_data\Andrew_181_Files - копия.xlsx')
data_features, data_target = pars_target(fn, NUMBER_OF_LETTERS, filesdir, target)
data_features = drop_feature(data_features)
data_features = drop_tails(data_features)
data_mean = mean(data_features, data_target)
data_mean = scale(data_mean)
print(data_mean)

# split data train - test
train, test = train_test_split(data_mean, test_size=0.2, random_state=47)

model_train(train.drop(['target1','target2'],1),train['target1'], 'target1')
model_train(train.drop(['target1','target2'],1),train['target2'], 'target2')

# save result only fit (target - 1)in file redicted_target1.txt
f_target1 = open('predicted_target1.txt', 'w')
result = model_test(test.drop(['target1','target2'],1), 'target1')
for i in range(len(result[0])):
    f_target1.write(result[0][i]+":\n")
    for value in result[1][i]:
        f_target1.write(str(value)+"\n")
    f_target1.write("\n")
f_target1.close()

# save result only fit (target - 2) in file redicted_target2.txt
f_target2 = open('predicted_target2.txt', 'w')
result = model_test(test.drop(['target1','target2'],1), 'target2')
for i in range(len(result[0])):
    f_target2.write(result[0][i]+":\n")
    for value in result[1][i]:
        f_target2.write(str(value)+"\n")
    f_target2.write("\n")
f_target2.close()






