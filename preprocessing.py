import pandas as pd

def mean(data_features, data_target):
    """

    :param data_features: data only with features and without target
    :param data_target: target variable with ID
    :return: concatenated dataset with averaged values in each column
    """
    data_stats = {'Mean': []}
    for df in data_features:
        data_stats['Mean'].append(df.mean())
    data_stats['Mean'] = pd.concat(data_stats['Mean'], 1).T
    data_mean = pd.concat([data_stats['Mean'], data_target], 1)
    data_mean = data_mean.drop(['ID'], 1)
    data_mean['target1'] = data_mean['target1'].astype('float64')
    data_mean['target2'] = data_mean['target2'].astype('float64')
    return data_mean

def scale(A):
    """
    :param A: variables
    :return:  normalized data
    """
    return (A - A.min()) / (A.max() - A.min())

def drop_feature(data_features):
    """
    :param data_features:data only with features and without target
    :return: data with drop ['x15', 'x16'] columns
    """
    for i in range(len(data_features)):
        df = data_features[i]
        df = df[(df['x15'] == 0) & (df['x16'] == 0)]
        df = df.drop(['x15', 'x16'], 1)
        data_features[i] = df
    return data_features

# Drop tails
def drop_tails(data_features):
    """
    :param data_features: data only with features and without target
    :return: returns a dataset with stripped tails
    """
    for i in range(len(data_features)):
        df = data_features[i]
        df = df[(df['x13'] < 1000) & (df['x6'] < 20)]
        data_features[i] = df
    return data_features