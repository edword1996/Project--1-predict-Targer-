import pandas as pd
import os
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)


def pars(files, file_target):
    """
    :param files: les that are in the folder
    :param file_target: this file which contains targets
    :return: returns everything that was in the directory
    """
    NUMBER_OF_LETTERS = -12
    target = None
    filenames = []
    for (_, _, fn) in os.walk(files):
        for file in fn:
            filenames.append(file[:NUMBER_OF_LETTERS])
            target = pd.read_excel(file_target, index_col=False)
    return filenames, fn, target


# Preprocessing:
files_1 = ('D:\learning\predict_value\new_data')


def dataframe_in_dict(fn, NUMBER_OF_LETTERS, root):
    """
    :param fn: file name
    :param NUMBER_OF_LETTERS:quantity letters in dataset
    :param root:root directory
    :return: dictionary with dataset
    """
    dict_data = {}
    for f in fn:
        if f[:NUMBER_OF_LETTERS] not in dict_data:
            dict_data.update({f[:NUMBER_OF_LETTERS]: [pd.read_csv(root + f)]})
        else:
            dict_data[f[:NUMBER_OF_LETTERS]].append(pd.read_csv(root + f))
    return dict_data


def pars_target(fn, NUMBER_OF_LETTERS, root, target):
    """

    :param fn:file name
    :param NUMBER_OF_LETTERS:quantity letters in dataset
    :param root:root directory
    :param target: target variable
    :return:2 lists, data_features = [] - data only with features, data_target = [] - target variable
    """

    data_features = []
    data_target = []
    proc = dataframe_in_dict(fn, NUMBER_OF_LETTERS, root)
    for key, val in proc.items():
        data_features.append(pd.concat(val, ignore_index=True))
        for targetIndex in range(target.shape[0]):
            if key[9:] == target.loc[targetIndex, 'ID'][9:]:
                data_target.append(target.loc[targetIndex])
    data_target = pd.concat(data_target, 1).T
    return data_features, data_target

