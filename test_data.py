import os
import pickle

def model_test(x, filename2):
    """
    :param x: dataset
    :param filename2:the file where the results are written
    :return: The function returns the names of the models to the file and targets
    """
    target = []
    modelnames = []
    for root, d, filenames in os.walk(filename2+'_models'):
        for f in filenames:
            loaded_model = pickle.load(open('/'.join([root,f]), 'rb'))
            modelnames.append(f[:-4])
            target.append(loaded_model.predict(x))
    return modelnames, target


