import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import pickle
import os


def model_train(x,y, filename2):
    """
    This function contains the training methods, on the basis of which we obtain the prediction estimate.
    :param x: dataset
    :param y: target data
    :param filename2: the file where the results are written
    """
    #Plain random forest>
    model_rf1 = RandomForestRegressor()
    model_rf1.fit(x,y)


    #GradientBoostingRegressor>
    #min_samples_leaf - The minimum number of samples in newly created leaves.
    # A split is discarded if after the split, one of the leaves would contain less then min_samples_leaf samples
    #learning_rate - learning rate shrinks the contribution of each tree by learning_rate
    #max_depth - maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.

    GBmodel_1 = GradientBoostingRegressor(min_samples_leaf=4, learning_rate=0.5, max_depth=7)
    GBmodel_1.fit(x,y)

    #Plain DecisionTreeRegressor
    line_reg1 = DecisionTreeRegressor()
    line_reg1.fit(x,y)

    #regularization
    #alpha=1.0 -  Regularization strength; must be a positive float.
    # Regularization improves the conditioning of the problem and reduces the variance of the estimates.

    reg1 = Ridge(alpha=1.0)
    reg1.fit(x,y)
    reg2 = Ridge(alpha=.5)
    reg2.fit(x,y)

    #Kmeans with randomforest
    #n_clusters - how many clusters the data will be split into
    kmeans_1 = KMeans(n_clusters=2)
    kmeans_1.fit(x)
    cluster1 = pd.Series(kmeans_1.predict(x), name='Cluster')


    train_cluster = pd.concat([x.reset_index(), cluster1], 1)

    model_rf = RandomForestRegressor()
    model_rf.fit(train_cluster,y)

    models_dir = filename2+'_models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # save result method in file
    filename = '/'.join([models_dir,'random_forest.txt'])
    pickle.dump(model_rf1, open(filename, 'wb'))

    filename = '/'.join([models_dir,'GradientBoostingRegressor.txt'])
    pickle.dump(GBmodel_1, open(filename, 'wb'))

    filename = '/'.join([models_dir,'decision_tree.txt'])
    pickle.dump(line_reg1, open(filename, 'wb'))

    filename = '/'.join([models_dir,'ridge_alpha1.0.txt'])
    pickle.dump(reg1, open(filename, 'wb'))

    filename = '/'.join([models_dir,'ridge_alpha0.5.txt'])
    pickle.dump(reg2, open(filename, 'wb'))






