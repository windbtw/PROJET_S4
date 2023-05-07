import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

##all the function for data_analysis

features = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "C", "A", "S", "M20", "GINI"]


def plot_df_PCA_reduction(galaxy_feature):
    """
    :param galaxy_feature: the galaxies will be colored accordingly to this parameter.
     Eitehr 'TYPE', 'Arms' or 'Elleptique_type'.
    :return: 2D PCA reduction of the dataset on PC1 and PC2
    """

    df = pd.read_csv('dat files/Base_fourier.dat', sep=" ")
    df1 = pd.read_csv('dat files/Base_x.dat', sep=" ")
    df_merge = pd.merge(df, df1, on='Name')
    df = df_merge.copy()
    pca = PCA()
    components = pca.fit_transform(df[features])
    plt.scatter(components[:, 0], components[:, 1], c=df[galaxy_feature])
    print(np.round(pca.explained_variance_ratio_[0] * 100))
    plt.xlabel("PC1 (" + str(np.round(pca.explained_variance_ratio_[0] * 100)) + "%)")
    plt.ylabel("PC2 (" + str(np.round(pca.explained_variance_ratio_[1] * 100)) + "%)")
    plt.show()


def plot_df_LDA_reduction(galaxy_feature):
    """
    :param galaxy_feature: the galaxies will be colored accordingly to this parameter.
     Eitehr 'TYPE', 'Arms' or 'Elleptique_type'.
    :return: LDA reduction of the dataset
    """

    df = pd.read_csv('dat files/Base_fourier.dat', sep=" ")
    df1 = pd.read_csv('dat files/Base_x.dat', sep=" ")
    df_merge = pd.merge(df, df1, on='Name')
    df = df_merge.copy()
    X = df[features]
    y = df[galaxy_feature]
    lda = LDA()
    X_lda = lda.fit_transform(X, y)

    if galaxy_feature == 'TYPE':
        plt.scatter(X_lda, np.zeros_like(y), c=y)
        plt.xlabel('LDA')
        plt.yticks([])
        plt.show()

    else:
        plt.scatter(X_lda[:, 0], X_lda[:, 1], c=df[galaxy_feature])
        plt.xlabel("LDA1(" + str(np.round(lda.explained_variance_ratio_[0] * 100)) + "%)")
        plt.ylabel("LDA2(" + str(np.round(lda.explained_variance_ratio_[1] * 100)) + "%)")
        plt.show()


features = ["C", "A", "S", "M20", "GINI"]


def pred_PCA(data_to_pred, feature_to_pred, nb_PC=2):
    """
    :param data_to_pred: file path (.dat format) of the data with which you want to predict the morphology.
    BE SURE THAT THE FOLLOWING FEATURES ARE IN THE FILE :
    "C", "A", "S", "M20", "GINI"
    :param feature_to_pred: Morphological feature you want to predict. Either 'Arms' or 'Type'
    :param nb_PC: number of principal components you want to use (default value is 2).
    :return: values predicted for each sample in input data
    """

    """Sorting data base"""

    df = pd.read_csv('dat files/Base_fourier.dat', sep=" ")
    df1 = pd.read_csv('dat files/Base_x.dat', sep=" ")
    df_merge = pd.merge(df, df1, on='Name')
    df = df_merge.copy()
    y = df[feature_to_pred]
    X = df[features]

    """Sorting data to predict"""

    DATA_TO_PREDICT = pd.read_csv(data_to_pred, sep=" ")
    df_pred = DATA_TO_PREDICT.copy()
    X_pred = df_pred[features]

    """Training prediction model"""

    pca = PCA()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    X_reduced_train = pca.fit_transform(scale(X_train))
    X_reduced_test = pca.transform(scale(X_test))[:, :nb_PC]
    regr = LinearRegression()
    regr.fit(X_reduced_train[:, :nb_PC], y_train)
    pred = regr.predict(X_reduced_test)
    print("The mean error of this model is :", np.sqrt(mean_squared_error(y_test, pred)))

    """Prediction"""

    X_red_pred = pca.transform(scale(X_pred))[:, :nb_PC]
    y_pred = regr.predict(X_red_pred)
    return y_pred


def pred_LDA(data_to_pred, feature_to_pred):
    """
    :param data_to_pred: file path (.dat format) of the data with which you want to predict the morphology.
    BE SURE THAT THE FOLLOWING FEATURES ARE IN THE FILE :
    "C", "A", "S", "M20", "GINI"
    :param feature_to_pred: Morphological feature you want to predict. Either 'Arms' or 'Type' or 'Elleptique_type'
    :return: values predicted for each sample in input data
    """

    """Sorting data base"""

    df = pd.read_csv('dat files/Base_fourier.dat', sep=" ")
    df1 = pd.read_csv('dat files/Base_x.dat', sep=" ")
    df_merge = pd.merge(df, df1, on='Name')
    df = df_merge.copy()
    y = df[feature_to_pred]
    X = df[features]

    """Sorting data to predict"""

    DATA_TO_PREDICT = pd.read_csv(data_to_pred, sep=" ")
    df_pred = DATA_TO_PREDICT.copy()
    X_pred = df_pred[features]

    """Training prediction model"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
    clf = LDA()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_train, y_train)
    print("The prediction is", round(accuracy, 2) * 100, "% accurate")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    """Prediction"""

    y_pred = clf.predict(X_pred)
    return y_pred




