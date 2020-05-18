# The following class run the following algorithm approach:
# Take: X1,..,Xn observations in our Q quotient space and a prediction rule (mean, regression function ..)
# Step1: training of the prediction rule
# Step2: compute non-conformity scores, as distances from the predicted values, and save the proper quantile
# Step3: scan over a grid in the graph space, computing scores
# Step4: prediction regions include all the points of the grid with scores smaller than the computed quantile


from core import Graph,GraphSet
import numpy as np
import random
import copy
import numpy as np
from matcher import Matcher, alignment, GA, ID, GAS, GAS1

from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from nonconformist.base import RegressorAdapter
from nonconformist.nc import RegressorNc


class MyRegressorAdapter(RegressorAdapter):
    def __init__(self, model, fit_params=None):
        super(MyRegressorAdapter, self).__init__(model, fit_params)

    def fit(self, x, y):
        '''
            x is a numpy.array of shape (n_train, n_features)
            y is a numpy.array of shape (n_train)

            Here, do what is necessary to train the underlying model
            using the supplied training data
        '''
        pass

    def predict(self, x):
        '''
            Obtain predictions from the underlying model

            Make sure this function returns an output that is compatible with
            the nonconformity function used. For default nonconformity functions,
            output from this function should be predicted real values in a
            numpy.array of shape (n_test)
        '''
        pass



class conformal(object):
    
    def __init__(self,graphset,predRule,match=GAS()):
        self.X = graphset.grow_to_same_size()  # Original Dataset
        self.aX = graphset.aX        # Aligned Dataset
        self.data = self.aX.to_matrix_with_attributes()
        self.predRule = predRule
        self.match = match
        # self.var=None
        # self.cov=None

    def predBands(self, e=0.00001, alpha=0.1):
        # my_regressor = None  # Initialize an object of your regressor's type
        # model = MyRegressorAdapter(my_regressor)
        # nc = RegressorNc(model)
        # icp = IcpRegressor(nc)  # Create an inductive conformal regressor
        #
        # # Divide the data into proper training set and calibration set (70% - 30%)
        # n = self.aX.size()
        # nt = int(np.floor(n*0.7))
        # idx = np.random.permutation(n)
        # idx_train, idx_cal = idx[:nt], idx[nt:]
        #
        # # Fit the ICP using the proper training set
        # icp.fit(1, self.data[idx_train, :])
        #
        # # Calibrate the ICP using the calibration set
        # icp.calibrate(1, self.data[idx_cal, :])

        # training and calibration sets
        n = self.X.size()
        nt = int(np.floor(n * 0.7))
        idx = np.random.permutation(n)
        idx_train, idx_cal = idx[:nt], idx[nt:]

        mu = self.predRule(self.X.sublist(idx_train), self.match)
        mu.align_and_est()
        MU = mu.mean



        dataSd = pd.Series(mu.aX.to_matrix_with_attr().apply(np.std, axis=0))

        graph_dev = self.X.sublist(idx_cal)  # is it a copy? no
        for i in range(len(graph_dev.X)):
            # Align graph_dev.X[i] to MU
            match.the_grow_and_set(graph_dev.X[i], MU)
            match.match(graph_dev.X[i], MU)
            graph_dev.X[i].permute(match.f)
        data_dev = graph_dev.to_matrix_with_attr()

        res = abs(data_dev - MU.to_vector_with_attributes().iloc[0])  # MU.to_vect.. is a dataFrame
        res_norm = res / (dataSd + e)
        scores = res_norm.max(axis=1)  # L1 norm
        err = np.quantile(scores, 1 - alpha)
        erri = err * dataSd  # err in direction i

        conformal = (MU.to_vector_with_attributes() - erri).append(MU.to_vector_with_attributes() + erri)
        conformal.index = ["min", "max"]
        conformal

        return
