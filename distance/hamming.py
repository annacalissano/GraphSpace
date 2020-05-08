from distance import distance
import scipy.spatial.distance
import numpy as np
from abc import ABCMeta, abstractmethod
import math


class hamming(distance):

    def __init__(self):
        distance.__init__(self)

    # @staticmethod
    def the_dis(self, x, y):
        # two integer
        if np.ndim(x)==0 and np.ndim(y)==0:
            _dis = scipy.spatial.distance.hamming(x,y)
            return _dis
        # One list one integer
        elif isinstance(x, list) and np.ndim(y)==0:
            n = len(x)
            y = [y] + [0] * (n - 1)
            _dis = scipy.spatial.distance.hamming(y, x)
            return _dis
        # One list one integer
        elif isinstance(y, list) and np.ndim(x)==0:
            n = len(y)
            x = [x] + [0] * (n - 1)
            _dis = scipy.spatial.distance.hamming(y, x)
            return _dis
        # two lists
        elif (isinstance(x, list) and isinstance(y, list)):
            nx = len(x)
            ny = len(y)
            # both null
            if (nx == 0 and ny == 0):
                return 0
            else:
                # one null
                if (nx == 0):
                    return scipy.spatial.distance.hamming(y,[0]*ny)
                if (ny == 0):
                    return scipy.spatial.distance.hamming(x,[0]*nx)

                # different length
                else:
                    if (nx <= ny):
                        n = ny
                        x = x + [0] * (n - nx)
                    else:
                        n = nx
                        y = y + [0] * (n - ny)

                    _dis = scipy.spatial.distance.hamming(y,x)
                    return _dis
        # two arrays (as rows/columns of pd.DataFrame) of equal length (as in GAS):
        else:
            _dis = scipy.spatial.distance.hamming(y,x)
        return _dis



    def node_dis(self, x, y):
        return self.the_dis(x, y)

    def edge_dis(self, x, y):
        return self.the_dis(x, y)

    def get_Instance(self):
        return 'hamming'