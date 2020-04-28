from distance import distance
import math

# Weighted euclidean distance
# every element of the vector is wighted differently
# Attention: this distance only work for scalar attributes on both nodes and edges!

class weighted_euclidean(distance):

    def __init__(self,weights):
        distance.__init__(self)
        # the idea is to have a weight for every component
        # the weights should be n^2, one for every element in the networks
        self.weights=weights

    # @staticmethod
    def the_dis(self, x, y):
        # two integer
        if (not isinstance(x, list) and not isinstance(y, list)):
            _dis = math.pow((x - y), 2)
            return _dis
        # two lists
        if (isinstance(x, list) and isinstance(y, list)):
            nx = len(x)
            ny = len(y)
            # both null
            if (nx == 0 and ny == 0):
                return 0
            else:
                # one null
                if (nx == 0):
                    return self.the_sim(y, y)
                if (ny == 0):
                    return self.the_sim(x, x)

                # different length
                else:
                    if (nx <= ny):
                        n = ny
                        x = x + [0] * (n - nx)
                    else:
                        n = nx
                        y = y + [0] * (n - ny)

                    _dis = 0
                    for i in range(n):
                        _dis += math.pow((x[i] - y[i]), 2)
                    return _dis
        # One list one integer
        if (isinstance(x, list) and not isinstance(y, list)):
            n = len(x)
            y = [y] + [0] * (n - 1)
            _dis = 0
            for i in range(n):
                _dis += math.pow((x[i] - y[i]), 2)
            return _dis
        # One list one integer
        if (not isinstance(x, list) and isinstance(y, list)):
            n = len(y)
            x = [x] + [0] * (n - 1)
            _dis = 0
            for i in range(n):
                _dis += math.pow((x[i] - y[i]), 2)
            return _dis

    # Compute pointwise product
    # @staticmethod
    def the_sim(self, x, y):
        # empty
        if (x == None or y == None):
            return 0
        if (x == None and y == None):
            print('Give me at least one non empty vector!')
            return 0
        # two integer
        if (not isinstance(x, list) and not isinstance(y, list)):
            _sim = x * y
            return _sim
        # two list
        if (isinstance(x, list) and isinstance(y, list)):
            if (len(x) <= len(y)):
                n = len(y)
                x = x + [0] * (n - len(x))
            else:
                n = len(x)
                y = y + [0] * (n - len(y))
            _sim = 0
            for i in range(n):
                _sim += x[i] * y[i]
            return _sim
        # one list one integer
        if (isinstance(x, list) and not isinstance(y, list)):
            n = len(x)
            y = [y] + [0] * (n - 1)
            _sim = 0
            for i in range(n):
                _sim += x[i] * y[i]
            return _sim
        if (not isinstance(x, list) and isinstance(y, list)):
            n = len(y)
            x = [x] + [0] * (n - 1)
            _sim = 0
            for i in range(n):
                _sim += x[i] * y[i]
            return _sim

    def node_dis(self, x, y):
        return self.the_dis(x, y)

    def node_sim(self, x, y):
        return self.the_sim(x, y)

    def edge_dis(self, x, y):
        return self.the_dis(x, y)

    def edge_sim(self, x, y):
        return self.the_sim(x, y)

    def get_Instance(self, name):
        return 'Euclidean'