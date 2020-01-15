# Object distance for Graphs
from abc import ABCMeta, abstractmethod
class distance:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        pass
 
    
    @abstractmethod
    def node_dis(self,x,y):
        pass
    @abstractmethod    
    def node_sim(self,x,y):
        pass
    
    @abstractmethod    
    def edge_dis(self,x,y):
        pass
    
    @abstractmethod    
    def edge_sim(self,x,y):
        pass
    
    @abstractmethod
    def get_Instance(self,name):
        pass