# GraphSpace
Graph Space Package both the python version and the C++ one.
First you can fine the part refferred to the python version and after the part relative to the C++ version.

This python package allow study population of networks in Graph Space [1] as a special case of Structure Space introduced in [2].
The package is organized as follow:

core:
- Graph: class creating a graph object
- GraphSet: class creating a graph set
- Mean: class computing the Frechèt Mean and the variance of a set of graphs [2]
- MeanIterative: bootstrapped version of Frechèt Mean [2]

distance:
- euclidean: compute the euclidean distance between nodes and vector attributes

matcher:
- Matcher: parent class created to match two graphs
- child classes: whatever algorithm used to match two different networks, based on topology, node attributes and edge attributes. 
Child class gives back a reordered node sequence optimizing the matching criteria
    ID: identity match
    GA: graduate assignment match[2]
    GAS, GAS1: solving directly the optimization problem (GAS1 tackles the linearized version)
- alignment: class aligning two graphs with a specified matcher

AlignCompute:
- mean_aac: compute the Frechet Mean with the AAC algorithm and a given matcher [1]
- gpca_aac: compute the Geodesic Principal Components with AAC and a given matcher [1]
- ggr_aac: compute the Generalized Geodesic Regression with AAC and a given matcher [1]

Acknowledgement: A great acknowledgment goes to Brijnesh Jain, whose code is used as a starting point for this package. Gianluca Zeni has been massively contributing on the implementation of the python package.

The C++ version of the code have been coded by Rossi Noemi and Savino Matteo and it is the implementation of the same package, but in C++ language.
All the code is in the src folder that schematically contains the following:

basic notion:
- Graph: class creating a graph object
- GraphSet: class creating a graph set with a function to compute the  Frechèt Mean

distance:
- euclidean: compute the euclidean distance between nodes and vector attributes

matcher:
- Matcher: parent class created to match two graphs
- child classes: whatever algorithm used to match two different networks, based on topology, node attributes and edge attributes. 
Child class gives back a reordered node sequence optimizing the matching criteria
    ID: identity match
    GA: graduate assignment match[2]

gpc:
- gpc: gpc class to compute the Geodesic Principal Components with AAC and a given matcher [1]

[1] Calissano, Anna, Feragen, Aasa and Vantini Simone "Graph Space: Geodesic Principal Components for aPopulation of Network-valued Data" MOX Report (2020)

[2]    B.  Clapper.  “Munkres  implementation  for  Python”.  In:  (2008).url:https://github.com/bmc/munkres.

[3] Jain, Brijnesh J., and Klaus Obermayer. "Structure spaces." Journal of Machine Learning Research 10.Nov (2009): 2667-2714.

[4] Gold, Steven, and Anand Rangarajan. "A graduated assignment algorithm for graph matching." IEEE Transactions on pattern analysis            and machine intelligence 18.4 (1996): 377-388.

[5]    Wenzel  Jakob,  Jason  Rhinelander,  and  Dean  Moldovan.pybind11  –  Seamless  operabilitybetween C++11 and Python. 2017. url:https://github.com/pybind/pybind11

[6]    Ryan A. Rossi and Nesreen K. Ahmed. “The Network Data Repository with InteractiveGraph Analytics and Visualization”. In:AAAI. 2015.url:http://networkrepository.com


