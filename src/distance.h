#ifndef DISTANCE_H_INCLUDED
#define DISTANCE_H_INCLUDED

#include<vector>
#include<string>

/**
*This alias is used to indicate the type of the attribute of a vertex or an edge. The T template indicates the type of the elements contained in the attribute
*/
template<class T>
using attr_type=std::vector<T>;

/**
* Pure virtual class with all the function that are redefined in the child class according to the distance you want to use 
*/
template<class T>
class distance{

public : 
/**
* Method to compute the distance between two nodes
*/
virtual T node_dis(attr_type<T> x,attr_type<T> y) =0;
 
virtual T node_sim(attr_type<T> x,attr_type<T> y) =0;

/**
* Method to compute the distance between two edges
*/
virtual T edge_dis(attr_type<T> x,attr_type<T> y) =0;

virtual T edge_sim(attr_type<T> x,attr_type<T> y) =0;

virtual std::string get_Instance() =0;

};


#endif // DISTANCE_H_INCLUDED
