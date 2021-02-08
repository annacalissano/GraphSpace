#ifndef ID_H_INCLUDED
#define ID_H_INCLUDED
#include "Graph.h"
#include "matcher.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <map>
#include "distance.h"
#include <memory>


/**
* Class that implements the ID matcher
*/
template<class T>
class ID: public matcher<T>{

public : 
/**
* Default constructor
*/
ID()=default;
/**
* Constructor if you want to set a specific distance 
*/
ID(Distance::distances _d):matcher<T>(_d){};
/**
* Method that takes as input two GraphPointer and compute the permutation vector that is the identity. Vector that is stored in the attribute f of the class Matcher 
*/
void match(GraphPointer<T> first_graph,GraphPointer<T> second_graph) override;

};

template<class T>
void ID<T>::match(GraphPointer<T> first_graph,GraphPointer<T> second_graph){
      
      int n=std::max(first_graph->get_n_nodes(), second_graph->get_n_nodes());
      
      this->f.resize(n); 

       
      for(int i=0; i<n; i++)
          this->f[i]=i;
      
      this->dist=matcher<T>::the_dis(first_graph,second_graph);
          
}
#endif //ID_H_INCLUDED
