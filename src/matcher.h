#ifndef MATCHER_H_INCLUDED
#define MATCHER_H_INCLUDED

#include "Graph.h"
#include "utils.h"
#include<memory>
#include<map>
#include "DistanceFactory.h"
#include<vector>
#include<string>

/**
*This alias is used to indicate the pointer to an object of class distance. The T template indicates the distance used 
*/
template<class T>
using DistancePointer = std::unique_ptr<distance<T>>;

/**
*Base abstruct class for matchers
*/
template<class T>
class matcher{

protected:

/**
*When match method is called, in this variable is stored the distance between the two matched graphs
*/
double dist;

/**
*Permutation of X to get close to Y
*/
std::vector<int> f; 

/**
*Poiter to the distance that is used in the match
*/
DistancePointer<T> distance;
public:
  
/**
*Constructor
*/
matcher() : distance(Distance::distanceFactory<T>(Distance::distances::euclidean)){};

/**
*Constructor
*/
matcher(Distance::distances _distance) : distance(Distance::distanceFactory<T>(_distance)) {};

/**
*The methods matches two graphs, saving in f the best permutation
*/   
virtual void match(GraphPointer<T> first_graph,GraphPointer<T> second_graph)=0;

/**
*The methods sets dist equal to distance of the graphs X and Y based on the permutation f.
*In order to have the right distance, you have to be sure that in f there is the right permutation vector, so before running this function you have to do the matching.
*/ 
double the_dis( GraphPointer<T>  X, GraphPointer<T>  Y);

/**
*The method returns the name of the distance used
*/  
std::string get_distance() ;

/**
*Getter for f  
*/  
std::vector<int> get_f();

/**
*Getter for dist  
*/  
double get_dist();

/**
*Setter of distance pointer  
*/  
void set_dist(Distance::distances _d);

virtual ~matcher()=default; 
};
     

template<class T>
double matcher<T>::the_dis(GraphPointer<T> X, GraphPointer<T> Y){

   if(X->get_n_nodes()!=Y->get_n_nodes())
       grow_and_set(X,Y);
     
   auto x=X->get_graph_map();
   auto y=Y->get_graph_map();
   
   std::set<std::pair<int,int>> x_keys=get_keys(x);
   std::set<std::pair<int,int>> y_keys=get_keys(y);
   
   int n=X->get_n_nodes();
   
   
   dist = 0;
  
   for( int i=0; i<n; ++i){
        int fi=f[i];
        dist+=distance->node_dis(x[std::make_pair(i,i)],y[std::make_pair(fi,fi)]);
        for(int j=i+1; j<n; ++j){
            int fj=f[j];
            if(x_keys.find(std::make_pair(i,j))!=x_keys.end() && y_keys.find(std::make_pair(fi,fj))!=y_keys.end())
                dist+=distance->edge_dis(x[std::make_pair(i,j)],y[std::make_pair(fi,fj)]);
            else{
               if(x_keys.find(std::make_pair(i,j))!=x_keys.end())
                    dist+=distance->edge_dis(x[std::make_pair(i,j)], std::vector<T>(x[std::make_pair(i,j)].size(),0));
               else if (y_keys.find(std::make_pair(fi,fj))!=y_keys.end())
                    dist+=distance->edge_dis(y[std::make_pair(fi,fj)], std::vector<T>(y[std::make_pair(fi,fj)].size(),0));
          
          }
          
         if(x_keys.find(std::make_pair(j,i))!=x_keys.end() && y_keys.find(std::make_pair(fj,fi))!=y_keys.end())
                dist+=distance->edge_dis(x[std::make_pair(j,i)],y[std::make_pair(fj,fi)]);
            else{
                if(x_keys.find(std::make_pair(j,i))!=x_keys.end())
                    dist+=distance->edge_dis(x[std::make_pair(j,i)], std::vector<T>(x[std::make_pair(j,i)].size(),0));
               else if (y_keys.find(std::make_pair(fj,fi))!=y_keys.end())
                   dist+=distance->edge_dis(y[std::make_pair(fj,fi)], std::vector<T>(y[std::make_pair(fj,fi)].size(),0));
          
         }
       }
   }
   
   return dist;
}

template<class T>
std::string matcher<T>::get_distance(){
  return distance->get_Instance();
}

template<class T>
std::vector<int> matcher<T>::get_f(){
  return f;
}

template<class T>
double matcher<T>::get_dist(){
  return dist; 
}

template<class T>
void matcher<T>::set_dist(Distance::distances _d){
   distance=Distance::distanceFactory<T>(_d); 
}

#endif // MATCHER_H_INCLUDED
