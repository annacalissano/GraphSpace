#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include<vector>
#include<set>
#include<map>
#include<tuple>
#include<algorithm>
#include<functional>
#include<execution>
#include<random>
#include<cmath>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues> 
#include "Graph.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

template<class T>
using GraphPointer= std::shared_ptr<Graph<T>>;

template<class T>
using attr_type=std::vector<T>;

template<class T>
using geodesic=Eigen::Matrix<T,Eigen::Dynamic,1>;

/**
*This function is the sum from i=a to b of i
*/
int summation(int a, int b){
    int sum=0;
    for(int i=a; i<=b; ++i)
        sum+=i;
        
    return sum;
}

/**
*This function return a set that contains all the key of a map
*/
template<class K, class V>
std::set<K> get_keys(const std::map<K,V> & map){

   std::set<K> result;
   
   for( auto elem : map)
       result.insert(elem.first);
   
   return result;
}

/**
*This function returns the linear combination ax*x+ay*y (version for integers)
*/
attr_type<int> add_lc(const double ax, attr_type<int>  x, const double ay, attr_type<int>  y){ 

    attr_type<int> result;
      
    if(x.size()==0 && y.size()==0)
        return result;
    
    if(x.size()==0)
       x.resize(y.size(),0);
       
    if(y.size()==0)
       y.resize(x.size(),0);
    
    result.resize(x.size());
    
    for(auto i=0; i<result.size(); ++i)
        result[i]=round(ax*x[i]+ay*y[i]);
    
    return result;
}

/**
*This function returns the linear combination ax*x+ay*y
*/
template<class T>
attr_type<T> add_lc(const double ax, attr_type<T>  x, const double ay, attr_type<T>  y){ 

    attr_type<T> result;
      
    if(x.size()==0 && y.size()==0)
        return result;
    
    if(x.size()==0)
       x.resize(y.size(),0);
       
    if(y.size()==0)
       y.resize(x.size(),0);
       
       result.resize(x.size());
    
    for(auto i=0; i<result.size(); ++i)
        result[i]=ax*x[i]+ay*y[i];
   
    
    return result;
}


/**
*This function is necessary to compute the mean
*/
template<class T>
GraphPointer<T> add(double ax, const GraphPointer<T> & A, double ay, const GraphPointer<T> & B){ 

   std::map<std::pair<int,int>, attr_type<T>> new_map;
   
   auto y=B->get_graph_map();  
   auto x=A->get_graph_map();  
   
   auto adjX=A->get_adj();
   auto adjY=B->get_adj();
   int nY=B->get_n_nodes();
  
   auto x_keys=get_keys(x);
   auto y_keys=get_keys(y);
   
   
   for( auto i=0; i<nY; ++i){
      if(x_keys.find(std::make_pair(i,i))!=x_keys.end() && y_keys.find(std::make_pair(i,i))!=y_keys.end())
         new_map[std::make_pair(i,i)]=add_lc(ax,x[std::make_pair(i,i)],ay,y[std::make_pair(i,i)]);
      else if(x_keys.find(std::make_pair(i,i))!=x_keys.end() && !(y_keys.find(std::make_pair(i,i))!=y_keys.end()))
         new_map[std::make_pair(i,i)]=add_lc(ax,x[std::make_pair(i,i)],ay,std::vector<T>());
      else if(!(x_keys.find(std::make_pair(i,i))!=x_keys.end()) && y_keys.find(std::make_pair(i,i))!=y_keys.end())
          new_map[std::make_pair(i,i)]=add_lc(ax,std::vector<T>(),ay,y[std::make_pair(i,i)]);
          
      std::set<int> linked_nodes;
      if(adjX[i].size()!=0 and adjY[i].size()!=0){
          linked_nodes.insert(adjX[i].begin(),adjX[i].end());
          linked_nodes.insert(adjY[i].begin(),adjY[i].end());
      }
      else{
          if(adjX[i].size()!=0 and adjY[i].size()==0)
              linked_nodes.insert(adjX[i].begin(),adjX[i].end());
          if(adjX[i].size()==0 and adjY[i].size()!=0)
              linked_nodes.insert(adjY[i].begin(),adjY[i].end());    
      }
      
      for (auto j : linked_nodes){
          if(x_keys.find(std::make_pair(i,j))!=x_keys.end() && y_keys.find(std::make_pair(i,j))!=y_keys.end())
              new_map[std::make_pair(i,j)]=add_lc(ax, x[std::make_pair(i,j)],ay,y[std::make_pair(i,j)]);
          else if (y_keys.find(std::make_pair(i,j))==y_keys.end())
              new_map[std::make_pair(i,j)]=add_lc(ax, x[std::make_pair(i,j)],ay,std::vector<T> ());
          else if (x_keys.find(std::make_pair(i,j))==x_keys.end())
              new_map[std::make_pair(i,j)]=add_lc(ax, std::vector<T> (), ay,y[std::make_pair(i,j)]);   
      
      }
      
   } 
   
   return std::make_shared<Graph<T>>(Graph<T>(new_map,true));
}

/**
*This function, given two graphs, adds empty nodes to the graphs with a lower number of nodes, until it has the size of the other
*/
template<class T>
void grow_and_set( GraphPointer<T> & X, GraphPointer<T> & Y){

   int nX=X->get_n_nodes();
   int nY=Y->get_n_nodes();
   int n=std::max(nX,nY);
   
   int n_attr=(X->get_graph_map())[std::make_pair(0,0)].size();
   std::vector<T> new_attribute = std::vector<T>(n_attr,0.);
   
   if(nX<n)
       X->grow(n,new_attribute);
   
   if(nY<n)
       Y->grow(n,new_attribute);
   
} 

/**
*This function, given a dataset, centers its column with respect to their mean
*/
Eigen::MatrixXd center(const Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> & dataset){

   Eigen::MatrixXd centered=(dataset.cast<double>()).rowwise() - (dataset.cast<double>()).colwise().mean();
   return centered;
   
}

/**
*This function, given a dataset, centers its column with respect to their mean
*/
Eigen::MatrixXd center(const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> & dataset){

   Eigen::MatrixXd centered=(dataset.cast<double>()).rowwise() - (dataset.cast<double>()).colwise().mean();
   return centered;
   
}

/**
*This function, given a dataset, centers its column with respect to their mean
*/
Eigen::MatrixXd center(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> & dataset){

   Eigen::MatrixXd centered=dataset.rowwise() - dataset.colwise().mean();
   return centered;
   
}

/**
*This function computes the covariance matrix
*/
template<class T>
Eigen::MatrixXd cov(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & dataset){
  
    Eigen::MatrixXd centered = center(dataset);
    return (centered.adjoint() * centered) / double(centered.rows());

}

/**
*This function scales a dataset
*/
template<class T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> scale_matrix(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> & dataset){

    Eigen::MatrixXd scale=center(dataset);
    Eigen::MatrixXd covariance=cov(dataset);
    
    for(int i=0; i<scale.cols(); ++i)
       for(int j=0; j<scale.col(i).size() ; ++j)
          if(covariance(i,i)!=0)
            scale.col(i)[j]/=sqrt(covariance(i,i));
     
    return scale.cast<T>();

}


/**
*This function computes the Principal Component of a dataset
*/
template<class T>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> PCA(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> dataset, int n_comp){

    int n=dataset.rows();
    int p=dataset.cols();
    
    py::object sklrn =py::module::import("sklearn.decomposition"); 
    
    py::object pca = sklrn.attr("PCA")(n_comp);
    py::object temp_scores= pca.attr("fit_transform")(dataset);
    Eigen::MatrixXd scores=temp_scores.cast<Eigen::MatrixXd>();
    Eigen::VectorXd  vals=(pca.attr("explained_variance_ratio_")).cast<Eigen::VectorXd>();
    Eigen::MatrixXd vecs=pca.attr("components_").cast<Eigen::MatrixXd>();
   
    auto result=std::make_tuple(scores,vecs,vals);
    
    return result;   
}

/**
*This function reconstruct a network from a vector.The element of the vector should be store in a certain order.
*/
template<class T>
GraphPointer<T> give_me_a_network(const geodesic<T> & geo,int v_a, int e_a, int N, bool oriented){
   
   int count_elem=0;
  
   std::map<std::pair<int,int>, attr_type<T>> graph_map;
   
  
  for(int i=0; i<N; i++){
     for(int j=0; j<N; j++){
        if(i==j)
            for(int k=0; k<v_a; k++){
               graph_map[std::make_pair(i,j)].push_back(geo[count_elem]);
               count_elem++;               
            }    
        else
            for(int k=0; k<e_a; k++){
               graph_map[std::make_pair(i,j)].push_back(geo[count_elem]); 
               count_elem++;
            }          
     }

  }
   
   
   return std::make_shared<Graph<T>>(Graph<T>(graph_map,oriented));


}

#endif 
