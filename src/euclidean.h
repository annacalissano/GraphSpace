#ifndef EUCLIDEAN_H_INCLUDED
#define EUCLIDEAN_H_INCLUDED

#include <cmath>      
#include <math.h>      
#include <iostream> 	
#include "distance.h"

/**
*This alias is used to indicate the type of the attribute of a vertex or an edge. The T template indicates the type of the elements contained in the attribute
*/
template<class T>
using attr_type=std::vector<T>;

/**
* Class that inherit form the class distance and that implements the euclidean distance 
*/

template<class T>
class euclidean: public distance<T>{

public : 

/**
* function to compute pointwise product
*/

T the_sim(attr_type<T> x,attr_type<T> y);
    
/**
* function to compute the distance between two nodes or two edges.
*/   
T the_dis(attr_type<T> x,attr_type<T> y); 

T node_dis(attr_type<T> x,attr_type<T> y) override;
 
T node_sim(attr_type<T> x,attr_type<T> y) override;
 
T edge_dis(attr_type<T> x,attr_type<T> y) override;

T edge_sim(attr_type<T> x,attr_type<T> y) override;
/**
* This method returns the name of the distance used
*/
std::string get_Instance() override;

};


template<class T>
T euclidean<T>::the_dis(attr_type<T> x,attr_type<T> y){
  
  T dis = 0; 

// If two integer 
if(x.size()==1 && y.size()==1){
  dis = std::pow(x[0]-y[0],2);
}

// If x is a vector and y is a number
if(x.size()>1 && y.size()==1 ){
  int n = x.size();
  for (size_t i = 0; i < n-1; ++i){
  y.push_back(0);
  }
  for(size_t j = 0 ; j < n; ++j){
  dis += std::pow(x[j]-y[j],2);
  }
}

// If y is a vector and x is a number
if(x.size()==1 && y.size()>1){
  int n = y.size();
  for (size_t i = 0; i < n-1; ++i){
  x.push_back(0);
  }
  for(size_t j = 0 ; j < n; ++j){
  dis += std::pow(x[j]-y[j],2);
  }
}

// If two vector

if(x.size()>1 && y.size()>1){
  int ny = y.size();
  int nx = x.size();
  int n = 0;
  
  // If both null
if(nx==0 && ny==0 ){
  return 0;
}else{
  if(nx==0){
  return the_sim(y,y);
  }
  if(ny==0){
  return the_sim(x,x);
  }
  // different length
  else{
    if(nx<=ny){
      n=ny;
      for (size_t i = nx; i < n; ++i){
        x.push_back(0);
      }
    }else{
       n=nx;
       for (size_t i = ny; i < n; ++i){
          y.push_back(0);
        }
      }
    for(size_t it = 0; it < n; ++it){
      dis += std::pow(x[it]-y[it],2);
    }
  }
}
}
      
return sqrt (dis);
}

template<class T>
T euclidean<T>::the_sim(attr_type<T> x,attr_type<T> y){

// If empty 
  if(x.size()==0 || y.size()==0){
    return 0;
  }

  if(x.size()==0 && y.size()==0){
    std::cout<<"Give me at least one non empty vector!" << std::endl;
    return 0;
  }

// two list 
  if(x.size()>1 && y.size()>1){
    int ny = y.size();
    int nx = x.size();
    int n = 0;
 
    if(nx<=ny){
      n=ny;
      for (size_t i = nx; i < n; ++i){
        x.push_back(0);
      }
    }else{
       n=nx;
        for (size_t i = ny; i < n; ++i){
          y.push_back(0);
        }
      }
      T sim = 0;
    for(size_t it = 0; it < n; ++it){
      sim += x[it]*y[it];
    }
  return sim;
  }

// one vector and one integer
  if(x.size()>1 && y.size()==1){
    int n = x.size();
    for (size_t i = 0; i < n-1; ++i){
      y.push_back(0);
    } 
    T sim = 0;
    for(size_t j = 0 ; j < n; ++j){
      sim += x[j]*y[j];
    }
    return sim; 
  }
  if(x.size()==1 && y.size()>1){
    int n = y.size();
    for (size_t i = 0; i < n-1; ++i){
      x.push_back(0);
    }
    T sim = 0;
    for(size_t j = 0 ; j < n; ++j){
      sim += x[j]*y[j];
    }
    return sim;
  }

// two integer
  if(x.size()==1 && y.size()==1){
    T sim = x[0]*y[0];
    return sim;
  }
 
  return 0;
}



template<class T>
T euclidean<T>::node_dis(attr_type<T> x,attr_type<T> y){
        return the_dis(x,y);
}
    
template<class T>    
T euclidean<T>::node_sim(attr_type<T> x,attr_type<T> y){
        return the_sim(x,y);
}

template<class T>    
T euclidean<T>::edge_dis(attr_type<T> x,attr_type<T> y){
        return the_dis(x,y);
}

template<class T>    
T euclidean<T>::edge_sim(attr_type<T> x,attr_type<T> y){
        return the_sim(x,y); 
}

template<class T>
std::string euclidean<T>::get_Instance(){
	return "euclidean";
}


#endif //EUCLIDEAN_H_INCLUDED
