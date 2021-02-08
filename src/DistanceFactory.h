#ifndef DISTANCEFACTORY_H_INCLUDED
#define DISTANCEFACTORY_H_INCLUDED
#include <memory>
#include <string>
#include <iostream>
#include "euclidean.h"

template<class T>
using attr_type=std::vector<T>;

template<class T>
using DistancePointer = std::unique_ptr<distance<T>>;

namespace Distance
{
  
  enum class distances {euclidean};
  
  
  template< class T>
  DistancePointer<T> distanceFactory(distances t);
  
  template<class T>
  class distanceHolder{
  public:
    distanceHolder()=default;
    distanceHolder(distances d_id):my_distance(distanceFactory<T>(d_id)){};
    void setDistance(distances d_id);
  private:
    DistancePointer<T> my_distance; 
  };
  
  template<class T>
  DistancePointer<T> distanceFactory(distances t){
    switch (t){
    case distances::euclidean : return std::make_unique<euclidean<T>>();
    default: return std::unique_ptr<distance<T>>();
    }
  } 
  
  template< class T>
  void distanceHolder<T>::setDistance(distances d_id){
    this->my_distance=distanceFactory<T>(d_id);
  }
  
}

#endif
