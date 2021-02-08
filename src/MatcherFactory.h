#ifndef MATCHERFACTORY_H_INCLUDED
#define MATCHERFACTORY_H_INCLUDED
#include <memory>
#include <string>
#include <iostream>
#include "matcher.h"
#include "ID.h"
#include "GA.h"

template<class T>
using attr_type=std::vector<T>;

template<class T>
using MatcherPointer = std::unique_ptr<matcher<T>>;

namespace Matcher
{
  enum class matchers {GA,ID};
  
  template< class T>
  MatcherPointer<T> matcherFactory(matchers t);
  
  template<class T>
  class matcherHolder{
  public:
    matcherHolder()=default;
    matcherHolder(matchers m_id):my_matcher(matcherFactory<T>(m_id)){};
    void setMatcher(matchers m_id);
  private:
    MatcherPointer<T> my_matcher; 
  };
  
  template<class T>
  MatcherPointer<T> matcherFactory(matchers t, Distance::distances d){
    switch (t){
    case matchers::GA : return std::make_unique<GA<T>>(d);
    case matchers::ID : return std::make_unique<ID<T>>(d);
    default: return std::unique_ptr<matcher<T>>();
    }
  } 
  
  template< class T>
  void matcherHolder<T>::setMatcher(matchers m_id){
    this->my_matcher=matcherFactory<T>(m_id);
  }
  
}

#endif
