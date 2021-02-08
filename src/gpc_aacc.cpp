#include "gpc.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/embed.h>
#include <pybind11/pytypes.h>
#include <typeinfo>
#include <unordered_map>
#include <map>
#include <list>
#include <vector>
#include <string>
#include<tuple>

namespace py = pybind11;

template<class T>
class Graph;

template<class T>
using GraphPointer= std::shared_ptr<Graph<T>>;

template<typename T>
using attr_type=std::vector<T>;

template<class T>
using geodesic=Eigen::Matrix<T,Eigen::Dynamic,1>;

template<class T>
using MatcherPointer = std::unique_ptr<matcher<T>>;

template<typename T>
py::list compute_gpc(std::vector<std::map<std::pair<int,int>,attr_type<T>>> graphset_maps, std::string orientation, int max_iterations, double tol, int n_comp,bool scale, double s_min, double s_max, std::string matcher_name){

   
   bool oriented=false;
   
   if(orientation=="directed")
      oriented=true;
      
   py::list result;
   
   Matcher::matchers my_match;
   
   if(!matcher_name.compare("Graduate assignment"))
      my_match=Matcher::matchers::GA;
   if(!matcher_name.compare("Identity"))
      my_match=Matcher::matchers::ID;
      
   GraphSet<T> gs=GraphSet<T>(graphset_maps,oriented, my_match, Distance::distances::euclidean);
   
   gpc<T> _gpc=gpc<T>(gs);
   
   
   auto result_gpc=_gpc.gpc_aac(max_iterations, tol, n_comp, scale, s_min, s_max);

   result.append(std::get<0>(result_gpc));
   result.append(std::get<1>(result_gpc));
   result.append(std::get<2>(result_gpc));
   result.append(_gpc.get_barycenter_net()->get_graph_map());
   result.append(_gpc.get_barycenter());
   result.append(_gpc.get_gs().get_permutation_vector());
   
   return result;
    
}

PYBIND11_MODULE(gpcc, m){

  m.def("compute_gpc", &compute_gpc<double>);
  m.def("compute_gpc", &compute_gpc<float>);
  m.def("compute_gpc", &compute_gpc<int>);
 
 
 #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
