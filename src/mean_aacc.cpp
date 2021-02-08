#include "GraphSet.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

namespace py = pybind11;

template<class T>
class Graph;

template<class T>
using GraphPointer= std::shared_ptr<Graph<T>>;

template<typename T>
using attr_type=std::vector<T>;


template<typename T>
py::list compute(std::vector<std::map<std::pair<int,int>,attr_type<T>>> graphset_maps, bool oriented, int max_iterations, double tol, std::string matcher_name){

   py::list result;
   
   Matcher::matchers my_match;
   
   if(!matcher_name.compare("Graduate assignment"))
      my_match=Matcher::matchers::GA;
   if(!matcher_name.compare("Identity"))
      my_match=Matcher::matchers::ID;
      
  
   GraphSet<T> gs(oriented);
   
      
   for(auto elem : graphset_maps){
       Graph<T> graph_to_add=Graph<T>(elem,oriented);
       gs.add_graph(std::make_shared<Graph<T>>(graph_to_add)); 
   }
   
   gs.set_match(my_match);
   
   gs.mean_aac(max_iterations,tol);

   result.append(gs.get_mean()->get_graph_map());
   result.append(gs.get_permutation_vector());
   result.append(gs.get_aligned_GraphSet_maps());
   
   return result;
    
   
}


PYBIND11_MODULE(meanc, m){

 m.def("compute", &compute<double>);
 m.def("compute", &compute<float>);
 m.def("compute", &compute<int>);
  
 
 #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
