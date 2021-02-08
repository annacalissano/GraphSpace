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
#include "Graph.h"
#include "MatcherFactory.h"

namespace py = pybind11;

template<typename T>
using attr_type=std::vector<T>;

template<typename T>
py::list GAc (std::map<std::pair<int,int>,attr_type<T>> first_graph_map, std::map<std::pair<int,int>,attr_type<T>> second_graph_map, bool oriented){

   py::list final_result;
   GraphPointer<T> first_graph=std::make_shared<Graph<T>>(Graph<T>(first_graph_map,oriented));
   GraphPointer<T> second_graph=std::make_shared<Graph<T>>(Graph<T>(second_graph_map,oriented));
   auto m=Matcher::matcherFactory<T>(Matcher::matchers::GA, Distance::distances::euclidean);
   m->match(first_graph, second_graph);
   
   for(auto elem : m->get_f())
     final_result.append(elem);
      
    return final_result;
}

template<typename T>
py::list IDc (std::map<std::pair<int,int>,attr_type<T>> first_graph_map, std::map<std::pair<int,int>,attr_type<T>> second_graph_map, bool oriented){
   
   py::list final_result;
   GraphPointer<T> first_graph=std::make_shared<Graph<T>>(Graph<T>(first_graph_map,oriented));
   GraphPointer<T> second_graph=std::make_shared<Graph<T>>(Graph<T>(second_graph_map,oriented));
   auto m=Matcher::matcherFactory<T>(Matcher::matchers::ID, Distance::distances::euclidean);
   m->match(first_graph, second_graph);
   
   for(auto elem : m->get_f())
     final_result.append(elem);
      
    return final_result;
}

PYBIND11_MODULE(matcherc, m){

 m.def("GAc", &GAc<double>);
 m.def("GAc", &GAc<float>);
 m.def("GAc", &GAc<int>);
 m.def("IDc", &IDc<double>);
 m.def("IDc", &IDc<float>);
 m.def("IDc", &IDc<int>);
 
 #ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
