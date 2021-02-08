#ifndef GRAPHSET_H_INCLUDED
#define GRAPHSET_H_INCLUDED
#include <random>
#include <omp.h>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include "Graph.h"
#include "MatcherFactory.h"
template<class T>
class Graph;

template<class T>
using GraphPointer= std::shared_ptr<Graph<T>>;

template<class T>
using attr_type=std::vector<T>;

template<class T>
using geodesic=Eigen::Matrix<T,Eigen::Dynamic,1>;

template<class T>
using MatcherPointer = std::unique_ptr<matcher<T>>;

/**
*Class GraphSet defines a set of graph.
*template T defines the type of data that is contained in the attributes
*A set of graph is defined as a vector of pointers to graph objects
*/

template<class T>
class GraphSet{
private:
    /**
    *Vector of original graphs
    */
    std::vector<GraphPointer<T>> graphset;  
    
    /**
    *Orientation of the graphs in the graphset
    */
    bool oriented;
    
    /**
    *Vector of Aligned graphs
    */
    std::vector<GraphPointer<T>> aligned_GraphSet;
    
    /**
    *Vector of permutation 
    */
    std::vector<std::vector<int>> permutation_vector;
    
    /**
    *Mean of the graphset
    */
    GraphPointer<T> mean;
    
    /**
    *Matcher associate to the graphset
    */
    Matcher::matchers my_matcher;
    
    /**
    *Distance that we want to use for performing matching operations
    */   
    Distance::distances dis;
    
    
public:
    GraphSet(bool _oriented);
    
    /**
    *Constructor
    */
    GraphSet(const std::vector<std::map<std::pair<int,int>,attr_type<T>>> & graph_maps, const bool orientation);
    
    /**
    *Constructor
    */
    GraphSet(const std::vector<std::map<std::pair<int,int>,attr_type<T>>> & graph_maps, const bool orientation, const Matcher::matchers _m, const Distance::distances _d);
    
    /**
    *The methods return a bool that indicates if the graphs in the graphset are orinted or not
    */
    bool is_oriented() const;
    
    /**
    *The method adds a graph to the GraphSet. A graph can be added to the graphset only if it has the same orientation as the graphset. 
    */
    void add_graph(GraphPointer<T> graph);
    
    /**
    *The method returns the vector of graph pointers
    */
    std::vector<GraphPointer<T>> get_graphset() const;
    
    /**
    *The method returns a vector of maps, related to the graphs contained in the GraphSet
    */
    std::vector <std::map<std::pair<int,int>,attr_type<T>>> get_graphset_maps() const;
    
    /**  
    *The method returns the vector of aligned graphs
    */
    std::vector<GraphPointer<T>> get_aligned_GraphSet() const;
    
    /**
    *The method returns a vector of maps, related to the aligned graphs
    */
    std::vector<std::map<std::pair<int,int>,attr_type<T>>> get_aligned_GraphSet_maps() const;
    
    /**
    *The method returns the vector of permutation
    */
    std::vector<std::vector<int>> get_permutation_vector() const;
    
    /**
    *The method allows to modify one component of the permutation vector with the permutation given as input
    */
    void set_permutation_vector(int index, std::vector<int> p);
    
    /**
    *The method returns the mean of the GraphSet
    */
    GraphPointer<T> get_mean() const;
    
    /**
    *The method returns the enum correspondent to the matcher used in the Graphset
    */
    Matcher::matchers get_matcher() const;
    
    /**
    *The methods set the enum of the matcher
    */
    void set_match(const Matcher::matchers _m);
    
    /**
    *The method returns the enum correspondent to the distance used in the Graphset
    */
    Distance::distances get_distance() const;
    
    /**
    *The method sets the enum of the distance
    */
    void set_distance(const Distance::distances _d);
    
    /**
    *The method returns the maximum number of nodes that a graph in the GraphSet has.
    */
    int get_n_max() const;
    
    /**
    *The methods returns the maximum dimension of vertex attributes among all graphs
    */
    int get_v_attr_max() const;
    
    /**
    *The methods returns the maximum dimension of edge attributes among all graphs
    */
    int get_e_attr_max() const;
    
    /**
    *The methods saves in the permutation_vector the permutations that have to be done to align the GraphSet with the specified input 
    */
    void align(GraphPointer<T> g);
    
    /**
    *The method returns the permuted graphset correspondent to the permutation vector 
    */
    GraphSet<T> permuted_graphset() const;
    
    /**
    *The method permutes the graphset and saves the permuted graphs obtained in the aligned_graphset 
    */
    void save_aligned();
    
    /**
    *The methods return a matrix associated to the graphset
    *Every row corresponds to a graph
    *Every columns correspond to a connection of the graph
    */
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> to_matrix_with_attr(bool aligned) const;
   
    /**
    *The methods return an estimate, but not the final value, of the mean of the GraphSet
    */
    GraphPointer<T> est(const GraphPointer<T> & m1) const;
    
    /**
    *The methods returns the Fréchet Mean of the GraphSet
    */
    void mean_aac(int max_iteration, double tol);
    
    /**
    *This methods reads a graph from a corrected formatted text file
    */
    void read_from_text(std::string file_name);   

};

template<class T>
GraphSet<T>::GraphSet(bool _oriented){
   
   oriented=_oriented; 
   set_match(Matcher::matchers::GA);
   set_distance(Distance::distances::euclidean);
}

template<class T>
GraphSet<T>::GraphSet(const std::vector<std::map<std::pair<int,int>,attr_type<T>>> & graph_maps, const bool orientation){

    oriented=orientation;
    
    for(auto elem: graph_maps){
        GraphPointer<T> graph_to_add=std::make_shared<Graph<T>>(Graph<T>(elem,orientation));
        graphset.push_back(graph_to_add);
    }
    set_distance(Distance::distances::euclidean);
    set_match(Matcher::matchers::GA);
}

template<class T>
GraphSet<T>::GraphSet(const std::vector<std::map<std::pair<int,int>,attr_type<T>>> & graph_maps, const bool orientation, const Matcher::matchers _m, const Distance::distances _d){

    oriented=orientation;
    
    for(auto elem: graph_maps){
        GraphPointer<T> graph_to_add=std::make_shared<Graph<T>>(Graph<T>(elem,orientation));
        graphset.push_back(graph_to_add);
    }
    
    set_match(_m);
    set_distance(_d);
    
}
template<class T>
bool GraphSet<T>::is_oriented() const{
 
   return oriented;
}


template<class T>
void GraphSet<T>::add_graph(GraphPointer<T> graph)
{
    if(graph->is_oriented()==oriented){
       graphset.push_back(graph);
       permutation_vector.resize(graphset.size());
       aligned_GraphSet.resize(graphset.size());
     }
     else{
        std::cerr << "You cannot add graph that has different orientation wrt the graphset";
        exit(1);  
     }
}


template<class T>
std::vector<GraphPointer<T>> GraphSet<T>::get_graphset() const
{
    return graphset;
}


template<class T>
std::vector <std::map<std::pair<int,int>,attr_type<T>>> GraphSet<T>::get_graphset_maps() const
{
   std::vector<std::map<std::pair<int,int>,attr_type<T>>> result;
   
   for (auto elem : graphset)
      result.push_back(elem->get_graph_map());
      
   return result;

}


template<class T>
std::vector<GraphPointer<T>> GraphSet<T>::get_aligned_GraphSet() const
{
    return aligned_GraphSet;
}


template<class T>
std::vector<std::map<std::pair<int,int>,attr_type<T>>> GraphSet<T>::get_aligned_GraphSet_maps() const
{
   std::vector<std::map<std::pair<int,int>,attr_type<T>>> result;
   
   for (auto elem : aligned_GraphSet)
      result.push_back(elem->get_graph_map());
      
   return result;

}


template<class T>
std::vector<std::vector<int>> GraphSet<T>::get_permutation_vector() const
{
   return permutation_vector;
}


template<class T>
void GraphSet<T>::set_permutation_vector(int index, std::vector<int> p){

    permutation_vector[index]=p;
}


template<class T>
GraphPointer<T> GraphSet<T>::get_mean() const
{
     return mean;
}

template<class T>
Matcher::matchers GraphSet<T>::get_matcher() const{

    return my_matcher;
}

template<class T>
void GraphSet<T>::set_match(const Matcher::matchers _m){

    my_matcher=_m;
}

template<class T>
Distance::distances GraphSet<T>::get_distance() const{

    return dis;
}

template<class T>
void GraphSet<T>::set_distance(const Distance::distances _d){

    dis=_d;
}

template<class T>
int GraphSet<T>::get_n_max() const{

    std::vector<int> n_nodes;
    
    for (auto graph : graphset)
        n_nodes.push_back(graph->get_n_nodes());
    
    return *std::max_element(n_nodes.begin(),n_nodes.end());

}


template<class T>
int GraphSet<T>::get_v_attr_max() const{

    std::vector<int> v_attr_sizes;
    
    for(auto graph : graphset)
       v_attr_sizes.push_back(graph->get_vertex_size());
    
    return *std::max_element(v_attr_sizes.begin(),v_attr_sizes.end());
}
    

template<class T>    
int GraphSet<T>::get_e_attr_max() const{

    std::vector<int> e_attr_sizes;
    
    for(auto graph : graphset)
       e_attr_sizes.push_back(graph->get_edge_size());
    
    return *std::max_element(e_attr_sizes.begin(),e_attr_sizes.end());
}


template<class T>
void GraphSet<T>::align(GraphPointer<T> g){
   
   int N=graphset.size();
   permutation_vector.resize(N);

   for(auto elem:graphset)
     elem->grow(get_n_max(),std::vector<T>(get_v_attr_max(),0.));
         
  #pragma omp parallel for  
       for(int i=0; i<N; i++){
           auto mp=Matcher::matcherFactory<T>(my_matcher,dis);
           mp->match(graphset[i],g);
           permutation_vector[i]=mp->get_f();
           
       } 
      
      
}


template<class T>
GraphSet<T> GraphSet<T>::permuted_graphset() const{

    GraphSet<T> result=GraphSet<T>(oriented);
    
    int N=graphset.size();
    
    for(int i=0; i<N; i++)
       result.add_graph(graphset[i]->permute(permutation_vector[i]));
               
    return result;

}

template<class T>
void GraphSet<T>::save_aligned(){

   int N=graphset.size();
   int n_nodes=get_n_max();
   aligned_GraphSet.resize(N);
   
   #pragma omp parallel for
       for(int i=0; i<N; i++){
          GraphPointer<T> aligned_graph=graphset[i]->permute(permutation_vector[i]);
          aligned_GraphSet[i]=aligned_graph; 
       }
}

template<class T>
GraphPointer<T> GraphSet<T>::est(const GraphPointer<T> & m1) const{

    auto mC=m1;
   
    for (double i=0; i<graphset.size(); ++i)
        mC=add(1.0/(i+1.0),aligned_GraphSet[i],i/(i+1.0),mC);
        
    return mC;
}

template<class T>
void GraphSet<T>::mean_aac(int max_iteration, double tol){
         
    int first_id=0; 
     
    GraphPointer<T> m_1=graphset[first_id];
    GraphPointer<T> m_2;
    
    auto mp=Matcher::matcherFactory<T>(my_matcher,dis);
    
    std::vector<double> previous_steps=std::vector<double>(5,0.);
        
    for(int k=1; k<=max_iteration; ++k){
    
       
       
       std::cout << "Start iteration " << k << std::endl << std::endl;
        
       //Graphset alignment 
       align(m_1);
       save_aligned();
         
       //Computation of the mean   
       GraphPointer<T> m_2=est(m_1);

       //Mean comparison
       mp->match(m_1,m_2);
       double step_range=mp->get_dist();
        
              
       std::cout << step_range << std::endl;
     
        
        if(step_range<tol || previous_steps==std::vector<double>(5,step_range)){
          this->mean=m_2;
           return;
        }
        else{
            m_1=m_2;
            
            //Steps update
            for(int i=0; i<4; ++i){
               previous_steps[i]=previous_steps[i+1];
               previous_steps[4]=step_range;
            }        
        }   
        
    }
    
    std::cout << "Max number of iteration reached"<< std::endl;
    
    this->mean=m_1;
 
}

template<class T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> GraphSet<T>::to_matrix_with_attr(bool aligned) const{

   int N=get_n_max();
   int v_a=get_v_attr_max();
   int e_a=get_e_attr_max();
   
   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
   
   mat.resize(graphset.size(),N*v_a+N*(N-1)*e_a);
  
   int count;
   
   
   for(int g=0; g<graphset.size(); ++g){
       count=0;
       std::map<std::pair<int,int>, attr_type<T>> g_map;
       if(aligned==false)
          g_map=graphset[g]->get_graph_map();
       else
          g_map=aligned_GraphSet[g]->get_graph_map(); 
                  
       for(int i=0; i<N; ++i){
           for(int j=0; j<N; ++j){
               if(i==j){
                   for(int k=0; k<v_a; ++k){
                       if(g_map.find(std::make_pair(i,j))!=g_map.end()){
                          mat(g,count)=g_map[std::make_pair(i,j)][k];}
                       else
                          mat(g,count)=0;
                       count++;
                   }
               }
               else{
                   for(int k=0; k<e_a; ++k){
                       if(g_map.find(std::make_pair(i,j))!=g_map.end())
                          mat(g,count)=g_map[std::make_pair(i,j)][k];
                       else
                          mat(g,count)=0;
                       count++;
                   }
               }    
           } 
       }
   }
   
   return mat;  
}

template<class T>
void GraphSet<T>::read_from_text(std::string file_name){

   std::ifstream file;
   file.open(file_name);

   if(!file){
      std::cerr << "Unable to open " << file_name;
      exit(1);
   }
   
   std::string line;
   
   std::map<std::pair<int,int>,attr_type<T>> graph_map;
   bool graph_data=false;
   
   while(std::getline(file,line)){
        
      std::istringstream record(line);
      std::string first;
      std::getline(record,first,' ');
     
     
     //Parameters for set up reading
      if( first.compare("GraphSet")==0){
         std::cout << "Start Parsing" << std::endl;
         continue;
      }
      
      if( first.compare("GRAPH_TYPE")==0){
         std::string second;
         std::getline(record,second,'\n');
         second.erase(std::prev(second.end()));
         if(second.compare("directed")==0)
            oriented=true;
         else
            oriented=false;
      }
      
      if( first.compare("LABELS")==0){
         continue;
      }
      
      if( first.compare("FEATURES")==0){
         continue;
      }
      
      if(first.compare("NODE_ATTR")==0){
         continue;
      }
      
      if(first.compare("EDGE_ATTR")==0){
         continue;
      }
      
      if(first.compare("NODE_ATTR")==0){
         continue;
      }
      
      if(first.compare("Graph")==0){
         continue;
      }
      
      if(first.compare("Attributes")==0){
         graph_data=true;
         continue;
      }
      
      if(first.compare("Adjency")==0){
          GraphPointer<T> graph_to_add=std::make_shared<Graph<T>>(Graph<T>(graph_map,oriented));
          add_graph(graph_to_add);
          graph_map.clear();
          graph_data=false;
          continue;
      }
      
   
      //Graphset reading
      if(graph_data){
          std::string second_node;
          std::getline(record,second_node,' ');   
      
          attr_type<T> attribute;
          std::string current_attribute;
          
          while(getline(record,current_attribute,' '))
             attribute.push_back(std::stod(current_attribute));          
          graph_map.insert(std::make_pair(std::make_pair(std::stoi(first),std::stoi(second_node)),attribute));
      
      }   
   }
   
   std::cout << "End Parsing" << std::endl;

}

#endif 
