#ifndef GRAPH_H_INCLUDED
#define GRAPH_H_INCLUDED
#include<map>
#include<set>
#include<list>
#include<memory>
#include <iostream>

template<class T>
class Graph;

template<class T>
using GraphPointer= std::shared_ptr<Graph<T>>;

/**
*This alias is used to indicate the type of the attribute of a vertex or an edge. The T template indicates the type of the elements contained in the attribute
*/
template<class T>
using attr_type=std::vector<T>;


/**
*Class Graph is used to define a graph object
*/
template<class T>
class Graph{

private:
    /**
    *Map that describes the graph, keys defines vertex and edge while values are the corresponding attribute
    */
    std::map<std::pair<int,int>,attr_type<T>> graph_map;
    
    /**
    *Number of vertices of the graph
    */
    int n_nodes;
    
    /**
    *Orientation: true if the graph is oriented, false otherwise
    */
    bool oriented;
    
    /**
    *Adjacency list of the graph
    */
    std::vector<std::vector<int>> adj;                    
    
public:
     
    /**
    *Constructor 
    */ 
    Graph(bool _oriented):oriented(_oriented),n_nodes(0){};
    
    /**
    *Constructor 
    */
    Graph(const std::map<std::pair<int,int>,attr_type<T>> & _graph_map, const bool _oriented);                                                     
    
    /**
    *Constructor 
    */
    Graph(std::map<std::pair<std::pair<int,int>,std::pair<int,int>>,double> product_graph_constructor,bool oriented);    
    
    /**
    *The methods adds a vertex to the graph
    */
    void add_vertex(const attr_type<T> &attribute, const int id_vertex);                                                                           
    
    /**
    *The method adds an edge to the graph 
    */
    void add_edge(const int id_vertex1,const int id_vertex2, const attr_type<T> & edge_attribute);
                                                                                                               
    /**
    *The methods returns the orientation of the graph
    */
    bool is_oriented() const; 
               
    /**
    *The methods returns a bool that indicates if the graph is empty
    */
    bool isempty() const; 
    
    /**
    *The methods return the number of the vertices of the graph
    */
    int get_n_nodes() const; 
    
    /**
    *The methods returns a list of the graph vertices id
    */
    std::list<int> get_vertices_id() const;   
    
    /**
    * The methods returns the map that describes the graph
    */
    std::map<std::pair<int,int>,attr_type<T>> get_graph_map() const; 
    
    /**
    *The method returns the size of the vertex attribute
    */
    int get_vertex_size() const;            
    
    /**
    *The method returns the size of the edge attribute
    */
    int get_edge_size() const;          
                                                         
    
    /**
    * The method returns the adjacency list of the graph
    */
    std::vector<std::vector<int>> get_adj() const;
    
    /**
    *The method constructs the adjacency list of the graph and it stores the matrix in the attribute
    */
    void construct_adj();
    
    /**
    *The method returns the permuted graph given the permutation to apply 
    */
    GraphPointer<T> permute(const std::vector<int> & f) const;
    
    /**
    *The method prints the graph in the map form
    */
    void print_map() const;
    
    /**
    *The methods returns a set that contains all the keys of the graph map, namely all the nodes couple present in the graph
    */
    std::set<std::pair<int,int>> get_keys() const;
    
    /**
    *The method increases the size of the graph creating new vertex with a specified input, until the chosen size is reached.
    * The new vertex created are not linked with other vertex already existing
    */
    void grow(int size, const attr_type<T> & new_attribute);
    
    /**
    *The method return a graph that has got multiplied attribute by the input constant
    */
    GraphPointer<T> scale(double a) const;
    
};


template<class T>
Graph<T>::Graph(const std::map<std::pair<int,int>,attr_type<T>> & _graph_map, const bool _oriented)
{
    graph_map=_graph_map;
    oriented=_oriented;
    n_nodes=0;
    
    for (auto it=_graph_map.begin(); it!=_graph_map.end(); ++it)
    {
        if (it->first.first==it->first.second)
        {
            n_nodes+=1;
        }
    }
    
    this->construct_adj();
}


template<class T>
int Graph<T>::get_n_nodes() const
{
    return n_nodes;
}


template<class T>
bool Graph<T>::isempty() const
{
    return bool(graph_map.size()==0);
}


template<class T>
void Graph<T>::add_vertex(const attr_type<T> & attribute, const int id_vertex)
{
    graph_map.insert(std::make_pair(std::make_pair(id_vertex,id_vertex),attribute));
    n_nodes++;
    this->construct_adj();
}


template<class T>
void Graph<T>::add_edge(const int id_vertex1, const int id_vertex2, const attr_type<T> & edge_attribute)
{
    graph_map.insert(std::make_pair(std::make_pair(id_vertex1,id_vertex2),edge_attribute));
    this->construct_adj();
}


template<class T>
bool Graph<T>::is_oriented() const
{
    return oriented;
}


template<class T>
std::list<int> Graph<T>::get_vertices_id() const
{
    std::list<int> vertices_id_vector;
    
    for (auto it=graph_map.begin(); it!=graph_map.end(); ++it)
    {
        if (it->first.first==it->first.second)
        {
            vertices_id_vector.push_back(it->first.first);
        }
    }
    return vertices_id_vector;
}


template<class T>
int Graph<T>::get_vertex_size() const{

    bool find=false;
    int size;
    
    for(auto it=graph_map.begin(); it!=graph_map.end() && find==false; ++it)
        if(it->first.first==it->first.second){
            find=true;
            size=it->second.size();
        }
    
    return size;
}


template<class T>
int Graph<T>::get_edge_size() const{

    bool find=false;
    int size;
    
    for(auto it=graph_map.begin(); it!=graph_map.end() && find==false; ++it)
        if(it->first.first!=it->first.second){
            find=true;
            size=it->second.size();
        }
    
    return size;
}


template<class T>
std::map<std::pair<int,int>,attr_type<T>> Graph<T>::get_graph_map() const
{
    return graph_map;
}


template<class T>
std::vector<std::vector<int>> Graph<T>::get_adj() const{

  return adj;
  
}


template<class T>
void Graph<T>::construct_adj(){

    adj=std::vector<std::vector<int>>();

    for(auto v : this->get_vertices_id() ){
       std::vector<int> temp;
       for( auto elem : graph_map){
          if(elem.first.first==v & elem.first.second!=v)
             temp.push_back(elem.first.second);
       }
       adj.push_back(temp);    
    }

  
}


template<class T>
GraphPointer<T> Graph<T>::permute(const std::vector<int> & f) const{

    
    std::map<std::pair<int,int>,attr_type<T>> permuted_graph_map;
    
    for(int i=0; i<f.size(); i++) {
       permuted_graph_map[std::make_pair(f[i],f[i])]=this->graph_map.at(std::make_pair(i,i));
       for(int j=0; j<adj[i].size(); j++){
           permuted_graph_map[std::make_pair(f[i],f[adj[i][j]])]=this->graph_map.at(std::make_pair(i,adj[i][j]));       
       }
    }

    return std::make_shared<Graph<T>>(Graph<T>(permuted_graph_map,oriented));    
}


template<class T>
void Graph<T>::print_map() const{

    for( auto elem: graph_map) {
        std::cout << "[" << elem.first.first << "," << elem.first.second << "]=[";
        for (auto i : elem.second)
            std::cout << i << ",";
        std::cout << "]"<< std::endl; 
    }

}


template<class T>
std::set<std::pair<int,int>> Graph<T>::get_keys() const{

   std::set<std::pair<int,int>> result;
   
   for( auto elem : graph_map )
        result.insert(elem.first);
        
   return result;
}


template<class T>
void Graph<T>::grow(int size, const attr_type<T> & new_attribute){

    attr_type<T> empty_vector; 
   
    if(size<=n_nodes)
        return;
    else{
        for(auto i=0; i<size; i++){
           if(i>=n_nodes)
               this->add_vertex(new_attribute,i);
        }
        n_nodes=size;
    }  
}


template<class T>
GraphPointer<T> Graph<T>::scale(double a) const{
    
    GraphPointer<T> result=std::make_shared<Graph<T>>(oriented);
    
    for( auto elem : graph_map){
       attr_type<T> attribute;
       for(auto i=0; i<elem.second.size(); ++i)
          attribute.push_back(elem.second[i]*a);
          
      if(elem.first.first==elem.first.second)
         result->add_vertex(attribute, elem.first.first);
      else
         result->add_edge(elem.first.first, elem.first.second, attribute);      
    }      
    
    return result;
}
#endif // GRAPH_H_INCLUDED
