#ifndef GPC_H_INCLUDED
#define GPC_H_INCLUDED
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include "GraphSet.h"
#include<random>
 
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

template<class T>
class gpc{

private:
    /**
    *GraphSet on which the algorithm for computing the Principal Components is applied
    */
    GraphSet<T> gs;
    /**
    *Barycenter of the matrix on which the Principal Component is computed
    */
    Eigen::Matrix<T,Eigen::Dynamic,1> barycenter;
    /**
    *Graph representation of the barycenter
    */
    GraphPointer<T> barycenter_net;

public:
    /**
    *Constructor
    */
    gpc(const GraphSet<T> & _gs): gs(_gs){};
    /**
    *Getter for the graphset
    */
    GraphSet<T> get_gs() const;
    /**
    *Getter for the barycenter
    */
    Eigen::Matrix<T,Eigen::Dynamic,1> get_barycenter() const;
    /**
    *Getter for the barycenter_net
    */
    GraphPointer<T> get_barycenter_net() const;
    /**
    *Setter of the barycenter
    */
    void set_barycenter(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>);
    /**
    *This method estimate the PCA with respect to a given alignment of the graphset. It is only an estimation, it is not the optimal one. 
    */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> est_pc(int n_comp, bool scale);
    /**
    *This method aligns the graphset with respect to a given geodesic
    */
    void align_geo(const geodesic<T> & geo, bool scale, int s_min, int s_max);
    /**
    *This method compute the Geodesic Principal Components of the graphset with the Align All and Compute principle.
    *It takes in input the maximum number of iterations(max_iterations), the tollerance (tol), the number of principal components wanted to be estimated (n_comp), a flag to indicate if you want to scale the PCA (scale) and the begin and end positionof the geodesic (?)
    */
    std::tuple<Eigen::MatrixXd, std::vector<std::map<std::pair<int,int>,attr_type<double>>>, Eigen::VectorXd> gpc_aac(int max_iterations, double tol, int n_comp, bool scale, double s_min, double     s_max);
};


template<class T>
GraphSet<T> gpc<T>::get_gs() const{

   return gs;
}

template<class T>
Eigen::Matrix<T,Eigen::Dynamic,1> gpc<T>::get_barycenter() const{

    return barycenter;
}

template<class T>
GraphPointer<T> gpc<T>::get_barycenter_net() const{

   return barycenter_net;
}

template<>
void gpc<int>::set_barycenter(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Mat ) {
    
    barycenter.resize(Mat.rows(),1);
    barycenter=((Mat.cast<double>()).colwise().mean()).cast<int>().transpose();
    
}

template<>
void gpc<float>::set_barycenter(Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Mat) {

    barycenter.resize(Mat.rows(),1);
    barycenter=(Mat.colwise().mean()).transpose();
    
}

template<>
void gpc<double>::set_barycenter(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> Mat) {

    barycenter.resize(Mat.rows(),1);
    barycenter=(Mat.colwise().mean()).transpose();
    
}

template<class T>
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> gpc<T>::est_pc(int n_comp, bool scale){

   int N=gs.get_graphset().size();
   
   GraphSet<T> permuted_gset=gs.permuted_graphset();
    
   Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Mat=permuted_gset.to_matrix_with_attr(false);
   
   
   if(scale==true)
      Mat=scale_matrix(Mat);
  
   
   set_barycenter(Mat);
   
   auto result=PCA(Mat,n_comp); 
   
   return result;
   
}


template<class T>
void gpc<T>::align_geo(const geodesic<T> & geo, bool scale, int s_min, int s_max){

   int v_a=gs.get_v_attr_max();
   int e_a=gs.get_e_attr_max();
   int N=gs.get_n_max();   
   
   GraphPointer<T> geo_net=give_me_a_network(geo,v_a,e_a,N,gs.is_oriented());
 
   GraphPointer<T> barycenter_net;
   
   if(scale==false){
      barycenter_net=give_me_a_network(barycenter,v_a,e_a,N,gs.is_oriented());
   }
    
   std::vector<double> dis;
   std::vector<std::vector<int>> f_i; 
   
   int count;
   #pragma omp parallel for private(count,dis, f_i)
   for(int i=0; i<gs.get_graphset().size(); ++i){
       dis.resize(10);
       f_i.resize(10);
 
      count=0;
      
      if(scale==true){
         for(int tilde=s_min; tilde<s_max; tilde+=abs(s_max-s_min)/10) { 
            auto mp=Matcher::matcherFactory<T>(gs.get_matcher(),gs.get_distance());
            mp->match(gs.get_graphset()[i], geo_net->scale(tilde));
            dis[count]=mp->get_dist();
            f_i[count]=mp->get_f();
            count++;
         }
      }   
      else{
         for(int tilde=s_min; tilde<s_max; tilde+=abs(s_max-s_min)/10) {   
             GraphPointer<T> G_tilde=add(1, barycenter_net, double(tilde), geo_net);
             auto mp=Matcher::matcherFactory<T>(gs.get_matcher(),gs.get_distance());
             mp->match(gs.get_graphset()[i],G_tilde);
             dis[count]=mp->get_dist();
             f_i[count]=mp->get_f();
             count++;
         }        
      }
      
     auto min_elem_it=std::min_element(dis.begin(), dis.end());
     gs.set_permutation_vector(i,f_i[min_elem_it-dis.begin()]);
     
   }

}

   
template<class T>
std::tuple<Eigen::MatrixXd, std::vector<std::map<std::pair<int,int>,attr_type<double>>>, Eigen::VectorXd> gpc<T>::gpc_aac(int max_iterations, double tol, int n_comp, bool scale, double s_min, double s_max){

   Eigen::MatrixXd final_scores;
   Eigen::VectorXd final_variance;
   std::vector<std::map<std::pair<int,int>,attr_type<double>>> final_map;
   final_map.resize(n_comp);
   std::vector<double> previous_steps=std::vector<double>(5,0.);
   
   /*
   std::default_random_engine gen;
    std::uniform_int_distribution<int> distr{0, int(gs.get_graphset().size()-1)};
    
   int first_id=distr(gen);*/
   
   GraphPointer<T> m_1=gs.get_graphset()[0];
   
   //Align the graphset wrt the first observation before compute the first estimate of the gpc
   gs.align(m_1);
   
   
   //Estimate the gpc
   std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> E_1=est_pc(n_comp,scale);
   std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd> E_2;
   
   for(int k=1; k<=max_iterations; ++k){
      align_geo(std::get<1>(E_1).row(0).cast<T>(), scale, s_min, s_max);
      std::cout << "Start Iteration " << k << std::endl;
      
      //Estimate the gpc
      E_2=est_pc(n_comp,scale);
      
      //Distance comparison
      double step_range=(std::get<2>(E_2)-std::get<2>(E_1)).norm();
      std::cout << step_range << std::endl;
      
      
      if(step_range<tol || previous_steps==std::vector<double>(5,step_range)){

         //Prepare the final values to be stored
         final_variance=std::get<2>(E_2);
         final_scores=std::get<0>(E_2);
         if(n_comp==1){
            Eigen::Matrix<double,Eigen::Dynamic,1> vec_row=std::get<1>(E_1).row(0);
            GraphPointer<double> gvecs=give_me_a_network(vec_row, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
            final_map[0]=gvecs->get_graph_map();
            barycenter_net=give_me_a_network(barycenter, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
         }
        else{
            for( int i=0; i<n_comp; i++){
               Eigen::Matrix<double,Eigen::Dynamic,1> vec_row=std::get<1>(E_2).row(i);
               GraphPointer<double> gvecs=give_me_a_network(vec_row, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
               final_map[i]=gvecs->get_graph_map();
            }
            barycenter_net=give_me_a_network(barycenter, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
         }
         std::cout << "STEP RANGE SMALLER THAN " << tol << std::endl;
         
         auto result=std::make_tuple(final_scores, final_map, final_variance);
         return result;
      }
      else
         E_1=E_2;
         
         for(int i=0; i<4; ++i){
               previous_steps[i]=previous_steps[i+1];
               previous_steps[4]=step_range;
         }   
   }
   
   std::cout << "Maximum Number of Iteration Reached" << std::endl;
   
   //Store the final value
   final_variance=std::get<2>(E_2);
   final_scores=std::get<0>(E_2);
   if(n_comp==1){
      Eigen::Matrix<double,Eigen::Dynamic,1> vec_row=std::get<1>(E_2).row(0);
      GraphPointer<double> gvecs=give_me_a_network(vec_row, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
      final_map[0]=gvecs->get_graph_map();
      barycenter_net=give_me_a_network(barycenter, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
   }
   else{
      for(int i=0; i<n_comp; i++){
         Eigen::Matrix<double,Eigen::Dynamic,1> vec_row=std::get<1>(E_2).row(i);
         GraphPointer<double> gvecs=give_me_a_network(vec_row, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
         final_map[i]=gvecs->get_graph_map();
      }
     barycenter_net=give_me_a_network(barycenter, gs.get_v_attr_max(), gs.get_e_attr_max(), gs.get_n_max(), gs.is_oriented());
   } 
   return std::make_tuple(final_scores, final_map, final_variance);
      
}
#endif
