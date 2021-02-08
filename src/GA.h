#ifndef GA_H_INCLUDED
#define GA_H_INCLUDED
#include "Graph.h"
#include "matcher.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <map>
#include "distance.h"
#include "Munkres.h"
#include <memory>


/**
* Definition of an eigen matrix of double with dynamic size
*/
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

/**
* Class that implemets the Graduate assigned algorithm to match two graph 
*/
template<class T>
class GA: public matcher<T>{

private : 
    double opt_b0 = 0.5;
    double opt_bf = 10.0;
    double opt_br = 1.075;

    // max # of iterations of inner and outer loop
    int opt_I0 = 4;
    int opt_I1 = 30;

    // precision for terminantion in inner and outer loop
    double opt_eps0 = 0.5;
    double opt_eps1 = 0.05;
    
    std::map<std::pair<int,int>,MatrixXd>  A;
    MatrixXd a;
    MatrixXd M;
    double b;
    bool swapped = false;
    

public : 
/**
* Default constructor
*/
GA()=default;
/**
* Constructor if you want to set a specific distance 
*/
GA(Distance::distances _d):matcher<T>(_d){};
/**
* Method that takes as input two GraphPointer and compute the permutation vector according with the Graduate assigned algorithm. Vector that is stored in the attribute f of the class Matcher 
*/
void match(GraphPointer<T> first_graph,GraphPointer<T> second_graph) override;
/**
* accessory function used by the match method to controll that all is going well 
*/
bool isStable(GraphPointer<T> first_graph,GraphPointer<T> second_graph,MatrixXd M1,MatrixXd M2,double eps);
/**
* accessory function used by the match method to setup all the variables needed
*/
void initializeMatchMatrix(GraphPointer<T> x,GraphPointer<T> y);
/**
* accessory function used by the match method to compute the matching with the hungarian algorithm in the Munkres function  
*/
void cleanup(GraphPointer<T> first_graph,GraphPointer<T> second_graph);
/**
* accessory function used by the match method to setup all the variables needed
*/
void setAssociationGraph(GraphPointer<T> first_graph,GraphPointer<T> second_graph);
};

template<class T>
void GA<T>::match(GraphPointer<T> first_graph,GraphPointer<T> second_graph){

  GraphPointer<T> x = first_graph;
  GraphPointer<T> y = second_graph;
  
  // check the dimensions and set everything to start
  if(first_graph->get_n_nodes() > second_graph->get_n_nodes()){
    y = first_graph; 
    x = second_graph;
    swapped = true;
  }
  if(x->get_n_nodes() == 1 && y->get_n_nodes() == 1){
    // devo sistemare questa parte 

    matcher<T>::f.resize(0);
    matcher<T>::f.push_back(0);
  }
  
  GA<T>::setAssociationGraph(x,y); 
  
  int nX = x->get_n_nodes();
  int nY = y->get_n_nodes();
  
  GA<T>::initializeMatchMatrix(x,y);
  
  std::vector<std::vector<int>> adjX = x->get_adj();
  std::vector<std::vector<int>> adjY = y->get_adj();
  
  //New parameters
  // Partial derivative matrix taylor espansion 
  MatrixXd Q(nX,nY);
  // M0 exp equation 	
  MatrixXd  M0(nX+1,nY+1);	
  // M1 scaled M0
  MatrixXd M1(nX+1,nY+1);
  
  // A loop
  b = opt_b0;
  while( b < opt_bf){
  
    // B loop
    for ( int t0 = 0 ; t0 < opt_I0 ; ++t0){
      // copy
      for ( int i = 0; i <= nX; ++i){
        M0.row(i) = M.row(i); 
      }
      //softmax
      for (int i = 0; i < nX; ++i){
        for (int j = 0; j < nY; ++j){
          Q.coeffRef(i,j) = a(i,j);
          int degX = adjX[i].size();
          for (int  k = 0; k < degX; ++k){
            int degY = adjY[j].size();
            if(degY!=0){
              for(int l = 0; l < degY; ++l){
                Q.coeffRef(i,j) += A[std::make_pair(i,j)](k,l)*M0.coeffRef(adjX[i][k],adjY[j][l]);
              }
            }
          }
          M(i,j) = std::exp(b*Q.coeffRef(i,j));
        }
      }
      // C loop
      for(int t1 = 0; t1 < opt_I1; ++t1){
        for(int i = 0; i <= nX; ++i){
          //copy
          M1.row(i) = M.row(i);
        }
        // normalize across all rows
        for(int i = 0; i <= nX; ++i){
          double row_sum = 0;
          for(int j = 0; j <= nY; ++j){
            row_sum += M(i,j);
          }
          for(int j = 0; j <= nY; ++j){
            M(i,j)/=row_sum;
          }
        }
        // normalize across all columns
        for(int j = 0; j <= nY; ++j){
          double col_sum = 0;
          for(int i = 0; i <= nX; ++i){
            col_sum += M(i,j);
          }
          for(int i = 0; i <= nX; ++i){
            M(i,j)/=col_sum;
          }
        }
        // check for convergence
        if(isStable(x,y,M,M1,opt_eps1)){
          break;
        }
      }
      if(isStable(x,y,M,M0,opt_eps0)){
        break;
      }
    }
    b *=opt_br;
  }
  
  cleanup(x,y);
  matcher<T>::dist=matcher<T>::the_dis(x,y);
}

// check how the algprithm is going
template<class T>
bool GA<T>::isStable(GraphPointer<T> first_graph,GraphPointer<T> second_graph,MatrixXd M1, MatrixXd M2,double eps){

        int nX=first_graph->get_n_nodes();
        int nY=second_graph->get_n_nodes();
        double err=((M1-M2).cwiseAbs()).sum(); 
        err/=nX*nY;
        return (err<eps);
}

// set parameters function
template<class T>
void GA<T>::initializeMatchMatrix(GraphPointer<T> x,GraphPointer<T> y){
  int nX = x->get_n_nodes();
  int nY = y->get_n_nodes();
  // Matching matrix variable
  M = MatrixXd::Constant(nX+1,nY+1,1.001);
  return ;
}

template<class T>
void GA<T>::setAssociationGraph(GraphPointer<T> first_graph,GraphPointer<T> second_graph){
   auto x=first_graph->get_graph_map();
   auto y=second_graph->get_graph_map();
   
   std::vector<std::vector<int>> adjX = first_graph->get_adj();
   std::vector<std::vector<int>> adjY = second_graph->get_adj();
   
   double scale = 0; 
   
   int nX=first_graph->get_n_nodes();
   int nY=second_graph->get_n_nodes();
   
   a.resize(nX,nY); 
   
   for(int i = 0; i < nX; ++i){
     int degX = adjX[i].size();
     for(int j = 0; j < nY; ++j){
       int degY = adjY[j].size();
      
       //node distance
       a(i,j) = matcher<T>::distance->node_sim(x[std::make_pair(i,i)],y[std::make_pair(j,j)]);
       scale = std::max(std::abs(a(i,j)),scale);
       A[std::make_pair(i,j)].resize(degX,degY); 
       
       for(int k = 0; k < degX; ++k){
         int k0 = adjX[i][k];
         for(int l = 0; l < degY; ++l){
           int l0 = adjY[j][l];
           
           A[std::make_pair(i,j)](k,l) = matcher<T>::distance->edge_sim(x[std::make_pair(i,k0)],y[std::make_pair(j,l0)]);
           scale = std::max(std::abs(A[std::make_pair(i,j)](k,l)),scale);
         }
       }
     }
   }
   if(scale == 0){
     return;
   }
   for(int i = 0; i < nX; ++i){
     for(int j = 0; j < nY; ++j){
       a(i,j)/=scale;
       int degX = adjX[i].size();
       for(int k = 0; k < degX; ++k){
         int degY = adjY[j].size();
         for(int l = 0; l < degY; ++l){
          A[std::make_pair(i,j)](k,l) /= scale;
         }
       }
     }
   }
   
return; 
}



//VERSIONE CON MUNKRES IMPLEMENTATO

template<class T>
void GA<T>::cleanup(GraphPointer<T> first_graph,GraphPointer<T> second_graph){
  
  int nX=first_graph->get_n_nodes();
  int nY=second_graph->get_n_nodes();
  
  std::vector<std::vector<double> > C(nX);
  double value = 0; 
  for(int i = 0; i < nX; ++i){
    for(int j = 0; j < nY; ++j){
      value = 1.0 - M(i,j);
      C[i].push_back(value);
      }
  }

 Munkres<double> munk=Munkres<double>();
 auto indexmatch=munk.compute(C);
  
 matcher<T>::f.resize(indexmatch.size());
  
  
 for( int i=0; i<indexmatch.size(); ++i){
     matcher<T>::f[i]=(indexmatch[i].second);
  }
  
 matcher<T>::f.resize(nX); 
  
  if(swapped){
    std::vector<int> g(nY);
    for( int j = 0; j < nY; ++j){
      g[j] = -1;
    }
    for( int i = 0; i < nX; ++i){ 
      g[matcher<T>::f[i]] = i; 
    }
    int c = nX;
    for( int l = 0; l < nY; ++l){
      if(g[l] == -1){
      	g[l]= c;
      	c++;
      	}
    }
    matcher<T>::f = g; 
  }

}
#endif //GA_H_INCLUDED
