#ifndef MUNKRES_H_INCLUDED
#define MUNKRES_H_INCLUDED
#include<vector>
#include<algorithm>
#include<iostream>

template<class T>
using Matrix= std::vector<std::vector<T>>;

template<class T>
class Munkres{

private:

   Matrix<T> C;
   std::vector<bool> row_covered;
   std::vector<bool> col_covered;
   int n=0;
   int Z0_r=0;
   int Z0_c=0;
   std::vector<std::vector<int>> marked;
   std::vector<std::vector<int>> path;

public:
   Matrix<T> pad_matrix(Matrix<T> & Matrix, T pad_value);

   std::vector<std::pair<int,int>> compute(Matrix<T> cost_matrix);

   Matrix<int> _make_matrix(int n, T value);

   void _clear_covers();
   
   std::pair<int,int> _find_a_zero(int i0, int j0);
   
   T _find_smallest();
   
   int _find_star_in_row(int row);
   
   int _find_star_in_col(int col);
   
   int _find_prime_in_row(int row);
   
   void _convert_path(std::vector<std::vector<int>> _path, int count);
   
   void _erase_primes();
   
   int step1();
   int step2();
   int step3();
   int step4();
   int step5();
   int step6();

     
};


template<class T>
Matrix<T> Munkres<T>::pad_matrix(Matrix<T> & Matrix, T pad_value){

   int max_columns=0;
   int total_rows=Matrix.size();

   for(int i=0; i<total_rows; ++i){
      int dim_col=Matrix[i].size();
      max_columns=std::max(max_columns, dim_col);
   }
   total_rows=std::max(max_columns, total_rows);

   std::vector<std::vector<T>>  result;

   for(int i=0; i<Matrix.size(); ++i){
      int row_len=Matrix[i].size();
      result.push_back(Matrix[i]);
   
      if(total_rows>row_len)
         result[i].push_back((total_rows-row_len)*pad_value);    
   }
   
   while(result.size() < total_rows)
       result.push_back(std::vector<T>(total_rows,pad_value));

   return result;
}

template<class T>
std::vector<std::pair<int,int>> Munkres<T>::compute(Matrix<T> cost_matrix){
    
    C=pad_matrix(cost_matrix,0.);
    C=cost_matrix;
    n=C.size();
   
    int original_length=cost_matrix.size();
    int original_width=cost_matrix[0].size();
    row_covered=std::vector<bool>(n,false);
    col_covered=std::vector<bool>(n,false);
    Z0_r=0;
    Z0_c=0;
    path=_make_matrix(2*n,0);
    marked=_make_matrix(n,0);
    
    int step=1;
   
   
     while(step!=7) {
         switch(step) {
         case 1:
            step = step1(); // => [2]
            break;
         case 2:
            step = step2(); // => [0, 3]
            break;
         case 3:
            step = step3(); // => [3, 4, 5]
            break;
         case 4:
            step = step4(); // => [2]
            break;
         case 5:
            step = step5();
            break; // => [3]
         case 6:
            step = step6();
            break;
         }
      }
    
     std::vector<std::pair<int,int>> result;
     for(int i=0; i<original_length; ++i){
        for(int j=0; j<original_width; ++j){
           if(marked[i][j]==1)
              result.push_back(std::make_pair(i,j));
        }
     }
     
     return result;
}

template<class T>
Matrix<int> Munkres<T>::_make_matrix(int n, T value){

    Matrix<int> result;
    result.resize(n);
    
    for(int i=0; i<n; ++i)
       for( int j=0; j<n; j++)
          result[i].push_back(value);
          
    return result;
            
}

template<class T>
void Munkres<T>::_clear_covers(){

    for(int i=0; i<n; ++i){
       row_covered[i]=false;
       col_covered[i]=false;    
    }
}

template <class T>
std::pair<int,int> Munkres<T>::_find_a_zero(int i0, int j0){
    int row=-1;
    int col=-1;
    
    int i=i0;
    int j;
    bool done=false;
    
    while(!done){
       j=j0;
       while(true){
          if(C[i][j]==0 && !row_covered[i] && !col_covered[j]){
             row=i;
             col=j;
             done=true;
          }
          j= (j+1) % n;
          if (j==j0)
             break;
       }
       i=(i+1)% n;
       if (i==i0)
          done=true;
    }
     
     return std::make_pair(row,col);
}

template<class T>
T Munkres<T>::_find_smallest(){
   T minval=1.79769e+308; //very big number
   for(int i=0; i<n; ++i){
      for(int j=0; j<n; ++j){
         if(!row_covered[i] && !col_covered[j]){
            if(minval>C[i][j])
               minval=C[i][j];
         }
      }
   }
   
   return minval;

}

template<class T>
int Munkres<T>::_find_star_in_row(int row){

   int col=-1;
   for(int j=0; j<n;++j){
      if(marked[row][j] ==1){
         col=j;
         break;
      }
   }
   
   return col;

}

template<class T>
int Munkres<T>::_find_star_in_col(int col){

   int row=-1;
   for(int i=0; i<n;++i){
      if(marked[i][col] ==1){
         row=i;
         break;
      }
   }
   
   return row;

}

template<class T>
int Munkres<T>::_find_prime_in_row(int row){

   int col=-1;
   for(int j=0; j<n; ++j){
      if(marked[row][j]==2){
         col=j;
         break;
      }
   }

   return col;
}


template<class T>
void Munkres<T>::_convert_path(std::vector<std::vector<int>> _path, int count){

   for(int i=0; i<count+1; ++i){
      if(marked[_path[i][0]][_path[i][1]]==1)
         marked[_path[i][0]][_path[i][1]]=0;
      else
         marked[_path[i][0]][_path[i][1]]=1;
   }

}

template<class T>
void Munkres<T>::_erase_primes(){

   for(int i=0; i<n; ++i){
      for(int j=0; j<n; ++j)
         if(marked[i][j]==2)
            marked[i][j]=0;
   }
}

template<class T>
int Munkres<T>::step1(){
    
    for(int i=0; i<n; ++i){
       int pos=std::min_element(C[i].begin(), C[i].end())-C[i].begin();
       T min_val=C[i][pos];
       for( int j=0; j<n; ++j)
           C[i][j]-= min_val;
    }
      
    return 2;
}

template<class T>
int Munkres<T>::step2(){
  
   for(int i=0; i<n; ++i){
      for(int j=0; j<n; ++j){
         if(C[i][j]==0 && !(col_covered[j]) && !(row_covered[i])){
         marked[i][j]=1;
         col_covered[j]=true;
         row_covered[i]=true;
         break;
         
         }
      
      }
   
   }
    
   _clear_covers();
 
   return 3;
}

template<class T>
int Munkres<T>::step3(){

   int count=0;
   
   for(int i=0; i<n; ++i){
      for(int j=0; j<n; ++j){
         if(marked[i][j] == 1 && !col_covered[j]){
            col_covered[j]=true;
            count+=1;
         }   
      }
   }
  
   if(count >= n){
      return 7;
      
   }
   else{
      return 4;  
   }
}

template<class T>
int Munkres<T>::step4(){
  
   int step=7;
   bool done=false;
   int row=0; 
   int col=0;
   int star_col=-1;
   while(!done){
      auto pair=_find_a_zero(row,col);
      row=pair.first;
      col=pair.second;
      if(row<0){
         done=true;
         step=6;
      }
      else{
         marked[row][col]=2; 
         int star_col=_find_star_in_row(row);
         if(star_col>=0){
            col=star_col;
            row_covered[row]=true;
            col_covered[col]=false;
         }
         else{
            done=true;
            Z0_r=row;
            Z0_c=col;
            step=5;
         }
      }
   }

   return step;
}

template<class T>
int Munkres<T>::step5(){

   
   int count=0;
   path[count][0]=Z0_r;
   path[count][1]=Z0_c;
   bool done=false;
   
   while(!done){
      int row=_find_star_in_col(path[count][1]);
      if(row>=0){
         count+=1;
         path[count][0]=row;
         path[count][1]=path[count-1][1];
      }
      else
         done=true;
         
      if(!done){
         int col=_find_prime_in_row(path[count][0]);
         count+=1;
         path[count][0]=path[count-1][0];
         path[count][1]=col;
      }
   }
   _convert_path(path,count);
   _clear_covers();
   _erase_primes();

   return 3;
}

template<class T>
int Munkres<T>::step6(){
  
   T minval=_find_smallest();
   int events=0;
   for(int i=0; i<n; ++i){
      for(int j=0; j<n;++j){
         if(row_covered[i]){
            C[i][j]+=minval;
            events+=1;
         }
         if(!col_covered[j]){
            C[i][j]-=minval;
            events+=1;
         }
         if(row_covered[i] && !col_covered[j])
            events-=2;
      }
   }
   
   if(events==0)
       std::cout << "Matrix cannot be solved";
       
   return 4;
}

#endif
