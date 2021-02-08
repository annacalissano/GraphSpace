#include "gpc.h"
#include <vector>
#include <map>


int main(){
     
    py::scoped_interpreter guard{}; 
     
    GraphSet<double> gs(false);
    
    gs.read_from_text("Pentagones_10_100_500_Perm.txt");

    
    gpc<double> _gpc=gpc<double>(gs);  
    
    
    auto result=_gpc.gpc_aac(5, 0.001, 3, false, 0, 10);
    
    std::cout <<"Explained Variance: " << std::endl << std::get<2>(result);
    
    return 0;
 }
