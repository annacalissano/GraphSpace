#include "gpc.h"
#include <vector>
#include <map>


int main(){
     
    GraphSet<double> gs(false);
    
    gs.read_from_text("Pentagones_10_100_500_Perm.txt");

    gs.mean_aac(10,0.01);
    
    gs.get_mean()->print_map();
   
    return 0;
 }
