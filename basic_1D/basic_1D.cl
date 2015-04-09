#define l_bg 7
#define l_ed 30
#define l_stp 1
#define idx_ofst 9
#define idx_stp 1
//#define LENGTH 10
#define sharing_sz 1
//#include <cmath>

__kernel void basic(__global int *A, __global int *B)
{
    for(int i = l_bg; i < l_ed; i += l_stp)
               B[i] =  5 * A[idx_stp * i + idx_ofst];
}   

/*__kernel void basic_stencil(__global int *A, __global int *B)
{
	int g_xid = get_global_id(1);
	int g_yid = get_global_id(2);
	B[g_yid][g_xid] =  A[idx_stp_y * g_yid + idx_ofst_y1][idx_stp * g_xid + idx_ofst_x1] 
			 + A[idx_stp_y * g_yid + idx_ofst_y2][idx_stp * g_xid + idx_ofst_x2] 
			 + A[idx_stp_y * g_yid + idx_ofst_y3][idx_stp * g_xid + idx_ofst_x3];
}   */

__kernel void basic_sharing(__global int *A, __global int *B)
{
    //Tiling shared-memory transfer
    //shr_A(shr_ofst, shr_stp, shr_sz)
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    //1. Sharing initialization
    int shr_sz;
    int shr_ofst = l_bg * idx_stp + idx_ofst;
    int shr_stp = l_stp * idx_stp;
    if((l_ed - l_bg) % l_stp == 0)
        shr_sz = (l_ed - l_bg )/ l_stp;
    else
        shr_sz = (l_ed - l_bg )/ l_stp + 1; //Count the residue as "1"

    __local int shr_A[sharing_sz];

    if(lsz < shr_sz){
	//Distribute work load onto each work item
        int shr_sz_pwi;
    	int res;
	if(shr_sz % lsz == 0){
		shr_sz_pwi = shr_sz / lsz;
		res = shr_sz_pwi;
	}
	else{
		shr_sz_pwi = shr_sz / lsz + 1;
		res = shr_sz % shr_sz_pwi;
	}
        int wi_ofst = lid * shr_sz_pwi;
        if(lid < lsz - 1){
            for(int i = wi_ofst; i < wi_ofst + shr_sz_pwi; i++)
                shr_A[i] = A[shr_ofst + i * shr_stp];
        }else{
            for(int i = wi_ofst; i < wi_ofst + res; i++)
                  shr_A[i] = A[shr_ofst + i * shr_stp];
        }
     }else{
        if(lid < shr_sz)
            shr_A[lid] = A[shr_ofst + lid * shr_stp];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    //Kernel Modification
    for(int i = l_bg; i < l_ed; i += l_stp)
              B[i] = 5 * shr_A[(i - l_bg)/ l_stp];  
}
