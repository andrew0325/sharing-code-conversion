#include <parameter.h>

__kernel void stencil(__global int *A, __global int *B)
{
	int xid = get_global_id(0);
	for(int i = 0; i < 100000000; i++)
		B[xid] =  A[1 * xid + idx_ofst_x1] + A[1 * xid + idx_ofst_x2];
}

__kernel void stencil_sharing(__global int *A, __global int *B)
{
	int g_xid = get_global_id(0);
	int l_xid = get_local_id(0);
	int diff = (abs_diff(idx_ofst_x1, idx_ofst_x2));
	if(get_local_size(0) > diff){
		__local int shr_A[shr_sz];
		int bg = min(idx_ofst_x1, idx_ofst_x2) + get_group_id(0) * get_local_size(0);	
		int ed = max(idx_ofst_x1, idx_ofst_x2) + (get_group_id(0) + 1) * get_local_size(0);
                int shr_ofst = bg + get_local_size(0) * get_group_id(0);
		int wl_pwi;
		int res;
        	wl_pwi = (ed - bg) / get_local_size(0);
		res = (ed - bg) % get_local_size(0);
                res += wl_pwi;
		int wi_ofst = l_xid * wl_pwi;
		if(l_xid < get_local_size(0) - 1){
            	    for(int i = wi_ofst; i < wi_ofst + wl_pwi; i++)
		        shr_A[i] = A[shr_ofst + i];
		}else{
            	    for(int i = wi_ofst; i < wi_ofst + res; i++)
			shr_A[i] = A[shr_ofst + i];
		}
                barrier(CLK_LOCAL_MEM_FENCE);
		for(int i = 0; i < 100000000; i++)
			B[g_xid] = shr_A[l_xid] + shr_A[l_xid + diff];
	}else{
		int xid = get_global_id(0);
		for(int i = 0; i < 100000000; i++)
			B[g_xid] =  A[1 * g_xid + idx_ofst_x1] + A[1 * g_xid + idx_ofst_x2];
	}
}

