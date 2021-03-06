#define idx_ofst_x1 0
#define idx_ofst_x2 1
#define LENGTH (4096*1024)
#define local_sz 256
#define abs_diff(X, Y) (((X) > (Y))? ((X) -(Y)):((Y) - (X)))
#define max(X, Y) (((X) > (Y))? (X):(Y))
#define min(X, Y) (((X) < (Y))? (X):(Y))

__kernel void stencil(__global int *A, __global int *B)
{
	int xid = get_global_id(0);
//	for(int i = 0; i < 100; i++)
		B[xid] =  A[xid + idx_ofst_x1] + A[xid + idx_ofst_x2];
}

__kernel void stencil_sharing(__global int *A, __global int *B, __local int *shr_A)
{
	int g_xid = get_global_id(0);
	int l_xid = get_local_id(0);
	int diff = (abs_diff(idx_ofst_x1, idx_ofst_x2));
	int bg = (min(idx_ofst_x1, idx_ofst_x2)) + get_group_id(0) * get_local_size(0);	
	int ed = (max(idx_ofst_x1, idx_ofst_x2)) + (get_group_id(0) + 1) * get_local_size(0);
	int wl_pwi = 1;
	int res = 1;
	res += (ed - bg) % get_local_size(0);
	int wi_ofst = l_xid * wl_pwi;
	if(l_xid < get_local_size(0) - 1){
	        shr_A[l_xid] = A[bg + l_xid];
	}else{
         	for(int i = wi_ofst; i < wi_ofst + res; i++){
			shr_A[i] = A[bg + i];
                }
	}
        barrier(CLK_LOCAL_MEM_FENCE);
		B[g_xid] = shr_A[l_xid] + shr_A[l_xid + diff];
}

