#define idx_ofst_x1 0
#define idx_ofst_x2 1
#define idx_ofst_x3 2
#define LENGTH (4096*1024)
#define local_sz 256
#define abs_diff(X, Y) (((X) > (Y))? ((X) -(Y)):((Y) - (X)))
#define abs_diff(X, Y) (((X) > (Y))? ((X) -(Y)):((Y) - (X)))
#define max2(X, Y) (((X) > (Y))? (X):(Y))
#define min2(X, Y) (((X) < (Y))? (X):(Y))
#define max3(X, Y, Z) ((max2((X), (Y))) > (Z)? (max2((X), (Y))):(Z))
#define min3(X, Y, Z) ((min2((X), (Y))) < (Z)? (min2((X), (Y))):(Z))
#define mid3(X, Y, Z) ((min2((X), (Y))) < (Z)? (max2((X), (Y))):(max2((min2((X), (Y))), (Z))))

__kernel void stencil(__global int *A, __global int *B)
{
	int xid = get_global_id(0);
//	for(int i = 0; i < 100; i++){
		B[xid] =  A[1 * xid + idx_ofst_x1] 
			+ A[1 * xid + idx_ofst_x2]
			+ A[1 * xid + idx_ofst_x3];
//	}
}

__kernel void stencil_sharing(__global int *A, __global int *B, __local int *shr_A)
{
	int g_xid = get_global_id(0);
	int l_xid = get_local_id(0);
	int max_ofst = (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3));
	int mid_ofst = (mid3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3));
	int min_ofst = (min3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3));
	int ofst_diff_max = (abs_diff(min_ofst, max_ofst));
	int ofst_diff_mid = (abs_diff(min_ofst, mid_ofst));
	int bg = min_ofst + get_group_id(0) * get_local_size(0);	
	int ed = max_ofst + (get_group_id(0) + 1) * get_local_size(0);
	int wl_pwi = 1;
	int res = 1;
	res += (ed - bg) % get_local_size(0);
	int wi_ofst = l_xid * wl_pwi;
	if(l_xid < get_local_size(0) - 1){
	        shr_A[wi_ofst] = A[bg + wi_ofst];
	}else{
         	for(int i = wi_ofst; i < wi_ofst + res; i++){
			shr_A[i] = A[bg + i];
                }
	}
        barrier(CLK_LOCAL_MEM_FENCE);
//	for(int i = 0; i < 100; i++){
		B[g_xid] = shr_A[l_xid] 
			 + shr_A[l_xid + ofst_diff_mid]
			 + shr_A[l_xid + ofst_diff_max];
//	}
}
