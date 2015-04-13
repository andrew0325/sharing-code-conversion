#define idx_ofst_x1 1
#define idx_ofst_x2 (-1)
#define idx_ofst_y1 1
#define idx_ofst_y2 (-1)
#define abs_diff(X, Y) (((X) > (Y))? ((X) -(Y)):((Y) - (X)))
#define max2(X, Y) (((X) > (Y))? (X):(Y))
#define min2(X, Y) (((X) < (Y))? (X):(Y))
#define max3(X, Y, Z) ((max2((X), (Y))) > (Z)? (max2((X), (Y))):(Z))
#define min3(X, Y, Z) ((min2((X), (Y))) < (Z)? (min2((X), (Y))):(Z))
#define mid3(X, Y, Z) ((min2((X), (Y))) < (Z)? (max2((X), (Y))):(max2((min2((X), (Y))), (Z))))
#define b_width 1024
#define b_height 1024
#define l_width 16
#define l_height 16
#define a_height (b_height + (abs_diff(idx_ofst_y1, idx_ofst_y2)))
#define a_width (b_width + (abs_diff(idx_ofst_x1, idx_ofst_x2)))
#define abs(X) (((X) < 0)? (0 - (X)):(X))
#define adj_x (abs(min2(idx_ofst_x1, idx_ofst_x2)))
#define adj_y (abs(min2(idx_ofst_y1, idx_ofst_y2)))
#define shr_height (l_height + (abs_diff(idx_ofst_y1, idx_ofst_y2)))
#define shr_width (l_width + (abs_diff(idx_ofst_x1, idx_ofst_x2)))

__kernel void stencil(__global int *A, __global int *B)
{
	int xid = get_global_id(0);
	int yid = get_global_id(1);
//	for(int i = 0; i < 3; i++){
		B[yid * b_width + xid] =  A[(yid + adj_y) * a_width + (xid + adj_x)] 
					+ A[(yid + adj_y) * a_width + (xid + adj_x + idx_ofst_x1)]
					+ A[(yid + adj_y) * a_width + (xid + adj_x + idx_ofst_x2)]
					+ A[(yid + adj_y + idx_ofst_y1) * a_width + (xid + adj_x)]
					+ A[(yid + adj_y + idx_ofst_y2) * a_width + (xid + adj_x)];
//	}
}

__kernel void stencil_sharing(__global int *A, __global int *B, __local int *shr_A)
{
	int g_xid = get_global_id(0);
	int g_yid = get_global_id(1);
	int l_xid = get_local_id(0);
	int l_yid = get_local_id(1);
	int l_edge = (abs(min(idx_ofst_x1, idx_ofst_x2)));
	int r_edge = (max(idx_ofst_x1, idx_ofst_x2));
	int u_edge = (abs(min(idx_ofst_y1, idx_ofst_y2)));
	int d_edge = (max(idx_ofst_y1, idx_ofst_y2));
	int gp_x_ofst = get_group_id(0)* get_local_size(0);
	int gp_y_ofst = get_group_id(1)* get_local_size(1);
	if(get_local_id(0) == 0){
		for(int i = 0; i <= l_edge; i++)
                        shr_A[(l_yid + adj_y) * shr_width + i] = A[(gp_y_ofst + l_yid + adj_y) * a_width + (gp_x_ofst + i)];
	}
	else if(get_local_id(0) == (get_local_size(0) - 1)){
		for(int i = l_xid; i <= l_xid + r_edge; i++)
                        shr_A[(l_yid + adj_y) * shr_width + (adj_x + i)] = A[(gp_y_ofst + l_yid + adj_y) * a_width + (gp_x_ofst + adj_x + i)];
	}
	if(get_local_id(1) == 0){
		for(int j = 0; j <= u_edge; j++)
                        shr_A[j * shr_width + (l_xid + adj_x)] = A[(gp_y_ofst + j) * a_width + (gp_x_ofst + l_xid + adj_x)];
	}
	else if((get_local_id(1) == get_local_size(1) - 1)){
		for(int j = l_yid; j <= l_yid + d_edge; j++)
                        shr_A[(j + adj_y) * shr_width + (l_xid + adj_x)] = A[(gp_y_ofst + adj_y + j) * a_width + (gp_x_ofst + l_xid + adj_x)];
	}
	else
		shr_A[(l_yid + adj_y) * shr_width + (l_xid + adj_x)] = A[(g_yid + adj_y) * a_width + (g_xid + adj_x)];
        barrier(CLK_LOCAL_MEM_FENCE);
//	for(int i = 0; i < 3; i++){
		B[g_yid * b_width + g_xid] = shr_A[(l_yid + adj_y) * shr_width + (l_xid + adj_x)]
					   + shr_A[(l_yid + adj_y) * shr_width + (l_xid + adj_x + idx_ofst_x1)]
					   + shr_A[(l_yid + adj_y) * shr_width + (l_xid + adj_x + idx_ofst_x2)]
					   + shr_A[(l_yid + adj_y + idx_ofst_y1) * shr_width + (l_xid + adj_x)]
					   + shr_A[(l_yid + adj_y + idx_ofst_y2) * shr_width + (l_xid + adj_x)];
//	}
}

