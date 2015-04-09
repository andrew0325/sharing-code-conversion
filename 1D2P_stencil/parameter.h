#ifndef BOUNDARY_H
#define BOUNDARY_H
#define idx_ofst_x1 0
#define idx_ofst_x2 99
#define LENGTH 200
#define local_sz 100
#define abs_diff(X, Y) ((X) > (Y))? ((X) -(Y)):((Y) - (X))
#define max(X, Y) ((X) > (Y))? (X):(Y)
#define min(X, Y) ((X) < (Y))? (X):(Y)
#define shr_sz LENGTH + (max(idx_ofst_x1, idx_ofst_x2))
#endif
