#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "2D5P_stencil.cl"
#define KERNEL_BASIC "stencil"
#define KERNEL_SHARING "stencil_sharing"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
#define b_width 120
#define b_height 120
#define l_width 20
#define l_height 20
#define a_height (b_height + (abs_diff(idx_ofst_y1, idx_ofst_y2)))
#define a_width (b_width + (abs_diff(idx_ofst_x1, idx_ofst_x2)))
#define abs(X) (((X) < 0)? (0 - (X)):(X))
#define adj_x (abs(min2(idx_ofst_x1, idx_ofst_x2)))
#define adj_y (abs(min2(idx_ofst_y1, idx_ofst_y2)))
#define shr_height (l_height + (abs_diff(idx_ofst_y1, idx_ofst_y2)))
#define shr_width (l_width + (abs_diff(idx_ofst_x1, idx_ofst_x2)))

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main() {
   /* Host/device data structures */
   cl_platform_id platform; 
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel_basic;
   cl_kernel kernel_sharing;
   size_t global_size[2] = {b_height, b_width};
   size_t local_size[2] = {l_height, l_width};
   size_t program_size, log_size;
   cl_int err, check;
   char* program_buffer;
   char* program_log;
   FILE* program_handle;
	
   /* Data and buffers */
   int *a_mat = (int*)malloc(sizeof(int) * (a_height) * (a_width));
   int *b_mat = (int*)malloc(sizeof(int) * (b_height) * (b_height));
   int *check_mat = (int*)malloc(sizeof(int) * (b_height) * (b_height));
   cl_mem a_buffer, b_buffer;
	   
   /* Initialize A, B, and check matrices */
   srand((unsigned int)time(0));
   for(int y = 0; y < a_height; y++){
	for(int x = 0; x < a_width; x++){
         a_mat[y * a_width + x] = (int)rand() / 100;
  //       a_mat[y * a_width + x] = 200;
	}
  }
   srand((unsigned int)time(0));
   for(int y = 0; y < b_height; y++){
	for(int x = 0; x < b_width; x++)
	         b_mat[y * b_width + x] = a_mat[(y + adj_y) * a_width + (x + adj_x)]
					+ a_mat[(y + adj_y) * a_width + (x + adj_x + idx_ofst_x1)]
					+ a_mat[(y + adj_y) * a_width + (x + adj_x + idx_ofst_x2)]
					+ a_mat[(y + adj_y + idx_ofst_y1) * a_width + (x + adj_x)]
					+ a_mat[(y + adj_y + idx_ofst_y2) * a_width + (x + adj_x)];
   }
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0){
        perror("Couldn't get the Platform");
        exit(1);    
   }
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err < 0){
        perror("Couldn't get the Device.");
        exit(1);
   }
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0){
        perror("Couldn't create the Context.");
   }
   
   //Create Queue 
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0){
        perror("Couldn't create the Queue.");
        exit(1);
   }
   // Create Buffer 
   a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(int) * (a_height) * (a_width), NULL, &err);
   if(err < 0){
        perror("A Buffer is not creatl_ed successfully.");
        exit(1);    
   }
   b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , sizeof(int) * (b_height) * (b_width), NULL, &err);
   if(err < 0){
        perror("B Buffer is not creatl_ed successfully.");
        exit(1);    
   }
   //Write into Buffer
   clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(int) * (a_height) * (a_width), a_mat, 0, NULL, NULL);
   if(err < 0){
      perror("Couldn't write into the Buffer.");
      exit(1);
   }
   //Build Program
   program_handle = fopen(PROGRAM_FILE, "r");
   if(program_handle ==NULL){
        perror("Couldn't find the program file.");
        exit(1);
   }

   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size +1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   //Create Program From File
   program = clCreateProgramWithSource(context, 1,(const char**)&program_buffer, &program_size, &err);
   if(err < 0){
        perror("Couldn't create the Progrom.");
        exit(1);
   }
   free(program_buffer);
   //Build program
   err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
   if(err < 0){
        //Find the size of log and print to std output
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size +1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
   }
   
   kernel_basic = clCreateKernel(program, KERNEL_BASIC, &err); 
   if(err < 0){
           perror("Couldn't create a multiplication Kernel");
           exit(1);
   }
   kernel_sharing = clCreateKernel(program, KERNEL_SHARING, &err); 
   if(err < 0){
           perror("Couldn't create a multiplication Kernel");
           exit(1);
   }
               
   if(l_width > (abs_diff(idx_ofst_x1, idx_ofst_x2)) && l_height > (abs_diff(idx_ofst_y1, idx_ofst_y2))){
	   err = clSetKernelArg(kernel_sharing, 0, sizeof(cl_mem), &a_buffer);
		printf("Sharing Kernel\n");
	   if(err < 0){
	        perror("Couldn't set an argument for the multiplication kernel_basic 0");
	        exit(1);
	   }
	   err |= clSetKernelArg(kernel_sharing, 1, sizeof(cl_mem), &b_buffer);
	   if(err < 0){
	           perror("Couldn't set an argument for the multiplication kernel_basic 1");
                   exit(1);
	    }
	   err |= clSetKernelArg(kernel_sharing, 2, sizeof(int) * (shr_width) * (shr_height), NULL);
	   if(err < 0){
	           perror("Couldn't set an argument for the multiplication kernel_basic 2");
                   exit(1);
	    }
   	    err = clEnqueueNDRangeKernel(queue, kernel_sharing, 2, NULL, global_size, local_size, 0, NULL, NULL);
	    if(err < 0){
	        perror("Couldn't enqueue the multiplication kernel_sharing.");
                printf("err: %d\n", err);
	        exit(1);
	    }
   }
   else{
		printf("Non-haring Kernel\n");
	   err = clSetKernelArg(kernel_basic, 0, sizeof(cl_mem), &a_buffer);
	   if(err < 0){
	        perror("Couldn't set an argument for the multiplication kernel_basic");
	        exit(1);
	   }
	   err |= clSetKernelArg(kernel_basic, 1, sizeof(cl_mem), &b_buffer);
	   if(err < 0){
	           perror("Couldn't set an argument for the multiplication kernel_basic");
                   exit(1);
	    }
   	    err = clEnqueueNDRangeKernel(queue, kernel_basic, 2, NULL, global_size, local_size, 0, NULL, NULL);
	    if(err < 0){
	        perror("Couldn't enqueue the multiplication kernel_basic.");
	        exit(1);
	    }
   }

   //Read output buffer
   err = clEnqueueReadBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(int) * (b_height) * (b_width), check_mat, 0, NULL, NULL);
   if(err < 0){
	printf("err: %d\n", err);
        perror("Couldn't read the output buffer");
        exit(1);
   }
   
   // Check result 
   check = 1;
   for(int y = 0; y < b_height; y++){
	for(int x = 0; x < b_width; x++){
	         if(b_mat[y * b_width + x] != check_mat[y * b_width + x]){
			check = 0;
	    		printf("CPU -> B[%d][%d]: %d\n", y, x, b_mat[y * b_width + x]);
			printf("GPU -> B[%d][%d]: %d\n", y, x, check_mat[y * b_width + x]);
			
		 }
	}
   }
   if(check)
      printf("basic check succeed.\n");
   else
      printf("basic check failed.\n");

   /* Deallocate resources */
   err = clReleaseMemObject(a_buffer);
   if(err < 0){
        perror("Fail to release C buffer.");
        exit(1);
   }
   err = clReleaseMemObject(b_buffer);
   if(err < 0){
        perror("Fail to release C buffer.");
        exit(1);
   }
   if(err < 0){
        perror("Fail to release C buffer.");
        exit(1);
   }
   clReleaseKernel(kernel_basic);
   clReleaseKernel(kernel_sharing);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
