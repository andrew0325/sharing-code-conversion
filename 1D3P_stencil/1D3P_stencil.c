#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "1D3P_stencil.cl"
#define KERNEL_BASIC "stencil"
#define KERNEL_SHARING "stencil_sharing"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define idx_ofst_x1 111
#define idx_ofst_x2 17
#define idx_ofst_x3 0
#define LENGTH 2000
#define local_sz 200
#define abs_diff(X, Y) (((X) > (Y))? ((X) -(Y)):((Y) - (X)))
#define max2(X, Y) (((X) > (Y))? (X):(Y))
#define min2(X, Y) (((X) < (Y))? (X):(Y))
#define max3(X, Y, Z) ((max2((X), (Y))) > (Z)? (max2((X), (Y))):(Z))
#define min3(X, Y, Z) ((min2((X), (Y))) < (Z)? (min2((X), (Y))):(Z))
#define mid3(X, Y, Z) ((min2((X), (Y))) < (Z)? (max2((X), (Y))):(max2((min2((X), (Y))), (Z))))
#define shr_sz (local_sz + (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3)) - (min3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3)))

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
   size_t global_size[1] = {LENGTH};
   size_t local_size[1] = {local_sz};
   size_t program_size, log_size;
   cl_int i, err, check;
   char* program_buffer;
   char* program_log;
   FILE* program_handle;

   /* Data and buffers */
   int *a_mat = (int*)malloc(sizeof(int) * (LENGTH + (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3))));
   int *b_mat = (int*)malloc(sizeof(int) * LENGTH);
   int *check_mat = (int*)malloc(sizeof(int) * LENGTH);
   cl_mem a_buffer, b_buffer;
   
   /* Initialize A, B, and check matrices */
   srand((unsigned int)time(0));
   for(i=0; i<LENGTH + (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3)); i++){
         a_mat[i] = (int)rand() / 100;
        // a_mat[i] = 200;
  }
   
   srand((unsigned int)time(0));
   for(i=0; i < LENGTH; i++){
         b_mat[i] = a_mat[1 * i + idx_ofst_x1] + a_mat[1 * i + idx_ofst_x2] + a_mat[1 * i + idx_ofst_x3];
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
   a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(int)*(LENGTH + (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3))), NULL, &err);
   if(err < 0){
        perror("A Buffer is not creatl_ed successfully.");
        exit(1);    
   }
   b_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , sizeof(int)*LENGTH, NULL, &err);
   if(err < 0){
        perror("B Buffer is not creatl_ed successfully.");
        exit(1);    
   }
   //Write into Buffer
   clEnqueueWriteBuffer(queue, a_buffer, CL_TRUE, 0, sizeof(int)*(LENGTH + (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3))), a_mat, 0, NULL, NULL);
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
                         
   //Create arguments for multiplication function
   int max_ofst = (max3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3));
   int min_ofst = (min3(idx_ofst_x1, idx_ofst_x2, idx_ofst_x3));
   int ofst_diff_max = (abs_diff(min_ofst, max_ofst));
   if(local_sz < ofst_diff_max){
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
   	    err = clEnqueueNDRangeKernel(queue, kernel_basic, 1, NULL, global_size, local_size, 0, NULL, NULL);
	    if(err < 0){
	        perror("Couldn't enqueue the multiplication kernel_basic.");
	        exit(1);
	    }
   }
   else{
	   err = clSetKernelArg(kernel_sharing, 0, sizeof(cl_mem), &a_buffer);
	   if(err < 0){
	        perror("Couldn't set an argument for the multiplication kernel_basic 0");
	        exit(1);
	   }
	   err |= clSetKernelArg(kernel_sharing, 1, sizeof(cl_mem), &b_buffer);
	   if(err < 0){
	           perror("Couldn't set an argument for the multiplication kernel_basic 1");
                   exit(1);
	    }
	   err |= clSetKernelArg(kernel_sharing, 2, sizeof(int) * shr_sz, NULL);
	   if(err < 0){
	           perror("Couldn't set an argument for the multiplication kernel_basic 1");
                   exit(1);
	    }
   	    err = clEnqueueNDRangeKernel(queue, kernel_sharing, 1, NULL, global_size, local_size, 0, NULL, NULL);
	    if(err < 0){
	        perror("Couldn't enqueue the multiplication kernel_basic.");
	        exit(1);
	    }
   }
    
   //Read output buffer
   err = clEnqueueReadBuffer(queue, b_buffer, CL_TRUE, 0, sizeof(int)*LENGTH, check_mat, 0, NULL, NULL);
   if(err < 0){
	printf("err: %d\n", err);
        perror("Couldn't read the output buffer");
        exit(1);
   }
   
   // Check result 
   check = 1;
   for(i=0; i<LENGTH; i++) {
         if(b_mat[i] != check_mat[i]) {
            check = 0;
	    printf("CPU -> B[%d]: %d\n", i, b_mat[i]);
	    printf("GPU -> B[%d]: %d\n", i, check_mat[i]);
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
