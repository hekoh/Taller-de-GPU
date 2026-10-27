#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>

#define CUDA_BLK 64
#define BLK 128

int *h_V;
int *d_V;



double dwalltime();
__global__ void reduccion(int *V);
__device__ void ultimoPaso(volatile int *SV, unsigned int tid);


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

int main (int argc, char *argv[])
{
   double timetick;
   int N, limite;

   if( (argc!=2) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0)){
	printf("\n Error en los parametros. Pasar N = tamaño de matriz divisible por 128\n");
	exit(1);
   }

/******************** inicializo ********************/
   timetick = dwalltime();
   h_V = (int *) malloc(N*sizeof(int));
   cudaMalloc(&d_V, N*sizeof(int));

   for(int i = 0; i < N; i++){
	h_V[i] = rand() % 100;
   }

   cudaMemcpy(d_V, h_V, N*sizeof(int), cudaMemcpyHostToDevice);
   
   unsigned int dimBlock = CUDA_BLK;

   printf("tiempo de inizialicacion: %f\n",dwalltime() - timetick);

/******************** invocacion del kernel ********************/

   timetick = dwalltime();
   dim3 dimGrid((N/CUDA_BLK)/2,1,1);

   reduccion<<<dimGrid,dimBlock,BLK*sizeof(int)>>>(d_V);
   HANDLE_ERROR( cudaPeekAtLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );

   printf("tiempo de reduccion: %f\n",dwalltime() - timetick);

/************************************************************/
   
   cudaMemcpy(h_V, d_V, N*sizeof(int),cudaMemcpyDeviceToHost);

   for(int i = 0; i < N/BLK; i++)
	printf("promedio: %d\n", (h_V[i]/BLK));

   free(h_V);
   cudaFree(d_V);
}



/************************************************************/

__global__ void reduccion(int *V){
  extern __shared__ volatile int SV[];
  unsigned int id = blockIdx.x * (blockDim.x*2) + threadIdx.x;
  
  SV[threadIdx.x] = V[id] + V[id + blockDim.x];

  __syncthreads();

/*  for(unsigned int s = blockDim.x/2; s >32; s >>= 1){
	if(threadIdx.x < s)
	   SV[threadIdx.x] += SV[threadIdx.x + s];
	__syncthreads();
  } */
  if(threadIdx.x < 32)
	ultimoPaso(SV, threadIdx.x);

  if(threadIdx.x == 0){
  	V[blockIdx.x] = SV[0];
  }
}


__device__ void ultimoPaso(volatile int *SV, unsigned int tid){
  	SV[tid] += SV[tid + 32];
	SV[tid] += SV[tid + 16];
	SV[tid] += SV[tid + 8];
	SV[tid] += SV[tid + 4];
	SV[tid] += SV[tid + 2];
	SV[tid] += SV[tid + 1];
}


double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

