#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>

#define CUDA_BLK 32

int *h_V;
int *d_V;



double dwalltime();
__global__ void sortImpar(int *V, int tam);
__global__ void sortPar(int *V, int tam);


int main (int argc, char *argv[])
{
   double timetick;
   int N;

   if( (argc!=2) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0))
   {
	printf("\n Error en los parametros. Pasar N = tamaño de matriz divisible por 32 \n");
	exit(1);
   }

/******************** inicializo ********************/
   timetick = dwalltime();
   h_V = (int *) malloc(N*sizeof(int)+1); //padding para el ultimo elemento
   cudaMalloc(&d_V, N*sizeof(int));

   for(int i = 0; i < N; i++){
	h_V[i] = rand() % 1000;
   }
   h_V[N] = 1;

   cudaMemcpy(d_V, h_V, N*sizeof(int), cudaMemcpyHostToDevice);
   
   dim3 dimBlock(CUDA_BLK);

  printf("tiempo de inizialicacion: %f\n",dwalltime() - timetick);

/******************** invocacion del kernel ********************/

   timetick = dwalltime();
   dim3 dimGrid((N / CUDA_BLK)+1,1,1);

   sortImpar<<<dimGrid,dimBlock>>>(d_V,N);
   cudaDeviceSynchronize();
   sortPar<<<dimGrid,dimBlock>>>(d_V,N);
   cudaDeviceSynchronize();
  
  printf("tiempo de reduccion: %f\n",dwalltime() - timetick);

/************************************************************/
   
   cudaMemcpy(h_V, d_V, N*sizeof(int),cudaMemcpyDeviceToHost);

   printf("Apariciones: %d\n",h_V[0]);

   free(h_V);
   cudaFree(d_V);
}

 


__global__ void sortPar(int *V, int tam){
   extern __shared__ int vector[];
   unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
   
   vector[threadIdx.x] = V[id];
   __syncthreads();

  if(id*2+2 < tam){
  	V[id*2+1] *= V[id*2+2];
  }

}


__global__ void sortImpar(int *V, int tam){
   extern __shared__ int vector[];
   unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

   vector[threadIdx.x] = V[id];

   if(id*2+1 < tam){
	V[id*2] *= V[id*2+1];
   }
}



double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

