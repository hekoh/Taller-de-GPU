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
__global__ void reduccion(int *V, int tam);
__global__ void calcularApariciones(int *V, int tam, int X);

int main (int argc, char *argv[])
{
   double timetick;
   int N,X;
   int limite;	

   if( (argc!=3) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0) || ((X = atoi(argv[2])) < 0) || X > 999)
   {
	printf("\n Error en los parametros. Pasar N = tamaño de matriz divisible por 32 \n X = elemento a encontrar entre 0 y 1000\n");
	exit(1);
   }

/******************** inicializo ********************/
   timetick = dwalltime();
   h_V = (int *) malloc(N*sizeof(int));
   cudaMalloc(&d_V, N*sizeof(int));

   for(int i = 0; i < N; i++){
	h_V[i] = rand() % 1000;
   }

   cudaMemcpy(d_V, h_V, N*sizeof(int), cudaMemcpyHostToDevice);
   
   dim3 dimBlock(CUDA_BLK);

  printf("tiempo de inizialicacion: %f\n",dwalltime() - timetick);

/******************** invocacion del kernel ********************/

   timetick = dwalltime();
   dim3 dimGrid((N / CUDA_BLK)+1,1,1);

   calcularApariciones<<<dimGrid,dimBlock>>>(d_V, N, X);
   cudaDeviceSynchronize();
   
   limite = N;

   while(limite > 1){
	dim3 dimGrid((limite / CUDA_BLK)+1,1,1);

	reduccion<<<dimGrid,dimBlock>>>(d_V,limite);
	cudaDeviceSynchronize();
	limite/= 2;
   } 
  
  printf("tiempo de reduccion: %f\n",dwalltime() - timetick);

/************************************************************/
   
   cudaMemcpy(h_V, d_V, N*sizeof(int),cudaMemcpyDeviceToHost);

   printf("Apariciones: %d\n",h_V[0]);

   free(h_V);
   cudaFree(d_V);
}

 


__global__ void calcularApariciones(int *V, int tam, int X){
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   
   if(id < tam){
	if(V[id] == X)
		V[id] = 1;
	else
		V[id] = 0;
   }
}


__global__ void reduccion(int *V, int tam){
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   int limite = (tam/2) +1;

  if(id < limite){
  	if(id + limite < tam)
		V[id] = V[id] + V[id+limite];
  } 
  

}



double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

