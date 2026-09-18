#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>

#define CUDA_BLK 64

int *h_V;
int *d_V;



double dwalltime();
__global__ void calcularVector(int *V, float prom, int tam);
__global__ void reduccion(int *V, int tam);


int main (int argc, char *argv[])
{
   double timetick;
   float prom = 0;
   int N;
   int limite;	

   if( (argc!=2) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0) )
   {
	printf("\n Error en los parametros. Pasar N = tamaño de matriz \n");
	exit(1);
   }

/******************** inicializo ********************/
   timetick = dwalltime();
   h_V = (int *) malloc(N*sizeof(int));
   cudaMalloc(&d_V, N*sizeof(int));

   for(int i = 0; i < N; i++){
	h_V[i] = rand() % N;
   }

   cudaMemcpy(d_V, h_V, N*sizeof(int), cudaMemcpyHostToDevice);
   
   dim3 dimBlock(CUDA_BLK);

  printf("tiempo de inizialicacion: %f\n",dwalltime() - timetick);
   
/******************** calculo el promedio (CPU) ********************/
   timetick = dwalltime();   
   for(int i = 0; i < N; i++){
	prom+= h_V[i];
   }
   prom/= N;
  
  printf("tiempo de calculo de promedio: %f\n",dwalltime() - timetick);
  



/******************** invocacion del kernel ********************/

   timetick = dwalltime();
   dim3 dimGrid((N / CUDA_BLK)+1,1,1);

   calcularVector<<<dimGrid,dimBlock>>>(d_V, prom, N);
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

   printf("%d\n",h_V[0]);

   free(h_V);
   cudaFree(d_V);
}

 


__global__ void calcularVector(int *V, float prom, int tam){
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   float aux = 0;
   
   if(id < tam){
	aux = V[id] - prom;
	V[id] = aux*aux;    
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

