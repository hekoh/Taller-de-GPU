#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>


#define CUDA_BLK 64


int *h_array, *d_array;
int N,C;

__constant__ int d_N,d_C;

double dwalltime();
__global__ void funcionGPU(int C, int N, int* V);




/****************** MAIN *******************/
int main(int argc, char *argv[])
{   
   double timetick;
   bool check = true;
   
   if( (argc!=3) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0))
   {
	printf("\n Error en los parametros. Pasar N = tamaño de vector \n C = constante a multiplicar");
	exit(1);
   }
     
   C = atoi(argv[2]);


   h_array = (int *) malloc(N*sizeof(int)); 
   for(int i = 0; i < N; i++)
   	h_array[i] = i;

 
/****** reservo memoria ******/

   timetick = dwalltime();
   cudaMalloc(&d_array, N*sizeof(int));
   printf("tiempo de alocacion en memoria: %f \n",dwalltime() - timetick);

   timetick = dwalltime();
   cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
   cudaMemcpyToSymbol(d_C,&C,sizeof(int));
   cudaMemcpyToSymbol(d_N,&N,sizeof(int));
   printf("tiempo de copia de memoria CPU => GPU: %f \n",dwalltime() - timetick);

   dim3 dimGrid(N / CUDA_BLK,1,1);	//defino grid y bloque
   dim3 dimBlock(CUDA_BLK);
   
   timetick = dwalltime();
   funcionGPU<<<dimGrid,dimBlock>>>(d_C,d_N,d_array);	//ejecuto kernel en GPU
   cudaDeviceSynchronize();	//sincronizo los hilos
   
   printf("\nTiempo de computo en GPU: %f",dwalltime() - timetick);

   cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU

   for(int i = 0; i < N; i++){
	check = check && (h_array[i] == i*C);
   }
   
   if(check)
 	printf("\nEl arreglo fue multiplicado correctamente");
   else
	printf("\nResultado erroneo, la matriz no fue multiplicada correctamente");


   free(h_array);
   cudaFree(d_array);

}





__global__ void funcionGPU (int C, int N, int* V){
   
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id < N)
	V[id] = V[id] * C;
}
	   
   
   


	














double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


