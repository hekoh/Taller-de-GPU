#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>


#define CUDA_BLK 64


int *h_A, *h_B, *h_C;
int *d_A, *d_B, *d_C;

__constant__ int d_N;
int N;

double dwalltime();
__global__ void funcionGPU(int N, int *A, int *B, int *C);




/****************** MAIN *******************/
int main(int argc, char *argv[])
{   
   double timetick;
   bool check = true;
   
   if( (argc!=2) || ((N = atoi(argv[1])) <= 0) || ((N*N % CUDA_BLK) != 0) )
   {
	printf("\n Error en los parametros. Pasar N = tamaño de matriz \n");
	exit(1);
   }
     
   

   timetick = dwalltime();
   h_A = (int *) malloc(N*N*sizeof(int));
   h_B = (int *) malloc(N*N*sizeof(int));
   h_C = (int *) malloc(N*N*sizeof(int));    

   for(int i = 0; i < N; i++){
	for(int j = 0; j < N; j++){
	    h_A[i*N +j] = rand() % N; //inicializo por filas
	    h_B[i*N+j] = rand() % N; 
	    h_C[i*N+j] = 0;  
	}
   }
   printf("tiempo de alocacion y inicializacion en CPU: %f\n", dwalltime() - timetick);
/****** reservo memoria ******/

   timetick = dwalltime();
   cudaMalloc(&d_A, N*N*sizeof(int));
   cudaMalloc(&d_B, N*N*sizeof(int));
   cudaMalloc(&d_C, N*N*sizeof(int));
   cudaMemset(d_C, 0, N*N*sizeof(int)); //inicializa en 0

   printf("tiempo de alocacion en memoria: %f \n",dwalltime() - timetick);

   timetick = dwalltime();
   cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
   cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_N,&N,sizeof(int));
   printf("tiempo de copia de memoria CPU => GPU: %f \n",dwalltime() - timetick);

   dim3 dimGrid(N*N / CUDA_BLK,1,1);	//defino grid y bloque
   dim3 dimBlock(CUDA_BLK);
   
   timetick = dwalltime();
   funcionGPU<<<dimGrid,dimBlock>>>(N, d_A, d_B, d_C);  //ejecuto kernel en GPU
   cudaDeviceSynchronize();	//sincronizo los hilos
   
   printf("\nTiempo de computo en GPU: %f",dwalltime() - timetick);

   cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU

   for(int i = 0; i < N; i++){
	for(int j = 0; j < N; j++){
	    check = check && (h_C[i*N+j] == h_A[i*N+j] + h_B[i*N+j]);
	}
   }
   
   if(check)
 	printf("\nLas matrices fueron sumadas correctamente\n");
   else
	printf("\nResultado erroneo, las matrices no fueron sumadas correctamente\n");


   free(h_A);
   cudaFree(d_A);
   free(h_B);
   cudaFree(d_B);
   free(h_C);
   cudaFree(d_C);
}





__global__ void funcionGPU (int N, int *A, int *B, int *C){
   
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   if(id < N*N)
	C[id] = A[id] + B[id];
}
	   
   
   


	














double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


