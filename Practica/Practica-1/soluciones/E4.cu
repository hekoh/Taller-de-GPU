#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>


#define CUDA_BLK 32


int *h_A, *h_B, *h_C, *h_check;
int *d_A, *d_B, *d_C;

__constant__ int d_N;
int N;

double dwalltime();
__global__ void funcionGPU(int N, int *A, int *B, int *C);

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}



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
   h_check = (int *) malloc(N*N*sizeof(int));

   for(int i = 0; i < N; i++){
	for(int j = 0; j < N; j++){
	    h_A[i*N +j] = rand() % N; //inicializo por filas
	    h_B[i+N*j] = rand() % N;  //inicializo por columnas
	    h_C[i*N+j] = 0;  
	    h_check[i*N+j] = 0;
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
   HANDLE_ERROR(cudaMemcpy(d_A, h_A, N*N*sizeof(int), cudaMemcpyHostToDevice)); // CPU -> GPU
   HANDLE_ERROR(cudaMemcpy(d_B, h_B, N*N*sizeof(int), cudaMemcpyHostToDevice));
   HANDLE_ERROR(cudaMemcpyToSymbol(d_N,&N,sizeof(int)));
   printf("tiempo de copia de memoria CPU => GPU: %f \n",dwalltime() - timetick);

   dim3 dimGrid(N / CUDA_BLK, N / CUDA_BLK,1);	//defino grid y bloque
   dim3 dimBlock(CUDA_BLK, CUDA_BLK);
   
   timetick = dwalltime();
   funcionGPU<<<dimGrid,dimBlock>>>(N, d_A, d_B, d_C);  //ejecuto kernel en GPU
   HANDLE_ERROR( cudaPeekAtLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );
   cudaDeviceSynchronize();	//sincronizo los hilos
   
   printf("\nTiempo de computo en GPU: %f",dwalltime() - timetick);

   cudaMemcpy(h_C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU


/********************** testeo resultado ********************/

   for(int i = 0; i < N; i++){
	for(int j = 0; j < N; j++){
	   for(int k = 0; k < N; k++)
		h_check[i*N+j] += h_A[i*N+k] * h_B[k+j*N];
	}
   }

   for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
		check = check && (h_check[i*N+j] == h_C[i*N+j]);
		printf("check: %d   gpu: %d\n",h_check[i*N+j],h_C[i*N+j]);
	}
   }

   
   if(check)
 	printf("\nLas matrices fueron multiplicadas correctamente\n");
   else
	printf("\nResultado erroneo, las matrices no fueron multiplicadas correctamente\n");


   free(h_A);
   cudaFree(d_A);
   free(h_B);
   cudaFree(d_B);
   free(h_C);
   cudaFree(d_C);
   free(h_check);
}





__global__ void funcionGPU (int N, int *A, int *B, int *C){
   int idX = blockIdx.x * blockDim.x + threadIdx.x;
   int idY = blockIdx.y * blockDim.y + threadIdx.y;
   if(idX < N && idY < N){
	for(int k = 0; k < N; k++){
		C[idY*N+idX] += A[idY*N+k] * B[k+idX*N];
	}
   }
}

	   
   
   


	














double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


