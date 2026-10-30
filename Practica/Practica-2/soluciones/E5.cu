#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>

#define CUDA_BLK 64

char *h_V, *d_V; 
char *h_VR, *d_VR;



double dwalltime();

__global__ void recorrido(char *V, char *VR);


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
   int N;
   char aux[4];
   int cant[4];

   if( (argc!=2) || ((N = atoi(argv[1])) <= 0)){
	printf("\n Error en los parametros. Pasar N = Tamaño de vector > 0\n");
	exit(1);
   }

/******************** inicializo ********************/
   timetick = dwalltime();
   h_V = (char *) malloc(N*sizeof(char));
   h_VR = (char *) malloc(N*sizeof(char));
   cudaMalloc(&d_V, N*sizeof(char));
   cudaMalloc(&d_VR, N*sizeof(char));
   aux[0] = 'A';
   aux[1] = 'C';
   aux[2] = 'G';
   aux[3] = 'T';
   
   for(int i = 0; i < N; i++){
	h_V[i] = aux[rand() % 4];
   }
   for(int i = 0; i < 4; i++){
	h_VR[i] = 0;	 
	cant[i] = 0;
   }

    cudaMemcpy(d_V, h_V, N*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_VR, h_VR, N*sizeof(char), cudaMemcpyHostToDevice);
 
   unsigned int dimBlock = CUDA_BLK;

   printf("tiempo de inizialicacion: %f\n",dwalltime() - timetick);

/******************** invocacion del kernel ********************/

   timetick = dwalltime();
   dim3 dimGrid(((N+CUDA_BLK-1)/CUDA_BLK),1,1);

   recorrido<<<dimGrid,dimBlock,CUDA_BLK*sizeof(char)>>>(d_V,d_VR);
   HANDLE_ERROR( cudaPeekAtLastError() );
   HANDLE_ERROR( cudaDeviceSynchronize() );

   printf("tiempo de reduccion: %f\n",dwalltime() - timetick);

/************************************************************/
   
   cudaMemcpy(h_V, d_V, N*sizeof(char),cudaMemcpyDeviceToHost);
   cudaMemcpy(h_VR, d_VR, N*sizeof(char),cudaMemcpyDeviceToHost);

 
   char auxC;
   for(int i = 0; i < N; i++){
	auxC = h_VR[i];
	switch(auxC){
	   case 'A':
		cant[0]++;
		break;
	   case 'C':
		cant[1]++;
		break;
	   case 'G':
		cant[2]++;
		break;
	   case 'T':
		cant[3]++;
		break;
	   case 'X':
		break;
	}
   }
	
   printf("AAAA: %d\n CCCC: %d\n GGGG: %d\n TTTT: %d\n", cant[0], cant[1], cant[2], cant[3]);

   free(h_V);
   free(h_VR);
   cudaFree(d_V);
   cudaFree(d_VR);
}



/************************************************************/


__global__ void recorrido(char *V, char *VR){
   char aux;
   int cant = 1;
   __shared__ char VS[CUDA_BLK-3];  
   
   int id = blockIdx.x * blockDim.x + threadIdx.x;

   VS[threadIdx.x] = V[id];

   __syncthreads();

   if(threadIdx.x < CUDA_BLK-3){
	aux = VS[threadIdx.x];
	for(int i = 1; i < 4; i++)
		cant+= aux == VS[threadIdx.x + i];
	if(cant == 4)
	   VR[id] = aux;
	else
	   VR[id] = 'X';
   }
}


double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

