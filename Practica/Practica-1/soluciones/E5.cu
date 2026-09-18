#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<sys/time.h>
#include<sys/resource.h>
#include <math.h>

#define CUDA_BLK 64


int *h_A,*h_max,*h_min;
int *d_A,*d_max,*d_min;

__constant__ int d_N;
int N;

double dwalltime();
__global__ void funcionGPU(int limite, int *min, int *max);




/****************** MAIN *******************/
int main(int argc, char *argv[])
{   
   double timetick;
   bool check = true;
   int grid;
   int limite;
   
   if( (argc!=2) || ((N = atoi(argv[1])) <= 0) || ((N % CUDA_BLK) != 0) )
   {
	printf("\n Error en los parametros. Pasar N = tamaño de matriz \n");
	exit(1);
   }
     
   

   timetick = dwalltime();
   h_A = (int *) malloc(N*sizeof(int));
   h_min = (int *) malloc(N*sizeof(int));
   h_max = (int *) malloc(N*sizeof(int));

   for(int i = 0; i < N; i++){
	h_A[i] = rand() % N;
	h_max[i] = h_A[i];
        h_min[i] = h_A[i];
   }
   printf("tiempo de alocacion y inicializacion en CPU: %f\n", dwalltime() - timetick);

/********** reservo memoria ***********/
   timetick = dwalltime();
   cudaMalloc(&d_A, N*sizeof(int));
   cudaMalloc(&d_min, N*sizeof(int));
   cudaMalloc(&d_max, N*sizeof(int));

   printf("tiempo de alocacion en memoria: %f \n",dwalltime() - timetick);

   timetick = dwalltime();
   cudaMemcpy(d_A, h_A, N*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
   cudaMemcpy(d_min, h_min, N*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU
   cudaMemcpy(d_max, h_max, N*sizeof(int), cudaMemcpyHostToDevice); // CPU -> GPU

   printf("tiempo de copia de memoria CPU => GPU: %f \n",dwalltime() - timetick);

     


/************** Computo ***************/
   limite = N;
   dim3 dimBlock(CUDA_BLK);
   timetick = dwalltime();

   while(limite > 1){
	   grid = ((limite / CUDA_BLK)+1); 
	   dim3 dimGrid(grid,1,1);	//defino grid y bloque

	   funcionGPU<<<dimGrid,dimBlock>>>(limite,d_min,d_max);  //ejecuto kernel en GPU
	   cudaDeviceSynchronize();	//sincronizo los hilos
	   limite = limite/2;
   }






/************************************************************************************/   
   
   printf("Tiempo de computo en GPU: %f\n",dwalltime() - timetick);

   cudaMemcpy(h_min, d_min, N*sizeof(int), cudaMemcpyDeviceToHost); // GPU -> CPU
   cudaMemcpy(h_max, d_max, N*sizeof(int), cudaMemcpyDeviceToHost);   
   
//   check = (h_min[0] == 0) && (h_max[0] == N-1);

   if(check)
 	printf("\n Los valores maximos y minimos fueron caluclados correctamente\n Min: %d\n Max:%d\n",h_min[0],h_max[0]);
   else
	printf("\nResultado erroneo, las matrices no fueron sumadas correctamente\n");

   free(h_A);
   free(h_max);
   free(h_min);
   cudaFree(d_max);
   cudaFree(d_min);
   cudaFree(d_A);
}




/********************** KERNEL ************************/
__global__ void funcionGPU (int tam, int *Vmin, int *Vmax){
   int id = blockIdx.x * blockDim.x + threadIdx.x;
   int min,max;
   int limite = (tam+1)/2;
   if(id < limite) {
	if((id + limite) < tam){
	   min = (Vmin[id+limite] < Vmin[id]) ? Vmin[id+limite] : Vmin[id];
	   max = (Vmax[id+limite] > Vmax[id]) ? Vmax[id+limite] : Vmax[id];
	}
	else {
	   min = Vmin[id];
           max = Vmax[id];
	}
	Vmin[id] = min;
	Vmax[id] = max;
   }
}
	   
   
   



double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


