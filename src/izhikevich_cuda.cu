#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

using namespace std;

#define Nexc 100 	// Количество возбуждающих (excitatory) нейронов
#define Ninh 25	    // Количество тормозных (inhibitory) нейронов
#define Nneur (Nexc + Ninh)

const int Ncon = Nneur * Nneur * 0.1;	// Количество сязей, 0.1 это вероятность связи между 2-мя случайными нейронами
const float h     = 0.5f; 	  // временной шаг интегрирования
const int   Tsim  = 1000/0.5f; // время симуляции в дискретных отсчетах

float psc_excxpire_time = 4.0f; // характерное вермя спадания постсинаптического тока, мс
float minWeight = 50.0f; // веса, размерность пкА
float maxWeight = 100.0f;

// Параметры нейрона
float Iex_max = 40.0f; // максимальный приложенный к нейрону ток 50 пкА
float a		  = 0.02f;
float b		  = 0.5f;
float c		  = -40.0f; // значение мембранного потенциала до которого он сбрасываеться после спайка
float d 	  = 100.0f;
float k		  = 0.5f;
float Vr	  = -60.0f;
float Vt	  = -45.0f;
float Vpeak	  = 35.0f;  // максимальное значение мембранного потенциала, при котором происходит сброс до значения с
float V0	  = -60.0f; // начальное значение для мембранного потенциала
float U0	  = 0.0f;   // начальное значение для вспомогательной переменной
float Cm      = 50.0f;  // электрическая ёмкость нейрона, размерность пкФ
float V_a     = 0.04f;
float V_b     = 5.0f;
float V_c     = 140.0f;

//float spike_times[Nneur*Tsim]; // времена спайков
//int spike_neurons[Nneur*Tsim]; // соответвующие номера нейронов
int spike_num = 0;

void init_connections(int *pre_conns, int *post_conns, float *weights) {
	for (int con_idx = 0; con_idx < Ncon; con_idx++){
		// случайно выбираем постсипантические и пресинаптические нейроны
		int pre = rand() % Nneur;
		int post = rand() % Nneur;
		pre_conns[con_idx] = pre;
		post_conns[con_idx] = post;
		weights[con_idx] = (rand() % ((int)(maxWeight - minWeight)*10))/10.0f + minWeight;
		if (pre >= Nexc){
			// если пресинаптический нейрон тормозный то вес связи идет со знаком минус
			weights[con_idx] = -weights[con_idx];
		}
	}
}

void init_neurons(float* Iex, float* Isyn,float *Vms, float *Ums){
	for (int neur_idx = 0; neur_idx < Nneur; neur_idx++){
		// случайно разбрасываем приложенные токи
		Iex[neur_idx] = (rand() % (int) (Iex_max*10))/10.0f;
		Isyn[neur_idx] = 0.0f;
		Vms[neur_idx] = V0; // Vms[t*Nneur + neur_idx] = V0;
		Ums[neur_idx] = U0; // Vms[t*Nneur + neur_idx] = V0;

	}
}

// Оригинальные функции

float izhik_Vm_CPU(int neuron, int time, float *Iex, float *Isyn, float *Vms, float *Ums){
	return (k*(Vms[time*Nneur + neuron] - Vr)*(Vms[time*Nneur + neuron] - Vt) - Ums[time*Nneur + neuron] + Iex[neuron] + Isyn[neuron])/Cm;
}

float izhik_Um_CPU(int neuron, int time, float *Vms, float *Ums){
	return a*(b*(Vms[time*Nneur + neuron] - Vr) - Ums[time*Nneur + neuron]);
}

// Функции по ТЗ
float izhik_Vm(int neuron, int time, float *Iex, float *Isyn, float **Vms, float **Ums){
	return (V_a*(Vms[neuron][time]*Vms[neuron][time]) + V_b*Vms[neuron][time] + V_c - Ums[neuron][time] + Iex[neuron] + Isyn[neuron])/Cm;
}

float izhik_Um(int neuron, int time, float **Vms, float **Ums){
	return a*(b*Vms[neuron][time]  - Ums[neuron][time]);
}

void save2file(float *spike_times, int *spike_neurons, float *Vms){
	ofstream res_file;
	res_file.open("rastr.csv");
	for (int k = 0; k < spike_num; k++){
		res_file << spike_times[k] << "; " << spike_neurons[k] + 1 << "; " << std::endl;
	}
	res_file.close();

	// Вычисление среднего по всей сети мембранного потенциала в каждый момент времени
	// нечто наподобие электроэнцефалографии
	res_file.open("oscill.csv");
	for (int t = 0; t < Tsim; t++){
		float Vm_mean= 0.0f;
		for (int m = 0; m < Nneur; m++){
			Vm_mean += Vms[t*Nneur + m];
		}
		Vm_mean /= Nneur;
		res_file << t*h << "; " << Vm_mean << "; " << endl;
	}
	res_file.close();
}



__global__ void izhikGPU(float *V, float *U, float *Iex, float *Isyn, int t, float Vr, float Vt, float Cm, float k){
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	int idx = iy*Nneur + ix;

	if (ix < Tsim && iy < Nneur)
	{
		V[t*Nneur + idx] = V[(t-1)*Nneur + idx] + h*(k*(V[(t-1)*Nneur + idx] - Vr)*(V[(t-1)*Nneur + idx] - Vt) - U[(t-1)*Nneur + idx] + Iex[idx] + Isyn[idx])/Cm;
	}
	//__syncthreads();
}

int main(int argc, char **argv)
{
	//int *pre_conns = new int[Ncon]; 	// индексы пресинаптических нейронов
	//int *post_conns = new int[Ncon]; 	// индексы постсинаптических нейронов
	//float *weights = new float[Ncon];	// веса связей
	//float *y = new float[Ncon*Tsim];	// переменная модулирующая синаптический ток в зависимости от спайков на пресинапсе
	//float *Iex = new float[Nneur*Tsim];		// внешний постоянный ток приложенный к нейрону
	//float *Isyn = new float[Nneur*Tsim];		// синаптичесий ток на каждый нейтрон
	//float *Vms = new float[Nneur*Tsim]; // мембранные потенциалы
	//float *Ums = new float[Nneur*Tsim]; // вспомогательные переменные модели Ижикевича
	//float *spike_times = new float[Nneur*Tsim];		// времена спайков
	//int *spike_neurons = new int[Nneur*Tsim];		// соответвующие номера нейронов

	int   *pre_conns;
	int   *post_conns;
	float *weights;
	float *y;
	float *Iex;
	float *Isyn;
	float *Vms;
	float *Ums;
	float *spike_times;
	int   *spike_neurons;

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	CHECK(cudaSetDevice(dev));

	__const__ float Vr	  = -60.0f;
	__const__ float Vt	  = -45.0f;
	__const__ float Cm    =  50.0f;
	__const__ float k     =  0.5f;

	cudaMallocManaged((void **)&pre_conns, Ncon*sizeof(int));
	cudaMallocManaged((void **)&post_conns, Ncon*sizeof(int));
	cudaMallocManaged((void **)&weights, Ncon*sizeof(int));
	cudaMallocManaged((void **)&y, Ncon*Tsim*sizeof(float));
	cudaMallocManaged((void **)&Iex, Nneur*Tsim*sizeof(float));
	cudaMallocManaged((void **)&Isyn, Nneur*Tsim*sizeof(float));
	cudaMallocManaged((void **)&Vms, Nneur*Tsim*sizeof(float));
	cudaMallocManaged((void **)&Ums, Nneur*Tsim*sizeof(float));
	cudaMallocManaged((void **)&spike_times, Nneur*Tsim*sizeof(float));
	cudaMallocManaged((void **)&spike_neurons, Nneur*Tsim*sizeof(int));

	memset(Vms,0,Nneur*Tsim*sizeof(float));
	memset(Ums,0,Nneur*Tsim*sizeof(float));
	memset(y, 0, Ncon*Tsim*sizeof(float));// иницилизируем 0
	memset(spike_times, 0, Nneur*Tsim*sizeof(float));
	memset(spike_neurons, 0, Nneur*Tsim*sizeof(int));
	init_connections(pre_conns, post_conns, weights);
	init_neurons(Iex,Isyn,Vms, Ums);

	//invoke kernel at host side
	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((Nneur + block.x - 1) / block.x, (Tsim + block.y - 1) / block.y);

	izhikGPU<<<grid, block>>>(Vms, Ums, Iex, Isyn,1, Vr, Vt, Cm, k);
	CHECK(cudaDeviceSynchronize());

	izhikGPU<<<grid, block>>>(Vms, Ums, Iex, Isyn,2, Vr, Vt, Cm, k);
	CHECK(cudaDeviceSynchronize());
	// check kernel error
    CHECK(cudaGetLastError());

   /*
   float expire_coeff = exp(-h/psc_excxpire_time);
   for (int t = 1; t < Tsim; t++){
   		// проходим по всем нейронам
   		for (int neur = 0; neur < Nneur; neur++){
   			Vms[t*Nneur + neur] = Vms[(t-1)*Nneur + neur] + h*izhik_Vm_CPU(neur, t-1, Iex, Isyn, Vms, Ums);
   			Ums[t*Nneur + neur] = Ums[(t-1)*Nneur + neur] + h*izhik_Um_CPU(neur, t-1, Vms, Ums);
   			//printf("izhik_Um_CPU = %f\n",izhik_Vm_CPU(neur, t-1, Iex, Isyn, Vms, Ums));
   			Isyn[neur] = 0.0f;
   			if (Vms[(t-1)*Nneur + neur] >Vpeak){
    			Vms[t*Nneur + neur] = c;
    			Ums[t*Nneur + neur] = Ums[(t-1)*Nneur + neur] + d;
    			spike_times[spike_num] = t*h;
    			spike_neurons[spike_num] = neur;
    			spike_num++;
    		}
    	}

    		// проходим по всем связям
    	for (int con = 0; con < Ncon; con++){
    		y[t*Ncon + con] = y[(t-1)*Ncon + con]*expire_coeff;
    		if (Vms[(t-1)*Nneur+(pre_conns[con])] > Vpeak){
    			y[t*Ncon + con] = 1.0f;
    		}
    			Isyn[post_conns[con]] += y[t*Ncon + con]*weights[con];
    	}
    }

    save2file(spike_times, spike_neurons, Vms);
    */
    for (int neur = 0; neur < Nneur; neur++){
    	//Vms[Nneur + neur] = Vms[ neur] + h*izhik_Vm_CPU(neur, 0, Iex, Isyn, Vms, Ums);
    	;
    }
    for(int i=0; i<100; i++){
    	//printf("Vms[i] = %f\n", Vms[Nneur + i]);
    	printf("Vms[i] = %f\n", Vms[2*Nneur + i]);
}
    CHECK(cudaFree(pre_conns));
    CHECK(cudaFree(post_conns));
    CHECK(cudaFree(weights));
    CHECK(cudaFree(y));
    CHECK(cudaFree(Iex));
    CHECK(cudaFree(Isyn));
    CHECK(cudaFree(Vms));
    CHECK(cudaFree(Ums));
    CHECK(cudaFree(spike_times));
    CHECK(cudaFree(spike_neurons));

    CHECK(cudaDeviceReset());

    //delete[] pre_conns;
    //delete[] post_conns;
    //delete[] weights;
    //delete[] y;
    //delete[] Iex;
    //delete[] Isyn;
    //delete[] Vms;
    //delete[] Ums;
    //delete[] spike_times;
    //delete[] spike_neurons;

    return EXIT_SUCCESS;
}
