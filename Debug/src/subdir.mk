################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/izhikevich_cuda.cu 

CU_DEPS += \
./src/izhikevich_cuda.d 

OBJS += \
./src/izhikevich_cuda.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


