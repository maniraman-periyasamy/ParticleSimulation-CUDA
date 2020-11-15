# ParticleSimulation-CUDA
A simple GPU accelerated particle simulator to simulate the movement of particle based on CUDA. The computational domain is split between GPU and CPU to share the work load.

Four different version of parallelization was implemented to benchmark various performanes.
  1. Full GPU implementation where the workload is full on GPU (particle_parallel_GPU.cu)
  2. CPU-GPU parllelization with Memcpy functionality of CUDA (particle_parallel_memcpy.cu)
  3. CPU-GPU parllelization using unified Memory of CUDA (particle_parallel_UMFB.cu, particle_parallel_UMVB.cu)
