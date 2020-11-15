# include <iostream>
# include <fstream>
# include <string>
# include <iomanip>
# include <cmath>
# include <istream>
# include <cuda.h>
# include <cuda_runtime.h>
# include <cassert>
# include <sys/stat.h>
# include <chrono>
# include <ctime>
# include <typeinfo>
# include <thrust/tuple.h>
# include <omp.h>
# define data_type double

using namespace std;

string part_input_file, part_out_name_base, vtk_out_name_base;
data_type timestep_length, time_end, epsilon, sigma, part_out_freq, vtk_out_freq, cl_workgroup_1dsize;
unsigned out_count = 0, vtk_count = 0, x_n, y_n, z_n, cell_size;

void checkError(cudaError_t err){
	if(err!= cudaSuccess){
		std::cout<<cudaGetErrorString(err)<<std::endl;
		exit(-1);
	}
}


//************************************** Structs for variables **************************************************

template <typename T>
struct particle{
	T x;
	T y;
	T z;
	T vx;
	T vy;
	T vz;
	T m;
	T force_x;
	T force_y;
	T force_z;
	int cell_num;
};

template <typename T>
struct Domain {
	T x_min;
	T x_max;
	T y_min;
	T y_max;
	T z_min;
	T z_max;
	unsigned int x_n;
	unsigned int y_n;
	unsigned int z_n;
	T r_cut;
	T x_len;
	T y_len;
	T z_len;
	int cell_size;
	int thread_size;
	int block_size;
	int cpu_x = 20;
	T max_vel = 48; //ramdom value either read from par file or taken as given
} ;

Domain<data_type> domain;

//************************************** Input Parameters from Par **************************************************

void input_para(string name)
{
    string testline;
    string::size_type sz;
    ifstream Input (name);

    if (!Input)
    {
        cout << "There was a problem opening the file. Press any key to close.\n";
    }
    while( Input.good() )
    {
        getline ( Input, testline, ' ');
        if (testline!="")
        {
            if (testline == "part_input_file")
            {
                getline ( Input >> ws, testline);
                if (isalpha(testline[testline.size() - 1]) == false)
                    testline.pop_back();
                part_input_file = testline ;
            }
            else if (testline == "timestep_length")
            {
                getline ( Input, testline);
                timestep_length = stof(testline, &sz) ;
            }
            else if (testline == "time_end")
            {
                getline ( Input, testline);
                time_end = stof(testline, &sz) ;
            }
            else if (testline == "epsilon")
            {
                getline ( Input, testline);
                epsilon = stof(testline, &sz) ;
            }
            else if (testline == "sigma")
            {
                getline ( Input, testline);
                sigma = stof(testline, &sz) ;
            }
            else if (testline == "part_out_freq")
            {
                getline ( Input, testline);
                part_out_freq = stof(testline, &sz) ;
            }
            else if (testline == "part_out_name_base")
            {
                getline ( Input >> ws, testline);
                if (isalpha(testline[testline.size() - 1]) == false)
                    testline.pop_back();
                part_out_name_base = testline ;
            }
            else if (testline == "vtk_out_freq")
            {
                getline ( Input, testline);
                vtk_out_freq = stof(testline, &sz) ;
            }
            else if (testline == "vtk_out_name_base")
            {
                getline ( Input >> ws, testline);
                if (isalpha(testline[testline.size() - 1]) == false)
                    testline.pop_back();
                vtk_out_name_base = testline ;
            }
            else if (testline == "cl_workgroup_1dsize")
            {
                getline ( Input, testline);
                domain.thread_size = stoi(testline);
            }
            else if (testline == "cl_workgroup_3dsize_x")
            {
                getline ( Input, testline);
            }
            else if (testline == "cl_workgroup_3dsize_y")
            {
                getline ( Input, testline);
            }
            else if (testline == "cl_workgroup_3dsize_z")
            {
                getline ( Input, testline);
            }
            else if (testline == "x_min")
            {
                getline ( Input, testline);
                domain.x_min = stof(testline, &sz) ;
            }
            else if (testline == "y_min")
            {
                getline ( Input, testline);
                domain.y_min = stof(testline, &sz) ;
            }
            else if (testline == "z_min")
            {
                getline ( Input, testline);
                domain.z_min = stof(testline, &sz) ;
            }
            else if (testline == "x_max")
            {
                getline ( Input, testline);
                domain.x_max = stof(testline, &sz) ;
            }
            else if (testline == "y_max")
            {
                getline ( Input, testline);
                domain.y_max = stof(testline, &sz) ;
            }
            else if (testline == "z_max")
            {
                getline ( Input, testline);
                domain.z_max = stof(testline, &sz) ;
            }
            else if (testline == "r_cut")
            {
                getline ( Input, testline);
                domain.r_cut = stof(testline, &sz) ;
            }
            else if (testline == "x_n")
            {
                getline ( Input, testline);
                domain.x_n = stoul(testline, nullptr,0) ;
            }
             else if (testline == "y_n")
            {
                getline ( Input, testline);
                domain.y_n = stoul(testline, nullptr,0) ;
            }
             else if (testline == "z_n")
            {
                getline ( Input, testline);
                domain.z_n = stoul(testline, nullptr,0) ;
            }
            else if (testline == "max_vel")
            {
                getline ( Input, testline);
                domain.max_vel = stof(testline, &sz) ;
            }
            else if (testline == "cpu_x")
            {
                getline ( Input, testline);
                domain.cpu_x = stof(testline, &sz) ;
            }
        }
    }
    

    domain.x_len = (domain.x_max - domain.x_min)/domain.x_n;
    domain.y_len = (domain.y_max - domain.y_min)/domain.y_n;
    domain.z_len = (domain.z_max - domain.z_min)/domain.z_n;
    domain.cell_size = ((domain.x_max - domain.x_min)*(domain.y_max - domain.y_min)*(domain.z_max - domain.z_min))/(domain.x_len*domain.y_len*domain.z_len);  
    domain.cpu_x = int(domain.x_n*domain.cpu_x/100);
    /*cout<<"part_input_file : "<<part_input_file<<"\t"<<part_input_file.size()<<endl;
    cout<<"timestep_length : "<<timestep_length<<endl;
    cout<<"time_end : "<<time_end<<endl;
    cout<<"epsilon : "<<epsilon<<endl;
    cout<<"sigma : "<<sigma<<endl;
    cout<<"part_out_freq : "<<part_out_freq<<endl;
    cout<<"part_out_name_base : "<<part_out_name_base<<"\t"<<part_out_name_base.size()<<endl;
    cout<<"vtk_out_freq : "<<vtk_out_freq<<endl;
    cout<<"vtk_out_name_base : "<<vtk_out_name_base<<"\t"<<vtk_out_name_base.size()<<endl;
    cout<<"x_min: "<<domain.x_min<<endl; 
    cout<<"y_min: "<<domain.y_min<<endl; 
    cout<<"z_min: "<<domain.z_min<<endl; 
    cout<<"x_max: "<<domain.x_max<<endl; 
    cout<<"y_max: "<<domain.y_max<<endl; 
    cout<<"z_max: "<<domain.z_max<<endl; 
    cout<<"x_n: "<<domain.x_n<<endl; 
    cout<<"y_n: "<<domain.y_n<<endl; 
    cout<<"z_n: "<<domain.z_n<<endl; 
    cout<<"r_cut: "<<domain.r_cut<<endl;
    cout<<"max_vel: "<<domain.max_vel<<endl;
    cout<<"cpu_x: "<<domain.cpu_x<<endl;*/
    
    
    struct stat st;
    if(stat(&part_out_name_base[0],&st) != 0)
    {
        const int dir_err = mkdir(&part_out_name_base[0], S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {   
            printf("Error creating directory!n");
            exit(1);
        }
        
    }
    
    string subName = part_out_name_base+ "/Particle_parallel_";
    
    if(stat(&subName[0],&st) != 0)
    {
        const int dir_err = mkdir(&subName[0], S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {   
            printf("Error creating directory!n");
            exit(1);
        }
        
    }
    
    
    
    
}

//************************************** Initializing Particles **************************************************
template <typename T>
__global__ void init_particle (particle<T> *elements, int num_par, int *cpu_par, int *gpu_par)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    while(id<num_par)
        {
            elements[id].cell_num = id;
            cpu_par[id] = -1;
            gpu_par[id] = -1;
            id = id + blockDim.x * gridDim.x;
        }
}

//************************************** Initializing & updating cell **************************************************
template <typename T>
 __global__ void init_cell (Domain<T> domain, int *cell_arr)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    while(id<domain.cell_size)
        {
            cell_arr[id] = -1;
            id = id + blockDim.x * gridDim.x;
        }
}


template <typename T>
__device__ void update_cell_calc (particle<T> *elements, int id, Domain<T> domain, int *cell_arr, int *cpu_par, int *gpu_par)
{
    
        int x_cell = (elements[id].x-domain.x_min)/domain.x_len;
        int y_cell = (elements[id].y-domain.y_min)/domain.y_len;
        int z_cell = (elements[id].z-domain.z_min)/domain.z_len;
        int cell_count = x_cell + y_cell*domain.x_n + z_cell*domain.x_n*domain.y_n;
        elements[id].cell_num = atomicExch(&cell_arr[cell_count], elements[id].cell_num);
        if(x_cell <= domain.cpu_x)
            cpu_par[id] = id;
        else
            gpu_par[id] = id;
  
}

template <typename T>
__global__ void update_cell (particle<T> *elements, int num_par, Domain<T> domain, int *cell_arr, int *cpu_par, int *gpu_par)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while(id<num_par)
    {
        update_cell_calc<T>(elements, id, domain, cell_arr, cpu_par, gpu_par);
        id = id + blockDim.x * gridDim.x;
    }
        
}



//************************************** Force Calculation using Lenards **************************************************
template <typename T>
__device__ thrust::tuple<T,T,T> force ( T dist_x1, T dist_y1, T dist_z1, particle<T> *elements, T epsilon, T sigma, int id , int curr_pos, int coord, Domain<T> domain, int *cell_arr)
{
    T result_x = 0,result_y = 0,result_z = 0;
    T x,y,z, mag;
    
    int temp = id;
    int curr_z = temp/(domain.x_n*domain.y_n);
    temp -=  curr_z*domain.x_n*domain.y_n;
    int curr_y = temp/domain.x_n;
    int curr_x = temp % domain.x_n;
    
    
   for(int p = curr_x-1; p<curr_x+2; ++p)
    {
        for(int q = curr_y-1; q<curr_y+2; ++q)
        {
            for(int r = curr_z-1; r<curr_z+2; ++r)
    
            {
                if ( p >= 0 && p < domain.x_n && q >= 0 && q < domain.y_n && r >= 0 && r < domain.z_n)
                {
                    int cell_index = p + q*domain.y_n + r*domain.y_n*domain.x_n;
                    int i = cell_arr[cell_index];
                
                    while (i != -1)
                    {
                        
                        if (i != curr_pos)
                        {       
                
                        if( dist_x1 < elements[i].x)
                        {   
                            if( abs(dist_x1 - elements[i].x) <= (abs(dist_x1 - domain.x_min) + abs(domain.x_max-elements[i].x)))
                                x = dist_x1 - elements[i].x ;
                        
                            else
                                x = (abs(dist_x1 - domain.x_min) + abs(domain.x_max-elements[i].x));
                        }
                        else
                        {
                            if(abs(elements[i].x - dist_x1) <= (abs(domain.x_max - dist_x1) + abs(elements[i].x-domain.x_min)))
                                x = dist_x1 - elements[i].x ;
                            else
                                x = -(abs(domain.x_max - dist_x1) + abs(elements[i].x-domain.x_min));
                        }
                    
                        if( dist_y1 < elements[i].y)
                        {   
                            if( abs(dist_y1 - elements[i].y) <= (abs(dist_y1 - domain.y_min) + abs(domain.y_max-elements[i].y)))
                                y = dist_y1 - elements[i].y ;
                            else
                                y = (abs(dist_y1 - domain.y_min) + abs(domain.y_max-elements[i].y));
                        }
                        else
                        {
                            if(abs(elements[i].y - dist_y1) <= (abs(domain.y_max - dist_y1) + abs(elements[i].y-domain.y_min)))
                                y = dist_y1 - elements[i].y ;
                            else
                                y = -(abs(domain.y_max - dist_y1) + abs(elements[i].y-domain.y_min));
                        }
                
                        if( dist_z1 < elements[i].z)
                        {   
                            if( abs(dist_z1 - elements[i].z) <= (abs(dist_z1 - domain.z_min) + abs(domain.z_max-elements[i].z)))
                                z = dist_z1 - elements[i].z ;
                            else
                                z = (abs(dist_z1 - domain.z_min) + abs(domain.z_max-elements[i].z));
                        }
                        else
                        {
                            if(abs(elements[i].z - dist_z1) <= (abs(domain.z_max - dist_z1) + abs(elements[i].z-domain.z_min)))
                                z = dist_z1 - elements[i].z ;
                            else
                                z = -(abs(domain.z_max - dist_z1) + abs(elements[i].z-domain.z_min));
                        }
                
                
                        // magnitude of the distance vector
                        mag = sqrt(pow(x,2)+ pow(y,2) + pow(z,2));
                        if(mag<=domain.r_cut)
                        {
            
                                result_x= result_x + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * x ));
                                result_y= result_y + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * y ));
                                result_z= result_z + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * z ));
                        }
                    
                    
                    }
                    i = elements[i].cell_num;
                }
                }
            }
        }
    }

    return thrust::make_tuple(result_x,result_y,result_z) ;
}


//************************************** Initialization of Forces **************************************************
template <typename T>
__global__ void init_force ( particle<T> *elements, T epsilon, T sigma, int num_par, Domain<T> domain, int * cell_arr)
{
        int curr_pos = threadIdx.x + blockIdx.x * blockDim.x;
        while( curr_pos < num_par)
        {
            thrust::tuple<T,T,T> result;
            int x_cell = (elements[curr_pos].x-domain.x_min)/domain.x_len;
            int y_cell = (elements[curr_pos].y-domain.y_min)/domain.y_len;
            int z_cell = (elements[curr_pos].z-domain.z_min)/domain.z_len;
            int id = x_cell + y_cell*domain.x_n + z_cell*domain.x_n*domain.y_n;
            
            result = force<T> (elements[curr_pos].x, elements[curr_pos].y, elements[curr_pos].z, elements, epsilon, sigma, id, curr_pos, 1, domain, cell_arr);         
            elements[curr_pos].force_x = thrust::get<0>(result);
            elements[curr_pos].force_y = thrust::get<1>(result);
            elements[curr_pos].force_z = thrust::get<2>(result);
            curr_pos = curr_pos + blockDim.x * gridDim.x;
        }
        
        
}


//************************************** Kernel to calculate force **************************************************
template <typename T>
__global__ void main_kernel_force (particle<T> *elements, T delta_t, T epsilon, T sigma, int num_par, Domain<T> domain, int * cell_arr, int *gpu_par )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    T force_x_o, force_y_o, force_z_o;
    
    while( id < num_par)
    {  
        if(gpu_par[id] != -1)
        {
            int x_cell = (elements[id].x-domain.x_min)/domain.x_len;
            int y_cell = (elements[id].y-domain.y_min)/domain.y_len;
            int z_cell = (elements[id].z-domain.z_min)/domain.z_len;
            int idc = x_cell + y_cell*domain.x_n + z_cell*domain.x_n*domain.y_n;
            
            thrust::tuple<T,T,T> result;
            
            force_x_o = elements[id].force_x;
            force_y_o = elements[id].force_y;
            force_z_o = elements[id].force_z;
    
            result = force<T>  (elements[id].x, elements[id].y, elements[id].z, elements, epsilon, sigma, idc, id, 1, domain, cell_arr); // Shooting
            elements[id].force_x = thrust::get<0>(result);
            elements[id].force_y = thrust::get<1>(result);
            elements[id].force_z = thrust::get<2>(result);
        
            elements[id].vx = elements[id].vx + ((force_x_o+elements[id].force_x)*delta_t/(2*elements[id].m));
            if(abs(elements[id].vx)>domain.max_vel)  elements[id].vx = domain.max_vel*(elements[id].vx/abs(elements[id].vx));
            elements[id].vy = elements[id].vy + ((force_y_o+elements[id].force_y)*delta_t/(2*elements[id].m));
            if(abs(elements[id].vy)>domain.max_vel)  elements[id].vy = domain.max_vel*(elements[id].vy/abs(elements[id].vy));
            elements[id].vz = elements[id].vz + ((force_z_o+elements[id].force_z)*delta_t/(2*elements[id].m));
            if(abs(elements[id].vz)>domain.max_vel)  elements[id].vz = domain.max_vel*(elements[id].vz/abs(elements[id].vz));
                
        }
        id = id + blockDim.x * gridDim.x;
    }
}








//************************************** Host to calculate force **************************************************
template <typename T>
__host__ void main_host_force (particle<T> *elements, T delta_t, T epsilon, T sigma, int num_par, Domain<T> domain, int * cell_arr, int *cpu_par )
{
  
    
  
  #pragma omp parallel for
  for(int j=0 ; j < num_par; ++j)
    {  
    
    if(cpu_par[j] != -1)
    {
       T force_x_o, force_y_o, force_z_o;
        int x_cell = (elements[j].x-domain.x_min)/domain.x_len;
        int y_cell = (elements[j].y-domain.y_min)/domain.y_len;
        int z_cell = (elements[j].z-domain.z_min)/domain.z_len;
        int idc = x_cell + y_cell*domain.x_n + z_cell*domain.x_n*domain.y_n;
           
          
           
            
        force_x_o = elements[j].force_x;
        force_y_o = elements[j].force_y;
        force_z_o = elements[j].force_z;          
            
            
        T result_x = 0;
        T result_y = 0;
        T result_z = 0;
        T x,y,z, mag;
        T dist_x1 = elements[j].x;
        T dist_y1 = elements[j].y;
        T dist_z1 = elements[j].z;
        int temp = idc;
        int curr_z = temp/(domain.x_n*domain.y_n);
        temp -=  curr_z*domain.x_n*domain.y_n;
        int curr_y = temp/domain.x_n;
        int curr_x = temp % domain.x_n;
    
        
        for(int p = curr_x-1; p<curr_x+2; ++p)
        {
            for(int q = curr_y-1; q<curr_y+2; ++q)
            {
        
                for(int r = curr_z-1; r<curr_z+2; ++r)
                {
                    if ( p >= 0 && p < domain.x_n && q >= 0 && q < domain.y_n && r >= 0 && r < domain.z_n)
                    {
                        int cell_index = p + q*domain.y_n + r*domain.y_n*domain.x_n;
                        int i = cell_arr[cell_index];
                
                        while (i != -1)
                        {
                            if (i != j)
                            {       
                
                                if( dist_x1 < elements[i].x)
                                {   
                                    if( abs(dist_x1 - elements[i].x) <= (abs(dist_x1 - domain.x_min) + abs(domain.x_max-elements[i].x)))
                                        x = dist_x1 - elements[i].x ;
                        
                                    else
                                        x = (abs(dist_x1 - domain.x_min) + abs(domain.x_max-elements[i].x));
                                }
                                else
                                {
                                    if(abs(elements[i].x - dist_x1) <= (abs(domain.x_max - dist_x1) + abs(elements[i].x-domain.x_min)))
                                        x = dist_x1 - elements[i].x ;
                                else
                                    x = -(abs(domain.x_max - dist_x1) + abs(elements[i].x-domain.x_min));
                                }
                    
                                if( dist_y1 < elements[i].y)
                                {   
                                    if( abs(dist_y1 - elements[i].y) <= (abs(dist_y1 - domain.y_min) + abs(domain.y_max-elements[i].y)))
                                        y = dist_y1 - elements[i].y ;
                                    else
                                        y = (abs(dist_y1 - domain.y_min) + abs(domain.y_max-elements[i].y));
                                }
                                else
                                {
                                    if(abs(elements[i].y - dist_y1) <= (abs(domain.y_max - dist_y1) + abs(elements[i].y-domain.y_min)))
                                        y = dist_y1 - elements[i].y ;
                                else
                                    y = -(abs(domain.y_max - dist_y1) + abs(elements[i].y-domain.y_min));
                                }
                                if( dist_z1 < elements[i].z)
                                {   
                                    if( abs(dist_z1 - elements[i].z) <= (abs(dist_z1 - domain.z_min) + abs(domain.z_max-elements[i].z)))
                                        z = dist_z1 - elements[i].z ;
                                    else
                                        z = (abs(dist_z1 - domain.z_min) + abs(domain.z_max-elements[i].z));
                                }
                                else
                                {
                                    if(abs(elements[i].z - dist_z1) <= (abs(domain.z_max - dist_z1) + abs(elements[i].z-domain.z_min)))
                                        z = dist_z1 - elements[i].z ;
                                    else
                                        z = -(abs(domain.z_max - dist_z1) + abs(elements[i].z-domain.z_min));
                                }
                
                
                                // magnitude of the distance vector
                                mag = sqrt(pow(x,2)+ pow(y,2) + pow(z,2));
                                if(mag<=domain.r_cut)
                                {
                           
                                    result_x= result_x + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * x ));
                                    result_y= result_y + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * y ));
                                    result_z= result_z + (((24 * epsilon/ (mag * mag)) * (pow((sigma/mag),6)) * (2 * (pow((sigma/mag),6)) -1 ) * z ));
                                }
                    
                    
                            }
                            i = elements[i].cell_num;
                        }
                    }
                }
            }
        }
        
        
        elements[j].force_x = result_x;
        elements[j].force_y = result_y;
        elements[j].force_z = result_z;
        
        elements[j].vx = elements[j].vx + ((force_x_o+elements[j].force_x)*delta_t/(2*elements[j].m));
        if(abs(elements[j].vx)>domain.max_vel)  elements[j].vx = domain.max_vel*(elements[j].vx/abs(elements[j].vx));
        elements[j].vy = elements[j].vy + ((force_y_o+elements[j].force_y)*delta_t/(2*elements[j].m));
        if(abs(elements[j].vy)>domain.max_vel)  elements[j].vy = domain.max_vel*(elements[j].vy/abs(elements[j].vy));
        elements[j].vz = elements[j].vz + ((force_z_o+elements[j].force_z)*delta_t/(2*elements[j].m));
        if(abs(elements[j].vz)>domain.max_vel)  elements[j].vz = domain.max_vel*(elements[j].vz/abs(elements[j].vz));
                
       
    }
}
    
}

//************************************** kernel to calculate distance **************************************************
template <typename T>
__global__ void main_kernel_dist (particle<T> *elements, T delta_t, T epsilon, T sigma, int num_par, Domain<T> domain, int * cell_arr )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    while( id < num_par)
    {  
           
        
           
            
            elements[id].x = elements[id].x + (delta_t * elements[id].vx) + ( (elements[id].force_x * delta_t * delta_t)/(2*elements[id].m));
            
            if( elements[id].x >= domain.x_max )
            {
                elements[id].x = elements[id].x - (domain.x_max-domain.x_min);
            }
            else if (elements[id].x <= domain.x_min)
            {
                elements[id].x = (domain.x_max-domain.x_min) + elements[id].x;
            }
        
            elements[id].y = elements[id].y + (delta_t * elements[id].vy) + ( (elements[id].force_y * delta_t * delta_t)/(2*elements[id].m));
            if( elements[id].y >= domain.y_max)
            {
                elements[id].y = elements[id].y - (domain.y_max-domain.y_min);
            }
            else if (elements[id].y <= domain.y_min)
            {
                elements[id].y = (domain.y_max-domain.y_min) + elements[id].y;
            }
            elements[id].z = elements[id].z + (delta_t * elements[id].vz) + ( (elements[id].force_z * delta_t * delta_t)/(2*elements[id].m));
            if( elements[id].z >= domain.z_max )
            {
                elements[id].z= elements[id].z - (domain.z_max-domain.z_min);
            }
            else if (elements[id].z <= domain.z_min)
            {
                elements[id].z = (domain.z_max-domain.z_min) + elements[id].z;
            }
                       
            id = id + blockDim.x * gridDim.x;
    }
}
            
//************************************** VTK and OUT file write **************************************************            
template <typename T>
void vtk (particle<T> *elements, int num_par)
{
    string vtk_name = part_out_name_base+"/Particle_parallel_/"+vtk_out_name_base + "_" + to_string(vtk_count) + ".vtk";
    fstream file;
    string type = typeid(elements[1].m).name();
    if(type == "d") type = "double";
    else if(type == "f") type = "float";
	file.open(vtk_name,ios::out);
	file << "# vtk DataFile Version 4.0" <<"\n"
			<< "hesp visualization file" << "\n"
			<< "ASCII" << "\n"
			<< "DATASET UNSTRUCTURED_GRID" << "\n"
			<< "POINTS "<< num_par<<" "<< type << "\n";
			for( int i = 0; i< num_par; ++i)
			{
			  file <<fixed<<setprecision(6)<< elements[i].x <<" "<<fixed<<setprecision(6)<< elements[i].y <<" "<<fixed<<setprecision(6)<< elements[i].z <<"\n";
			}
    file << "CELLS 0 0" << "\n"
			<< "CELL_TYPES 0" << "\n"
			<< "POINT_DATA " <<num_par<< "\n"
			<< "SCALARS m "<< type << "\n"
			<< "LOOKUP_TABLE default" << "\n";
			for( int i = 0; i< num_par; ++i)
			{
			  file <<fixed<<setprecision(6)<< elements[i].m <<"\n";
			}
    file << "VECTORS v "<< type <<"\n";
			for( int i = 0; i< num_par; ++i)
			{
			  file<<fixed<<setprecision(6) << elements[i].vx <<" "<<fixed<<setprecision(6)<< elements[i].vy <<" "<<fixed<<setprecision(6)<< elements[i].vz <<"\n";
			}
	file.close();
	vtk_count++;
}

template <typename T>
void out ( particle<T> *elements, int num_par)
{
    string out_name = part_out_name_base+"/Particle_parallel_/"+part_out_name_base + "_" + to_string(out_count) + ".out";
    fstream file;
	file.open(out_name,ios::out);
	file << num_par <<"\n";
	for(int i = 0; i< num_par; ++i)
	{
        file<<fixed<<setprecision(6)<< elements[i].m << " "<<fixed<<setprecision(6)<< elements[i].x <<" "<<fixed<<setprecision(6)<< elements[i].y <<" "<<fixed<<setprecision(6)<< elements[i].z <<" "<<fixed<<setprecision(6)<< elements[i].vx<<" "<<fixed<<setprecision(6)<< elements[i].vy << " "<<fixed<<setprecision(6)<< elements[i].vz<< endl;
    }
	file.close();
	out_count++;

}


//************************************** Main  **************************************************


int main( int argc, char *argv[] )
{
    data_type t = 0;
    int N,i=0,count = 0, *d_cell_arr, *cpu_par, *gpu_par;
    double negative_ratio = 1, positiv_ratio = 1;
    bool  file_val = true;
    string::size_type sz; 
    string testline, name;
    input_para(argv[1]);
    
    assert((domain.x_max-domain.x_min)/domain.x_n >= domain.r_cut && (domain.y_max-domain.y_min)/domain.y_n >= domain.r_cut && (domain.z_max-domain.z_min)/domain.z_n >= domain.r_cut);
    
    
    ifstream Input (part_input_file);
    getline ( Input, testline);
    N = atoi(testline.c_str()) ;
    
    
    particle<data_type> *elements;
    checkError(cudaMallocManaged(&elements,N*sizeof(particle<data_type>))); // create object for particle structure in device
    checkError(cudaMallocManaged((void**)&cpu_par,N*sizeof(int))); 
    checkError(cudaMallocManaged((void**)&gpu_par,N*sizeof(int))); 
  
    //************************************** Read input values **************************************************
    
    while( !Input.eof() && file_val )
    {
            
            file_val = getline ( Input, testline,' ');
            if(file_val == false )
                break;
            elements[i].m = stod(testline, &sz) ;
            getline ( Input, testline,' ');
            elements[i].x = stod(testline, &sz) ;
            if( elements[i].x > domain.x_max )
            {
                elements[i].x = elements[i].x - (domain.x_max-domain.x_min);
            }
            else if (elements[i].x < domain.x_min)
            {
                elements[i].x = (domain.x_max-domain.x_min) + elements[i].x;
            }
            getline ( Input, testline,' ');
            elements[i].y = stod(testline, &sz) ;
            if( elements[i].y > domain.y_max)
            {
                elements[i].y = elements[i].y - (domain.y_max-domain.y_min);
            }
            else if (elements[i].y < domain.y_min)
            {
                elements[i].y = (domain.y_max-domain.y_min) + elements[i].y;
            }
            getline ( Input, testline,' ');
            elements[i].z = stod(testline, &sz) ;
            if( elements[i].z > domain.z_max )
            {
                elements[i].z = elements[i].z - (domain.z_max-domain.z_min);
            }
            else if (elements[i].z < domain.z_min)
            {
                elements[i].z = (domain.z_max-domain.z_min) + elements[i].z;
            }
            getline ( Input, testline,' ');
            elements[i].vx = stod(testline, &sz) ;
            getline ( Input, testline,' ');
            elements[i].vy = stod(testline, &sz) ;
            getline ( Input, testline,'\n');
            elements[i].vz = stod(testline, &sz) ;
            elements[i].force_x = 0;
            elements[i].force_y = 0;
            elements[i].force_z = 0;
            elements[i].cell_num = i;
            cpu_par[i] = -1;
            gpu_par[i] = -1;
            i++;
    }
    
    
    domain.block_size = N/domain.thread_size +1;
    
    auto start = std::chrono::system_clock::now();
    
    
    
    
    checkError(cudaMallocManaged((void**)&d_cell_arr,domain.cell_size*sizeof(int))); // // cell array initializatio in device
    init_cell<data_type><<<domain.block_size,domain.thread_size>>>(domain, d_cell_arr); // initializing cell array with -1
    
    
    update_cell<data_type><<<domain.block_size,domain.thread_size>>>(elements,N,domain, d_cell_arr, cpu_par, gpu_par); // update cell array with particle pisition
    checkError(cudaPeekAtLastError());
    checkError(cudaDeviceSynchronize());
    
     
    init_force<data_type><<<domain.block_size,domain.thread_size>>>(elements,epsilon,sigma, N, domain, d_cell_arr); // calculate initial force
    checkError(cudaPeekAtLastError());
    checkError(cudaDeviceSynchronize());
    
    string result_name = part_out_name_base+"/Particle_parallel_/result_time.txt";
    fstream file1;
    file1.open(result_name,std::ios_base::app);
    
  while(count <= int(std::round(time_end/timestep_length)))
    {
    
    
     // ************************ write out vtk and out **********************
        /*if((count%int(vtk_out_freq)) == 0 )
        {
            vtk<data_type>(&elements[0],N);
        }
        
        if( (count%int(part_out_freq)) == 0 )
        {
            out<data_type>(&elements[0],N);
        }
      */
      // ************************ Calculate distance **********************
      
        main_kernel_dist<data_type><<<domain.block_size,domain.thread_size>>>(elements,timestep_length,epsilon,sigma,N,domain,d_cell_arr);
        checkError(cudaPeekAtLastError());
        checkError(cudaDeviceSynchronize());
        
        init_particle<data_type><<<domain.block_size,domain.thread_size>>>(elements,N,cpu_par,gpu_par);
        init_cell<data_type><<<domain.block_size,domain.thread_size>>>(domain, d_cell_arr);
        checkError(cudaPeekAtLastError());
        checkError(cudaDeviceSynchronize());
        
    
        
        update_cell<data_type><<<domain.block_size,domain.thread_size>>>(elements,N,domain, d_cell_arr, cpu_par, gpu_par); // update cell array with particle pisition
        checkError(cudaPeekAtLastError());
        checkError(cudaDeviceSynchronize());
        
      // ************************ calculate force **********************  
        
        
        cudaEvent_t start, stop;
        checkError(cudaEventCreate(&start));
        checkError(cudaEventCreate(&stop));
        
        checkError(cudaEventRecord(start));
        
        main_kernel_force<data_type><<<domain.block_size,domain.thread_size>>>(elements,timestep_length,epsilon,sigma,N,domain,d_cell_arr, gpu_par);
        checkError(cudaEventRecord(stop));
        
        
        auto start_cpu = std::chrono::system_clock::now();
        main_host_force<data_type>(elements,timestep_length,epsilon,sigma,N,domain,d_cell_arr, cpu_par);
        auto end_cpu = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds_cpu = end_cpu-start_cpu;
        
        
        
        checkError(cudaEventSynchronize(stop));
        checkError(cudaPeekAtLastError());
        //checkError(cudaDeviceSynchronize());
        
        float milliseconds = 0;
        checkError(cudaEventElapsedTime(&milliseconds, start, stop));
         
        
        
        float time_ratio = (milliseconds/(elapsed_seconds_cpu.count()*1000));
        
        if(time_ratio<1)
           {
                negative_ratio = 1;
                positiv_ratio = positiv_ratio *1.2;
                domain.cpu_x = int(domain.cpu_x/positiv_ratio);
                if( domain.cpu_x == 0) domain.cpu_x = 1;
            }
        else if(time_ratio>1.5)
           {
                if( domain.cpu_x == 0) domain.cpu_x = 1;
                positiv_ratio = 1;
                negative_ratio = negative_ratio*1.2;
                domain.cpu_x = int(domain.cpu_x*negative_ratio);
                //if(domain.cpu_x > domain.x_n-1) domain.cpu_x = domain.x_n-1;
            }
        else
        {   
            positiv_ratio = 1;
            negative_ratio = 1;
        }
                
        //if(count<500)  
        //cout<<count<<"\tGpu: "<<milliseconds<<"\tCPU :"<<elapsed_seconds_cpu.count()*1000<< "\t cpu_x :"<<domain.cpu_x<<endl;
        file1<<count<<"\tGpu: "<<milliseconds<<"\tCPU :"<<elapsed_seconds_cpu.count()*1000<< "\t cpu_x :"<<domain.cpu_x<<endl;
        
        //cout<<count<<endl;
        t = t+timestep_length;
        count++;
    }
     
    /*for(int i =0; i<N; ++i)
    {
        cout<<i<<"  "<<cpu_par[i]<<"\t"<<gpu_par[i]<<"\t";
        if(i%5 == 0)
            cout<<endl;
    }*/
   
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "elapsed time " << elapsed_seconds.count() << "s\n";
    file1<<"elapsed time " << elapsed_seconds.count() << "s\n";
    string vtk_name = part_out_name_base+"/Results"+ ".txt";
    fstream file;
    file.open(vtk_name,std::ios_base::app);
    file<<"unified Fixed_block : "<<elapsed_seconds.count() << "s\n";
    file.close();
    file1.close();
   
}
