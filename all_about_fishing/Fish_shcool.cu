#include <iostream>
#include <curand.h>
#include <string>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <ctime>
#include <chrono>
//#include <sciplot/sciplot.hpp>
using namespace std::chrono;
//#include "matplotlibcpp.h"

const int tpb=8;
int const n_sector=16*16;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


extern "C" __global__ void Uniform_To_Gauss(double *in,int n){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        double A=sqrt(-2.0*log(in[2*id]));
        double B=2.*M_PI*in[2*id+1];
        in[2*id]=A*cos(B);
        in[2*id+1]=A*sin(B);
    }
}
extern "C" __global__ void Init_0_mod(double *x, double *y, double *theta, double *will,int n,double scale){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<n){
        x[id]-=0.5;
        x[id]*=scale;
        y[id]-=0.5;
        y[id]*=scale;
        will[id]=0;
        theta[id]*=6.283185307179586;


    }
}

extern "C" __global__ void Comp_min_alpha_data(int *d_sector_id,double *d_rr,int *d_sektor_min_id,int n){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;
    if (id_i < n){
        int tt;
        int temp_id;
        int min_id[n_sector];                   // massiv dl9 xraneni9 indeksov blishaishix chastic j dl9 chasticu i (v zadannom sektore) 
        double temp_rr;
        double min_rr[n_sector];                // massiv dl9 xraneni9 kvadratov rassto9nii do blishaishix chastic  (v zadannom sektore) 
        for(int i=0;i<n_sector;i++){            
            min_rr[i]=100000;                   // predpologaets9 cho chasticu tak daleko ne udal9uts9     
            min_id[i]=-1;                       // priznak dl9 pustux sektarov
        }
        // -------------------------------------------
        // Poisk blishaishix chastic v kashdom sektore 
        // -------------------------------------------
        tt=id_i*n;                              // id paru chastic 
        for(int j=0;j<id_i;j++){
            temp_id=d_sector_id[tt];            // id sektora v kotorom okazuvaets9 chastica j (znachei9 ot 0 do n_sector)
            temp_rr=d_rr[tt];                   // rassto9nie meshdu chasticami i and j,   d_rr -- masssiv kvadratov rassto9ni meshdu parami (n x n)
            if(temp_rr<min_rr[temp_id]){
                min_rr[temp_id]=temp_rr;
                min_id[temp_id]=j;              // prisvaivanie id chasticu j, kotora9 naibolee blizka k i v sootvetstvuushem sektore  
                //printf("%d",min_id[temp_id]);
            }
            tt++;
        }
        tt++;                                   // propusk paru i==j
        for(int j=id_i+1;j<n;j++){
            temp_id=d_sector_id[tt];            // id sektora v kotorom okazuvaets9 chastica j (znachei9 ot 0 do n_sector)
            temp_rr=d_rr[tt];                   // rassto9nie meshdu chasticami i and j,   d_rr -- masssiv kvadratov rassto9ni meshdu parami (n x n)
            if(temp_rr<min_rr[temp_id]){
                min_rr[temp_id]=temp_rr;
                min_id[temp_id]=j;              // prisvaivanie id chasticu j, kotora9 naibolee blizka k i v sootvetstvuushem sektore  
                //printf("%d",min_id[temp_id]);
            }
            tt++;
        }
        // -------------------------------------------
        // zapis' v global'nue massivu
        // -------------------------------------------
        tt=id_i*n_sector; 
        for(int i=0;i<n_sector;i++){
            d_sektor_min_id[tt]=min_id[i];
            //if(id_i==8-1) printf("%d %d\n",i,min_id[i]);
            tt++;
        }
    }
}

extern "C" __global__ void Comp_Force(double *x, double*y, double *theta, double *f_x, double *f_y, double *f_theta, double *f_w, double *f_sum_w,int n,double koef2,double *rr,double *alpha,int *sektor_id,double h_alpha){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;
    int id_j = blockIdx.y*blockDim.y+threadIdx.y;
    if ( id_i < n && id_j <n && id_i!=id_j){
        double dx=x[id_j]-x[id_i];
        double dy=y[id_j]-y[id_i];
        double dx2=dx*dx;
        double dy2=dy*dy;
        double r2inv=1.0/(dx2+dy2+0.000000000001);
        double rinv=sqrt(r2inv);

        
        double thetaj=theta[id_j];
        double thetai=theta[id_i];
        double costj=cos(thetaj);
        double sintj=sin(thetaj);
        double costi=cos(thetai);
        double sinti=sin(thetai);
        double cosdr=dx*rinv;
        double sindr=dy*rinv;
        int    id=id_j+id_i*n;
        
        double sinphi=costi*sintj-costj*sinti;
        double w=(1.0+(costi*cosdr+sinti*sindr));
        double t_alpha=acos(dx*rinv);
        if(dy<0) t_alpha=6.283185307179586-t_alpha;
        rr[id]=dx2+dy2;
        alpha[id]=t_alpha;
        sektor_id[id]=__double2int_rd(t_alpha/h_alpha);

        f_sum_w[id]=(-dx*sinti+dy*costi+koef2*sinphi)*w/(1.0+1.0/rinv);

        
        f_w[id]=w;

    }
}
extern "C" __global__ void Update_State(double *x, double*y,double *vx, double*vy, double *theta, double *f_x, double *f_y, double *f_theta, double *f_w, double *f_sum_w,double *d_will, int n,double dt,double koef, double *f_rand, bool *map, int n_sp,double k_sp,double costd,double sintd,double v_sp){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;
    if ( id_i < n){
        double fsum_x=0;
        double fsum_y=0;
        double fsum_theta=0;
        double fsum_w=0;
        double fsum_sum_w=0;
        int    tt=id_i*n;
        double costi=cos(theta[id_i]);
        double sinti=sin(theta[id_i]);
        double sinphi=sinti*costd-costi*sintd;
        double signsinphi=1;
        if(sinphi<0) signsinphi=-1;
        for(int i=0;i<n;i++){
            if(i==id_i){
                tt++;
                continue;
            }

            if(map[tt]==true){
                fsum_w+=f_w[tt];
                fsum_sum_w+=f_sum_w[tt];
                map[tt]=false;

            }
            tt++;

        }

        if(id_i<n_sp){
            x[id_i]+=v_sp*(cos(theta[id_i]))*dt;
            y[id_i]+=v_sp*(sin(theta[id_i]))*dt;
            vx[id_i]=v_sp*cos(theta[id_i]);
            vy[id_i]=v_sp*sin(theta[id_i]);
        }
        else{
            x[id_i]+=(cos(theta[id_i]))*dt;
            y[id_i]+=(sin(theta[id_i]))*dt;
            vx[id_i]=cos(theta[id_i]);
            vy[id_i]=sin(theta[id_i]);
        }
        if(fsum_w>0.0000001){   
            theta[id_i]+=(fsum_sum_w/fsum_w)*dt;
            //printf("theta[%d]+\n",id_i);
        }

        
        theta[id_i]+=koef*f_rand[id_i]*sqrt(dt);

        if(id_i<n_sp) theta[id_i]-=k_sp*signsinphi*sqrt(0.5*(1.0-(costd*costi+sintd*sinti)));



    }
}

extern "C" __global__ void Comp_map(int *d_sektor_min_id, double*d_rr, double *d_alpha, int n,double h_alpha,int *d_ray_closest_id){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;           // indeks chasticu
    int id_j = blockIdx.y*blockDim.y+threadIdx.y;           // indeks lucha 
    if ( id_i < n && id_j<n_sector){                       // 
        double rrmin=10000000.0;
        double t_rr;
        double alpha_ray=h_alpha*id_j;
        int min_id=-1;
        int tt=id_i*n_sector;                               // obxod vsex chastic iz sektorov dl9 id_i chasticu 
        int temp_id;
        int temp_id_pair;
        for(int i=0;i<n_sector;i++){                        // obxod vsex chastic iz sektorov dl9 id_i chasticu 
            temp_id=d_sektor_min_id[tt];                    // id chasticu iz i-togo sektora dl9 id_i chasticu
            if(temp_id!=-1){                                // v sektore est' chastica
                temp_id_pair=id_i*n+temp_id;                    // id paru chastic id_i i id_j v massive n x n
                t_rr=sqrt(d_rr[temp_id_pair])/cos(alpha_ray-d_alpha[temp_id_pair]);

                if(t_rr<rrmin && t_rr>0){                       // moshno sravnivat' kvadratu rassto9nii, oni imeut znak iz-za cosinusa
                    rrmin=t_rr;
                    min_id=temp_id_pair;  
                    //printf("%f",min_id);
                }
            }
            tt++;
        }

        d_ray_closest_id[id_i*n_sector+id_j]=min_id;

    }
}

extern "C" __global__ void Comp_map_2(int *d_ray_closest_id, bool *d_map, int n){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;           // indeks chasticu
    int id_j = blockIdx.y*blockDim.y+threadIdx.y;           // indeks lucha 
    if ( id_i < n && id_j<n){
        int tt=id_i*n_sector;
        //int temp_id;
        int local_id=id_i*n+id_j;
        for(int i=0;i<n_sector;i++){
            if(d_ray_closest_id[tt]==local_id){
                d_map[local_id]=true;
                //d_ray_closest_id[tt]=-1;
            }
            tt++;
        }
    }
}
extern "C" __global__ void Map_sym(bool *d_map,int n){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;           // indeks chasticu
    int id_j = blockIdx.y*blockDim.y+threadIdx.y;           // indeks lucha 
    if ( id_i < n && id_j<id_i){
        int id_1=id_i*n+id_j;
        int id_2=id_j*n+id_i;
        if(d_map[id_1]==true) d_map[id_2]=true;
        if(d_map[id_2]==true) d_map[id_1]=true;
    }
}
extern "C" __global__ void Vor_Clear(int *d_sektor_min_id,int *d_ray_closest_id,int n){
    int id_i = blockIdx.x*blockDim.x+threadIdx.x;           // indeks chasticu
    int id_j = blockIdx.y*blockDim.y+threadIdx.y;           // indeks lucha 
    if ( id_i < n && id_j<n_sector){ 
        int tt=id_i*n_sector+id_j;
        d_sektor_min_id[tt]=0;
        d_ray_closest_id[tt]=0;
    }
}

class System{
    int     n;          // chislo agentov
    double  *x;         // Koordinatu agentov 
    double  *y;         // Koordinatu agentov 
    double  *theta;     // uglu orientacii agentov 
    double  *vx;        // massiv skorostei (unused atm)
    double  *vy;        // massiv skorostei (unused atm)
    bool    *map;
    // GPU
    double  *d_x;       // masssiv koordinat at gpu
    double  *d_y;       // masssiv koordinat at gpu
    double  *d_theta;   // masssiv orientacii at gpu
    double  *d_vx;      // masssiv skorostei at gpu
    double  *d_vy;      // masssiv skorostei at gpu
    double  *d_temp;    // massiv dl9 xraneni9 vremennux dannux, poka ispol'zuets9 tol'ko dl9 rascheta sluchainoi silu
    double  *d_f_x;     // masssvi sil 
    double  *d_f_y;     // masssvi sil 
    double  *d_will;
    
    double  *d_f_theta; // masssiv momentov sil
    double  *d_f_w;           // usrednenie s vesami dl9 sil orientacii
    double  *d_f_sum_w;       // usrednenie s vesami dl9 sil orientacii
    double  *d_r_closest;   
    
    double  *d_rr;              // n x n massiv kvardata rassto9nii meshdu parami chastic 
    int     *d_sector_id;       // n x n id sektora v kotorui popadaet chastica
    double  *d_alpha;           // n x n massiv uglov chastic 
    bool    *d_map;             // n x n massiv dl9 usreneneni po sosed9m 0 -- ne usredn9t' 1 -- usredn9t'
    double  h_alpha;            // uglocoi razmer sektora
    int     *d_sektor_min_id;   // massiv id blishaishix chastic v sektorax dl9 kashdoi chasticu (n x n_sector)
    int     *d_ray_closest_id;    //id blishaiesh chasticu na dannom luche (n x n_sector)
    
    int     bpg_n;
    dim3    dimBlock;           // dl9 rascheta massivov n x n (paru chastic) 
    dim3    dimGrid;            // dl9 rascheta massivov n x n (paru chastic) 
    dim3    dimBlock_map;       // dl9 rascheta massivov n x n_sektor+1 (vse luchi dl9 chastic   dl9 approximacii 9cheek voronogo) 
    dim3    dimGrid_map;        // dl9 rascheta massivov n x n_sektor+1 (vse luchi dl9 chastic   dl9 approximacii 9cheek voronogo) 
    //
    curandGenerator_t gen;
    int n_rand;
    int bpg_n_rand;
    int n_sp;
    double k_sp;

    std::ofstream f_py;
    std::ofstream f_vmd;
    std::ofstream f_vor;
    std::ofstream f_log;
    bool out_py;
    bool out_vmd;
    bool out_vor;
    bool vis;
    double phi_d;
public:
    double  dt;          // time step size 
    double  time;
    int     timestep;
    System(int _n,int seed);
    ~System();
    void Init_0(double scale);                  // scale opredel9et nachal'nuu plotnost' klastera
    void Next_step(double koef2,double koef4,double dt,int _n_sp,double _k_sp,double phi_d,double v_sp);
    void Next_step_dump(double koef2,double koef4,double dt,int step,int _n_sp,double _k_sp,double phi_d,double v_sp);
    void Dump_State();
    void Dump_Vor_Dat(std::ofstream &fout);
    void Dump_Init(bool out_py,bool out_vmd,bool out_vor,std::string &name,bool _vis);

};
System::System(int _n,int seed){
    n=_n;
    x=new double[n];
    y=new double[n];
    theta=new double[n];
    vx=new double[n];
    vy=new double[n];
    map=new bool[n*n];
    h_alpha=2.0*M_PI/n_sector;

    std::cout<<"n="<<n<<"\n";

    
    bpg_n=static_cast<int>((n+tpb-1)/tpb);
    n_rand=static_cast<int>(n/2.0);
    bpg_n_rand=static_cast<int>((n_rand+tpb-1)/tpb);
    dimBlock.x=tpb;
    dimBlock.y=tpb;
    dimBlock.z=1;
    dimGrid.x=static_cast<int>((n+tpb+1)/tpb);
    dimGrid.y=static_cast<int>((n+tpb+1)/tpb);
    dimGrid.z=1;
    
    dimBlock_map.x=tpb;
    dimBlock_map.y=tpb;
    dimBlock_map.z=1;
    dimGrid_map.x=static_cast<int>((n+tpb+1)/tpb);
    dimGrid_map.y=static_cast<int>((n_sector+tpb+1)/tpb);
    dimGrid_map.z=1;
    
    
    cudaMalloc((void**)&d_x,sizeof(double)*n);
    cudaMalloc((void**)&d_y,sizeof(double)*n);
    cudaMalloc((void**)&d_theta,sizeof(double)*n);
    cudaMalloc((void**)&d_will,sizeof(double)*n);
    cudaMalloc((void**)&d_vx,sizeof(double)*n);
    cudaMalloc((void**)&d_vy,sizeof(double)*n);
    cudaMalloc((void**)&d_temp,sizeof(double)*n);
    cudaMalloc((void**)&d_f_x,sizeof(double)*n*n);
    cudaMalloc((void**)&d_f_y,sizeof(double)*n*n);
    cudaMalloc((void**)&d_f_theta,sizeof(double)*n*n);
    cudaMalloc((void**)&d_f_w,sizeof(double)*n*n);
    cudaMalloc((void**)&d_f_sum_w,sizeof(double)*n*n);
    cudaMalloc((void**)&d_r_closest,sizeof(double)*n);
   
    cudaMalloc((void**)&d_rr,sizeof(double)*n*n);
    cudaMalloc((void**)&d_sector_id,sizeof(int)*n*n);
    cudaMalloc((void**)&d_alpha,sizeof(double)*n*n);
    cudaMalloc((void**)&d_map,sizeof(bool)*n*n);  
    cudaMalloc((void**)&d_sektor_min_id,sizeof(int)*n*n_sector);  
    cudaMalloc((void**)&d_ray_closest_id,sizeof(int)*n*(n_sector));  
    
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,seed);
}
System::~System(){
    delete [] x;
    delete [] y;
    delete [] theta;
    delete [] vx;
    delete [] vy;
    delete [] map;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_theta);
    cudaFree(d_temp);
    cudaFree(d_f_x);
    cudaFree(d_f_y);
    cudaFree(d_f_theta);
    cudaFree(d_f_w);
    cudaFree(d_f_sum_w);
    cudaFree(d_r_closest);
    cudaFree(d_will);
    
    cudaFree(d_rr);
    cudaFree(d_sector_id);
    cudaFree(d_alpha);
    cudaFree(d_map);
    cudaFree(d_sektor_min_id);
    cudaFree(d_ray_closest_id);
    f_log.close();
    if(out_py) f_py.close();
    if(out_vmd) f_vmd.close();
    if(out_vor) f_vor.close();
}
void System::Dump_Init(bool _out_py,bool _out_vmd,bool _out_vor,std::string &name,bool _vis){
    out_py=_out_py;
    out_vmd=_out_vmd;
    out_vor=_out_vor;
    vis=_vis;
    std::string  tname;
    tname="dump/log_"+name+".txt";
    f_log.open(tname.c_str());
    if(out_vmd){
        tname="dump/"+name+"_vmd.lammpstrj";
        f_vmd.open(tname.c_str());
    }
    if(out_py){
        tname="dump/"+name+"_py.lammpstrj";
        f_py.open(tname.c_str());
    }
    if(out_vor){
        tname="dump/"+name+"_vor.txt";
        f_vor.open(tname.c_str());
    }
}
void System::Dump_State(){
    cudaMemcpy(x,d_x,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(y,d_y,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vx,d_vx,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vy,d_vy,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(theta,d_theta,n*sizeof(double),cudaMemcpyDeviceToHost);
    double mean_x=0;
    double mean_y=0;
    double mean_px=0;
    double mean_py=0;
    double rrx;
    double rry;
    double erx;
    double ery;
    double rr;
    double mean_erx=0;
    double mean_ery=0;
    double mean_vx=0;
    double mean_vy=0;
    double size=0;
    double M=0;
    for(int i=0;i<n;i++){
        mean_x+=x[i];
        mean_y+=y[i];
        mean_px+=cos(theta[i]);
        mean_py+=sin(theta[i]);
    }
    mean_x/=n;
    mean_y/=n;
    for(int i=0;i<n;i++){
        rrx=x[i]-mean_x;
        rry=y[i]-mean_y;
        rr=sqrt(rrx*rrx+rry*rry);
        if(size<rr) size=rr;
        erx=rrx/rr;
        ery=rry/rr;
        
        M+=erx*sin(theta[i])-ery*cos(theta[i]);
        
        mean_erx+=erx;
        mean_ery+=ery;
    }
    mean_px/=n;
    mean_py/=n;
    mean_erx/=n;
    mean_ery/=n;
    M/=n*(sqrt(mean_erx*mean_erx+mean_ery*mean_ery)*sqrt(mean_px*mean_px+mean_py*mean_py));
    if(out_py){
        f_py<<"ITEM: TIMESTEP\n";
        f_py<<timestep<<"\n";
        f_py<<"ITEM: NUMBER OF ATOMS\n";
        f_py<<n<<"\n";
        f_py<<"ITEM: BOX BOUNDS pp pp pp\n";
        f_py<<-1000.0<<" "<<1000.0<<"\n";
        f_py<<-1000.0<<" "<<1000.0<<"\n";
        f_py<<-1<<" "<<1<<"\n";
        f_py<<"ITEM: ATOMS id x y theta\n";
        for(int i=0;i<n;i++){
            f_py<<i<<" ";
            f_py<<x[i]-mean_x<<" ";
            f_py<<y[i]-mean_y<<" ";
            f_py<<theta[i]<<"\n";
        }
    }
    if(out_vmd){
        f_vmd<<"ITEM: TIMESTEP\n";
        f_vmd<<timestep<<"\n";
        f_vmd<<"ITEM: NUMBER OF ATOMS\n";
        f_vmd<<2*n<<"\n";
        f_vmd<<"ITEM: BOX BOUNDS pp pp pp\n";
        f_vmd<<-1000.0<<" "<<1000.0<<"\n";
        f_vmd<<-1000.0<<" "<<1000.0<<"\n";
        f_vmd<<-1<<" "<<1<<"\n";
        f_vmd<<"ITEM: ATOMS id type x y\n";
        double scale=0.05;
        //std::cout<<n_sp<<"\n";
        for(int i=0;i<n_sp;i++){
            f_vmd<<2*i<<" ";
            f_vmd<<2<<" ";
            f_vmd<<x[i]-mean_x+scale*cos(theta[i])<<" ";
            f_vmd<<y[i]-mean_y+scale*sin(theta[i])<<"\n";
            f_vmd<<2*i+1<<" ";
            f_vmd<<3<<" ";
            f_vmd<<x[i]-mean_x-scale*cos(theta[i])<<" ";
            f_vmd<<y[i]-mean_y-scale*sin(theta[i])<<"\n";
        }
        for(int i=n_sp;i<n;i++){
            f_vmd<<2*i<<" ";
            f_vmd<<0<<" ";
            f_vmd<<x[i]-mean_x+scale*cos(theta[i])<<" ";
            f_vmd<<y[i]-mean_y+scale*sin(theta[i])<<"\n";
            f_vmd<<2*i+1<<" ";
            f_vmd<<1<<" ";
            f_vmd<<x[i]-mean_x-scale*cos(theta[i])<<" ";
            f_vmd<<y[i]-mean_y-scale*sin(theta[i])<<"\n";
        }
    }
    //flog<<"time mean_x mean_y\n";
    
    
    
    f_log<<mean_x<<"\t"<<mean_y<<"\t"<<mean_px<<"\t"<<mean_py<<"\t"<<mean_erx<<"\t"<<mean_ery<<"\t"<<M<<"\t"<<size<<"\n";
}
void System::Init_0(double scale){
    time=0;
    timestep=0;
    curandGenerateUniformDouble(gen, d_x, n);
    curandGenerateUniformDouble(gen, d_y, n);
    curandGenerateUniformDouble(gen, d_theta, n);
    Init_0_mod<<<bpg_n,tpb>>>(d_x,d_y,d_theta,d_will,n,scale);
}
void System::Next_step(double koef2,double koef4,double _dt,int _n_sp,double _k_sp,double phi_d,double v_sp){

    dt=_dt;
    curandGenerateUniformDouble(gen, d_temp, n);                // generaci9 uniform rand chisle
    Uniform_To_Gauss<<<bpg_n_rand,tpb>>>(d_temp,n_rand);        // transform uniform to gausse 
    
    Comp_Force<<<dimGrid,dimBlock>>>(d_x,d_y,d_theta,d_f_x,d_f_y,d_f_theta,d_f_w,d_f_sum_w,n,koef2,d_rr,d_alpha,d_sector_id,h_alpha);    //vucgislenie par sil
    //Test_1_CPU();
    Comp_min_alpha_data<<<bpg_n,tpb>>>(d_sector_id,d_rr,d_sektor_min_id,n);
    //Test_2_CPU();

    Comp_map<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_rr,d_alpha,n,h_alpha,d_ray_closest_id);
    Comp_map_2<<<dimGrid,dimBlock>>>(d_ray_closest_id,d_map,n);
    Map_sym<<<dimGrid,dimBlock>>>(d_map,n);
    //Test_3_CPU();
    
    
    
    n_sp=_n_sp;
    k_sp=_k_sp;
    Update_State<<<bpg_n,tpb>>>(d_x,d_y,d_vx,d_vy,d_theta,d_f_x,d_f_y,d_f_theta,d_f_w,d_f_sum_w,d_will,n,dt,koef4,d_temp,d_map,n_sp,k_sp,cos(phi_d),sin(phi_d),v_sp);
    Vor_Clear<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_ray_closest_id,n);
}
void System::Next_step_dump(double koef2,double koef4,double _dt,int step,int _n_sp,double _k_sp,double phi_d,double v_sp){
    //koef1 = I_f/\pi
    //koef2 = I_\|
    //koef3 =
    //koef4 = I_n
    dt=_dt;
    curandGenerateUniformDouble(gen, d_temp, n);                // generaci9 uniform rand chisle
    Uniform_To_Gauss<<<bpg_n_rand,tpb>>>(d_temp,n_rand);        // transform uniform to gausse 
    
    Comp_Force<<<dimGrid,dimBlock>>>(d_x,d_y,d_theta,d_f_x,d_f_y,d_f_theta,d_f_w,d_f_sum_w,n,koef2,d_rr,d_alpha,d_sector_id,h_alpha);    //vucgislenie par sil
    //Test_1_CPU();
    Comp_min_alpha_data<<<bpg_n,tpb>>>(d_sector_id,d_rr,d_sektor_min_id,n);
    //Test_2_CPU();
    
    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize());
    Comp_map<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_rr,d_alpha,n,h_alpha,d_ray_closest_id);
    Comp_map_2<<<dimGrid,dimBlock>>>(d_ray_closest_id,d_map,n);
    Map_sym<<<dimGrid,dimBlock>>>(d_map,n);
    //Test_3_CPU();
    
    // Dump to FIle
    n_sp=_n_sp;
    k_sp=_k_sp;
    f_log<<step<<"\t"<<step*dt<<"\t";
    Dump_State();
    cudaMemcpy(map,d_map,n*n*sizeof(bool),cudaMemcpyDeviceToHost);
    int tt=0;
    for(int ix=0;ix<n;ix++){
        f_vor<<ix<<" ";
        for(int iy=0;iy<n;iy++){
            if(map[ix*n+iy]) f_vor<<iy<<" ";
            tt++;
        }
        f_vor<<"\n";
    }
    // 

    Update_State<<<bpg_n,tpb>>>(d_x,d_y,d_vx,d_vy,d_theta,d_f_x,d_f_y,d_f_theta,d_f_w,d_f_sum_w,d_will,n,dt,koef4,d_temp,d_map,n_sp,k_sp,cos(phi_d),sin(phi_d),v_sp);
    Vor_Clear<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_ray_closest_id,n);
}

void System::Dump_Vor_Dat(std::ofstream &fout){
    curandGenerateUniformDouble(gen, d_temp, n);                // generaci9 uniform rand chisle
    Uniform_To_Gauss<<<bpg_n_rand,tpb>>>(d_temp,n_rand);        // transform uniform to gausse 
    
    Comp_Force<<<dimGrid,dimBlock>>>(d_x,d_y,d_theta,d_f_x,d_f_y,d_f_theta,d_f_w,d_f_sum_w,n,1,d_rr,d_alpha,d_sector_id,h_alpha);    //vucgislenie par sil
    //Test_1_CPU();
    Comp_min_alpha_data<<<bpg_n,tpb>>>(d_sector_id,d_rr,d_sektor_min_id,n);
    //Test_2_CPU();
    
    Comp_map<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_rr,d_alpha,n,h_alpha,d_ray_closest_id);
    Comp_map_2<<<dimGrid,dimBlock>>>(d_ray_closest_id,d_map,n);
    Map_sym<<<dimGrid,dimBlock>>>(d_map,n);
    //Test_3_CPU();

    cudaMemcpy(x,d_x,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(y,d_y,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(map,d_map,n*n*sizeof(bool),cudaMemcpyDeviceToHost);
    int tt=0;
    for(int ix=0;ix<n;ix++){
        for(int iy=0;iy<n;iy++){
            if(map[ix*n+iy]) fout<<ix<<" "<<iy<<"\n";
            tt++;
        }
    }
    Vor_Clear<<<dimGrid_map,dimBlock_map>>>(d_sektor_min_id,d_ray_closest_id,n);
}


class Timing{
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point now;
    double tnow;
    int nprint;
    int old_step;
    int eta;
    double tps;
    int nmax;
public:
    Timing(int _nmax){
        start=high_resolution_clock::now();
        nprint=1;
        old_step=0;
        nmax=_nmax;
    }
    void Tick(int i){
        // i -- step 
        now = high_resolution_clock::now();
        tnow=std::chrono::duration<double>(now-start).count();
        if(static_cast<int>(tnow/10)==nprint){
            std::cout<<"Time ";
            if(nprint/360<10) std::cout<<"0";
            std::cout<<nprint/360<<":";
            if((nprint/6)%60<10) std::cout<<"0";
            std::cout<<(nprint/6)%60<<":"<<nprint%6<<"0 | Step "<<i<<" / "<<nmax<<" | TPS ";
            tps=0.1*(i-old_step);
            eta=static_cast<int>((nmax-i)/tps);
            std::cout<<tps<<" | ETA ";
            if(eta/3600<10) std::cout<<"0";
            std::cout<<eta/3600<<":";
            if((eta/60)%60<10) std::cout<<"0";
            std::cout<<(eta/60)%60<<":";
            if(eta%60<10) std::cout<<"0";
            std::cout<<eta%60<<"\n";
            old_step=i;
            nprint++;
        }
    }
    void Full_Time(){
        now = high_resolution_clock::now();
        tnow=std::chrono::duration<double>(now-start).count();
        std::cout << tnow << std::endl;
    }
};


int main(){
    int         n=10;         // chislo agentov
    bool        out_vmd=true;
    bool        out_py=false;
    bool        out_vor=false;
    bool        vis=false;
    double      scale=20.0;     // parameter nachal'noi plotnosti sistemu
    double      Ip=6.0;         // parametr modeli vuravnivanie
    double      In=0.2;         // parametr modeli shum
    double      dt=0.01;        // sahg po vremeni
    int         nrelax=20000;   // chislo shagov dl9 relaksacii
    int         nmax=100000;    // kolichestvo shagov
    int         n_dump=10;      // chastota vuvoda
    int         n_sp=0;         // chislo osobux rub
    int         seed=0;
    double      k_sp=0;         // koefficient dl9 osobux rub
    double      v_sp=1.0;       // skorost' osobux rub
    double      omega_sp=0.0;   // uglova9 skorost' vrasheni9 phi_d
    double      phi_d=0;        // peremenna9 dl9 xrarneni9 ugla osobux rub
    std::string  name;          // Nlog_Nfish_size_Ip_In_dt_Nrelax_Nrun_thermo_Nspecial_kFspec_Vspec_kf
    
    std::fstream fin("in.txt");
    fin>>n;fin>>scale;fin>>Ip;fin>>In;fin>>dt;fin>>name;fin>>out_vmd;fin>>out_py;fin>>out_vor;fin>>nrelax;fin>>nmax;fin>>n_dump;fin>>n_sp;fin>>k_sp;fin>>v_sp;fin>>omega_sp;fin>>seed;
    
    
    System system(n,seed);
    system.Init_0(scale);
    Timing tm(nmax);
    system.Dump_Init(out_py,out_vmd,out_vor,name,vis);
    
    std::cout<<"Start\n";
    
    // relaxation
    for(int i=0;i<nrelax;i++){
        if(i%n_dump==0) system.Next_step_dump(Ip,In,dt,i,n_sp,0,phi_d,v_sp);
        else system.Next_step(Ip,In,dt,n_sp,0,phi_d,v_sp);
        tm.Tick(i);
    }  
    //
    for(int i=0;i<nmax;i++){
        if(i%n_dump==0) system.Next_step_dump(Ip,In,dt,i+nrelax,n_sp,k_sp,phi_d,v_sp);
        else system.Next_step(Ip,In,dt,n_sp,k_sp,phi_d,v_sp);
        tm.Tick(i);
        phi_d+=omega_sp*dt;
    }
    std::cout<<"\nStop\n";
    tm.Full_Time();
    fin.close();
    return 0;
};




































