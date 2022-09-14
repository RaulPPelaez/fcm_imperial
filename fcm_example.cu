


#include "fcm_interface.cuh"


using real = uammd_fcm::real;
template<class T>using gpu_container = uammd_fcm::cached_vector<T>;

int main(){
  real hydrodynamicRadius = 1.0;
  real viscosity = 1/(6*M_PI);
  real Lx,Ly,Lz;
  Lx = Ly = Lz = 128;
  real tolerance = 1e-4;
  auto fcm = uammd_fcm::initializeFCM(hydrodynamicRadius, viscosity, Lx, Ly, Lz, tolerance);

  int i_numberParticles = 100;
  gpu_container<real> i_pos(3*i_numberParticles);
  thrust::fill(i_pos.begin(), i_pos.end(), 0);
  int o_numberParticles = 10;
  gpu_container<real> o_pos(3*o_numberParticles);
  thrust::fill(o_pos.begin(), o_pos.end(), 0);
  gpu_container<real> forceTorque(2*3*i_numberParticles);
  thrust::fill(forceTorque.begin(), forceTorque.end(), 0);

  auto i_pos_ptr = thrust::raw_pointer_cast(i_pos.data());
  auto o_pos_ptr = thrust::raw_pointer_cast(o_pos.data());
  auto ft_ptr = thrust::raw_pointer_cast(forceTorque.data());

  auto result = uammd_fcm::computeHydrodynamicDisplacements(fcm,
							    i_pos_ptr, ft_ptr, i_numberParticles,
							    o_pos_ptr, o_numberParticles);

  auto monopoles = result.first;
  auto dipoles = result.second;


  return 0;
}
