


#include "fcm_interface.cuh"


using real = uammd_fcm::real;

int main(){
  //Initialize
  real hydrodynamicRadius = 1.0;
  real viscosity = 1/(6*M_PI);
  real Lx,Ly,Lz;
  Lx = Ly = Lz = 128;
  real tolerance = 1e-4;
  auto fcm = uammd_fcm::initializeFCM(hydrodynamicRadius, viscosity, Lx, Ly, Lz, tolerance);

  //One single particle at the origin
  int i_numberParticles = 1;
  thrust::device_vector<real> i_pos(3*i_numberParticles);
  thrust::fill(i_pos.begin(), i_pos.end(), 0);
  int o_numberParticles = 1;
  thrust::device_vector<real> o_pos(3*o_numberParticles);
  thrust::fill(o_pos.begin(), o_pos.end(), 0);
  thrust::device_vector<real> forceTorque(2*3*i_numberParticles);
  thrust::fill(forceTorque.begin(), forceTorque.end(), 0);

  forceTorque[0] = 1; //A force on the X direction


  auto i_pos_ptr = thrust::raw_pointer_cast(i_pos.data());
  auto o_pos_ptr = thrust::raw_pointer_cast(o_pos.data());
  auto ft_ptr = thrust::raw_pointer_cast(forceTorque.data());
  auto result = uammd_fcm::computeHydrodynamicDisplacements(fcm,
							    i_pos_ptr, ft_ptr, i_numberParticles,
							    o_pos_ptr, o_numberParticles);
  thrust::host_vector<uammd::real> h_result(result.begin(), result.end());
  //Linear displacement for the first particle in the x direction
  std::cerr<<h_result[0]<<std::endl;

  return 0;
}
