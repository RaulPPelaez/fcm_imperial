

#include "Integrator/BDHI/FCM/FCM_kernels.cuh"
#include "uammd.cuh"
#include "Integrator/BDHI/FCM/FCM_impl.cuh"
namespace uammd_fcm{
  using Kernel = uammd::BDHI::FCM_ns::Kernels::Gaussian;
  using KernelTorque = uammd::BDHI::FCM_ns::Kernels::GaussianTorque;
  using FCM = uammd::BDHI::FCM_impl<Kernel, KernelTorque>;

  template<class T> using cached_vector = uammd::BDHI::cached_vector<T>;
  using uammd::real3;
  using uammd::real4;
  using uammd::real;

  auto initializeFCM(real hydrodynamicRadius, real viscosity,
		     real Lx, real Ly, real Lz,
		     real tolerance){

    FCM::Parameters par;
    par.viscosity = viscosity;
    par.box = uammd::Box({Lx, Ly, Lz});
    par.hydrodynamicRadius = hydrodynamicRadius;
    par.tolerance = tolerance;
    auto fcm = std::make_shared<FCM>(par);
    return fcm;
  }

  namespace detail{

    struct InterleavedPermute{
      int instance;
      InterleavedPermute(int instance):instance(instance){}

      __host__ __device__ auto operator()(int i){
	return 2*i+instance;
      }
    };
  }

  template<class Iter>
  auto make_interleaved_iterator(Iter it, int instance){
    auto cit = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
					       detail::InterleavedPermute(instance));
    return thrust::make_permutation_iterator(it, cit);
  }

  std::pair<cached_vector<real3>, cached_vector<real3>>
  computeHydrodynamicDisplacements(std::shared_ptr<FCM> fcm,
				   real* i_pos, real* i_forceTorque,
				   int i_numberParticles,
				   real* o_pos, int o_numberParticles,
				   cudaStream_t st = 0){
    real temperature = 0;
    real prefactor = 0;
    auto i_ft = (real3*)i_forceTorque;

    auto i_force = make_interleaved_iterator(i_ft, 0);
    auto i_torque = make_interleaved_iterator(i_ft, 1);
    return fcm->computeHydrodynamicDisplacements((real3*)i_pos, i_force, i_torque, i_numberParticles,
						 (real3*)o_pos, o_numberParticles,
						 temperature, prefactor, st);
  }

}
