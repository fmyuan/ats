/* -*-  mode: c++; indent-tabs-mode: nil -*- */

/*
  ATS

  Simple EOS for compressibility with pressure.

  http://software.lanl.gov/ats/trac

  Authors: Ethan Coon (ecoon@lanl.gov)
*/

#ifndef AMANZI_RELATIONS_EOS_LINEAR_HH_
#define AMANZI_RELATIONS_EOS_LINEAR_HH_

#include "Teuchos_ParameterList.hpp"

#include "Factory.hh"
#include "eos_constant_molar_mass.hh"

namespace Amanzi {
namespace Relations {

// Equation of State model
class EOSLinear : public EOSConstantMolarMass {

public:
  explicit EOSLinear(Teuchos::ParameterList& eos_plist);

  virtual double MassDensity(std::vector<double>& params) override { return rho_ * (1+beta_*std::max(params[1] - 101325., 0.)); }
  virtual double DMassDensityDp(std::vector<double>& params) override { return params[1] > 101325. ? rho_ * beta_ : 0.; }
  virtual double DMassDensityDT(std::vector<double>& params) override { return 0.; }

private:
  virtual void InitializeFromPlist_();

  Teuchos::ParameterList eos_plist_;
  double rho_;
  double beta_;

  static Utils::RegisteredFactory<EOS,EOSLinear> factory_;

};

} // namespace
} // namespace

#endif
