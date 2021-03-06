/*
  The manning coefficient with variable litter model is an algebraic model with dependencies.
  
  Constant values.

  Authors: Ethan Coon (ecoon@lanl.gov)
*/

#ifndef AMANZI_FLOW_MANNING_LITTER_COEFFICIENT_CONSTANT_MODEL_HH_
#define AMANZI_FLOW_MANNING_LITTER_COEFFICIENT_CONSTANT_MODEL_HH_

#include "manning_coefficient_litter_model.hh"

namespace Amanzi {
namespace Flow {
namespace Relations {

class ManningCoefficientLitterConstantModel : public ManningCoefficientLitterModel {
 public:
  ManningCoefficientLitterConstantModel(Teuchos::ParameterList& plist) {
    n_ = plist.get<double>("manning coefficient [s m^-1/3]"); }
    
  
  double ManningCoefficient(double litter_depth, double ponded_depth) const {
    return n_; }

  double DManningCoefficientDLitterThickness(double litter_depth, double ponded_depth) const {
    return 0; }

  double DManningCoefficientDPondedDepth(double litter_depth, double ponded_depth) const {
    return 0; }

 protected:
  double n_;

  
};

} //namespace
} //namespace
} //namespace

#endif
