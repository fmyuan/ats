/*
  ATS is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (ecoon@lanl.gov)
*/
//! A subgrid model for determining the area fraction of land, water, and snow within a grid cell with subgrid microtopography.

#include "subgrid_microtopography.hh"
#include "area_fractions_threecomponent_microtopography_evaluator.hh"

namespace Amanzi {
namespace SurfaceBalance {
namespace Relations {

// Constructor from ParameterList
AreaFractionsThreeComponentMicrotopographyEvaluator::AreaFractionsThreeComponentMicrotopographyEvaluator(Teuchos::ParameterList& plist) :
    SecondaryVariableFieldEvaluator(plist)
{
  //
  // NOTE: this evaluator simplifies the situation by assuming constant
  // density.  This make it so that ice and water see the same geometry per
  // unit pressure, which isn't quite true thanks to density differences.
  // However, we hypothesize that these differences, on the surface (unlike in
  // the subsurface) really don't matter much. --etc
  snow_subgrid_transition_ = plist_.get<double>("snow transition height [m]", 0.02);
  min_area_ = plist_.get<double>("minimum fractional area [-]", 1.e-5);
  if (min_area_ < 0.) {
    Errors::Message message("AreaFractionsTwoComponentEvaluator: Minimum fractional area should be >= 0.");
    Exceptions::amanzi_throw(message);
  }

  // get domain names
  domain_ = Keys::getDomain(my_key_);
  domain_snow_ = Keys::readDomainHint(plist_, domain_, "surface", "snow");

  // get dependencies
  delta_max_key_ = Keys::readKey(plist_, domain_, "microtopographic relief", "microtopographic_relief");
  dependencies_.insert(delta_max_key_);

  delta_ex_key_ = Keys::readKey(plist_, domain_, "excluded volume", "excluded_volume");
  dependencies_.insert(delta_ex_key_);

  ponded_depth_key_ = Keys::readKey(plist_, domain_, "ponded depth", "ponded_depth");
  dependencies_.insert(ponded_depth_key_);

  snow_depth_key_ = Keys::readKey(plist_, domain_snow_, "snow depth", "depth");
  dependencies_.insert(snow_depth_key_);

  vol_snow_depth_key_ = Keys::readKey(plist_, domain_snow_, "volumetric snow depth", "volumetric_depth");
  dependencies_.insert(vol_snow_depth_key_);
}


void
AreaFractionsThreeComponentMicrotopographyEvaluator::EvaluateField_(const Teuchos::Ptr<State>& S,
        const Teuchos::Ptr<CompositeVector>& result)
{
  Epetra_MultiVector& res = *result->ViewComponent("cell",false);

  const Epetra_MultiVector& pd = *S->GetFieldData(ponded_depth_key_)->ViewComponent("cell",false);
  const Epetra_MultiVector& sd = *S->GetFieldData(snow_depth_key_)->ViewComponent("cell",false);
  const Epetra_MultiVector& vsd = *S->GetFieldData(vol_snow_depth_key_)->ViewComponent("cell",false);
  const Epetra_MultiVector& del_max = *S->GetFieldData(delta_max_key_)->ViewComponent("cell", false);
  const Epetra_MultiVector& del_ex = *S->GetFieldData(delta_ex_key_)->ViewComponent("cell", false);

  for (int c=0; c!=res.MyLength(); ++c) {
    // calculate area of land
    AMANZI_ASSERT(Flow::Microtopography::validParameters(del_max[0][c], del_ex[0][c]));
    double liquid_water_area = Flow::Microtopography::dVolumetricDepth_dDepth(pd[0][c], del_max[0][c], del_ex[0][c]);
    double wet_area = Flow::Microtopography::dVolumetricDepth_dDepth(pd[0][c] + std::max(sd[0][c],0.0), del_max[0][c], del_ex[0][c]);

    // now partition the wet area into snow and water
    if (vsd[0][c] >= wet_area * snow_subgrid_transition_) {
      res[2][c] = wet_area;
      res[1][c] = 0.;
      res[0][c] = 1 - wet_area;
    } else {
      res[2][c] = vsd[0][c] / snow_subgrid_transition_;

      // how much of the remainder goes to water?
      res[1][c] = std::min(wet_area - res[2][c], liquid_water_area);
      res[0][c] = 1 - res[1][c] - res[2][c];
    }

    // if any area fraction is less than eps, give it to the others
    if (res[0][c] > 0 && res[0][c] < min_area_) {
      if (res[1][c] < min_area_) {
        res[2][c] = 1.;
        res[1][c] = 0.;
        res[0][c] = 0.;
      } else {
        res[1][c] += res[0][c] * res[1][c] / (res[1][c] + res[2][c]);
        res[2][c] += res[0][c] * res[2][c] / (res[1][c] + res[2][c]);
        res[0][c] = 0.;
      }
    } else if (res[1][c] > 0 && res[1][c] < min_area_) {
      if (res[2][c] < min_area_) {
        res[0][c] = 1.;
        res[1][c] = 0.;
        res[2][c] = 0.;
      } else {
        res[0][c] += res[1][c] * res[0][c] / (res[0][c] + res[2][c]);
        res[2][c] += res[1][c] * res[2][c] / (res[0][c] + res[2][c]);
        res[1][c] = 0.;
      }
    } else if (res[2][c] > 0 && res[2][c] < min_area_) {
      res[0][c] += res[2][c] * res[0][c] / (res[0][c] + res[1][c]);
      res[1][c] += res[2][c] * res[1][c] / (res[0][c] + res[1][c]);
      res[2][c] = 0.;
    }

    AMANZI_ASSERT(std::abs(res[0][c] + res[1][c] + res[2][c] - 1.0) < 1.e-6);
    AMANZI_ASSERT(-1.e-10 <= res[0][c] && res[0][c] <= 1.+1.e-10);
    AMANZI_ASSERT(-1.e-10 <= res[1][c] && res[1][c] <= 1.+1.e-10);
    AMANZI_ASSERT(-1.e-10 <= res[2][c] && res[2][c] <= 1.+1.e-10);

    res[0][c] = std::min(std::max(0.,res[0][c]), 1.);
    res[1][c] = std::min(std::max(0.,res[1][c]), 1.);
    res[2][c] = std::min(std::max(0.,res[2][c]), 1.);
  }
}

void
AreaFractionsThreeComponentMicrotopographyEvaluator::EnsureCompatibility(const Teuchos::Ptr<State>& S) {
  // see if we can find a master fac
  auto my_fac = S->RequireField(my_key_, my_key_);
  my_fac->SetMesh(S->GetMesh(domain_))
      ->SetGhosted()
      ->SetComponent("cell", AmanziMesh::CELL, 3);

  // Check plist for vis or checkpointing control.
  bool io_my_key = plist_.get<bool>("visualize", true);
  S->GetField(my_key_, my_key_)->set_io_vis(io_my_key);
  bool checkpoint_my_key = plist_.get<bool>("checkpoint", false);
  S->GetField(my_key_, my_key_)->set_io_checkpoint(checkpoint_my_key);


  for (auto dep_key : dependencies_) {
    auto fac = S->RequireField(dep_key);
    if (Keys::getDomain(dep_key) == domain_snow_) {
      fac->SetMesh(S->GetMesh(domain_snow_))
          ->SetGhosted()
          ->AddComponent("cell", AmanziMesh::CELL, 1);
    } else {
      fac->SetMesh(S->GetMesh(domain_))
          ->SetGhosted()
          ->AddComponent("cell", AmanziMesh::CELL, 1);
    }

    // Recurse into the tree to propagate info to leaves.
    S->RequireFieldEvaluator(dep_key)->EnsureCompatibility(S);
  }
}

} //namespace
} //namespace
} //namespace

