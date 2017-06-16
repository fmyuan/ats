#include "liquid_gas_energy_evaluator.hh"

namespace Amanzi {
namespace Energy {
namespace Relations {

Utils::RegisteredFactory<FieldEvaluator,LiquidGasEnergyEvaluator> LiquidGasEnergyEvaluator::reg_("liquid+gas energy");

} //namespace
} //namespace
} //namespace
