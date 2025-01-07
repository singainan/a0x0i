#pragma once
#include "CommonMacros.h"
SUPPRESS_WARNINGS_START
SUPPRESS_STL_WARNINGS
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"

SUPPRESS_WARNINGS_END

#include <memory>

namespace opml
{

std::unique_ptr<mlir::Pass> createOptimlirShardingPass();
std::unique_ptr<mlir::Pass> createOptimlirShardedSpmdPass();

#define GEN_PASS_REGISTRATION
#include "Optimlir/OptimlirPasses.h.inc"

}    // namespace opml
