#pragma once
#include "CommonMacros.h"
SUPPRESS_WARNINGS_START
SUPPRESS_STL_WARNINGS
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Optimlir/OptimlirOps.h.inc"

SUPPRESS_WARNINGS_END
