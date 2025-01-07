//===- OptimlirOps.cpp - Optimlir dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "CommonMacros.h"

#include "Optimlir/OptimlirDialect.h"
#include "Optimlir/OptimlirOps.h"

SUPPRESS_WARNINGS_START
SUPPRESS_STL_WARNINGS

#include "mlir/IR/OpImplementation.h"

SUPPRESS_WARNINGS_END

#define GET_OP_CLASSES
#include "Optimlir/OptimlirOps.cpp.inc"
