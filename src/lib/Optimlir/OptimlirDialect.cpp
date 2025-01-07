//===- OptimlirDialect.cpp - Optimlir dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Optimlir/OptimlirDialect.h"
#include "Optimlir/OptimlirOps.h"

using namespace mlir;
using namespace opml;

//===----------------------------------------------------------------------===//
// Optimlir dialect.
//===----------------------------------------------------------------------===//

void OptimlirDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "Optimlir/OptimlirOps.cpp.inc"
        >();
}
