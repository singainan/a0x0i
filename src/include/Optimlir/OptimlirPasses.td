#ifndef OPTIMLIR_PASSES
#define OPTIMLIR_PASSES

include "mlir/Pass/PassBase.td"

def OptimlirSharding : Pass<"opml-sharding"> {
  let constructor = "::opml::createOptimlirShardingPass()";
  let dependentDialects = [
    "::mlir::tosa::TosaDialect",
  ];
}



#endif //OPTIMLIR_PASSES