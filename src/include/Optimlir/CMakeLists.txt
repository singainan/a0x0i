add_mlir_dialect(OptimlirOps optimlir)
add_mlir_doc(OptimlirDialect OptimlirDialect Optimlir/ -gen-dialect-doc)
add_mlir_doc(OptimlirOps OptimlirOps Optimlir/ -gen-op-doc)


set(LLVM_TARGET_DEFINITIONS OptimlirPasses.td)
mlir_tablegen(OptimlirPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIROptimlirPassesIncGen)
add_mlir_doc(OptimlirPasses OptimlirPasses Optimlir/ -gen-pass-doc)
