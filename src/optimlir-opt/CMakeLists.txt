get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIROptimlir
        TorchMLIRInitAll
        TorchMLIRTorchDialect
        TorchMLIRTorchPasses
        )
add_llvm_executable(optimlir-opt optimlir-opt.cpp)

llvm_update_compile_flags(optimlir-opt)
target_link_libraries(optimlir-opt PRIVATE ${LIBS})
