add_mlir_dialect_library(MLIROptimlir
                OptimlirDialect.cpp
                OptimlirOps.cpp
                OptimlirShardingPass.cpp
        ADDITIONAL_HEADER_DIRS
                ../include/Optimlir

        DEPENDS
                MLIROptimlirOpsIncGen
                MLIROptimlirPassesIncGen
        
	LINK_LIBS PUBLIC
	        MLIRIR
                MLIRInferTypeOpInterface
                TorchMLIRInitAll
                TorchMLIRTorchDialect
                TorchMLIRTorchPasses
	)
