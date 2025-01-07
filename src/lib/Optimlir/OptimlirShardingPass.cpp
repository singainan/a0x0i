#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#define GEN_PASS_CLASSES
#include "Optimlir/OptimlirPasses.h"

#include <iostream>

namespace opml
{

static void injectSharding(mlir::Operation* op, mlir::Value input, llvm::StringRef clusterName)
{
    // mlir::MLIRContext* context = op->getContext();
    // mlir::OpBuilder    builder(context);
    // mlir::Value        shardedInput = builder.create<mlir::mesh::ShardOp>(op->getLoc(), input);
    //
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto rank   = tensor.getRank();

    mlir::SmallVector<mlir::SmallVector<mlir::mesh::MeshAxis>> splitAxes(rank);
    for (int i = 0; i < rank - 1; i++) { splitAxes[i] = mlir::SmallVector<mlir::mesh::MeshAxis>(0); }
    splitAxes[rank - 1] = mlir::SmallVector<mlir::mesh::MeshAxis>(1, mlir::mesh::MeshAxis{0});
    mlir::SmallVector<mlir::mesh::MeshAxis> partialAxes;

    mlir::IRRewriter rewriter(op->getContext());
    rewriter.setInsertionPointAfter(op);
    auto shardOp = rewriter.create<mlir::mesh::ShardOp>(op->getLoc(), input);
    auto attr    = mlir::mesh::MeshShardingAttr::get(
        op->getContext(), ::mlir::FlatSymbolRefAttr::get(op->getContext(), clusterName), splitAxes, partialAxes, mlir::mesh::Partial::Sum);
    shardOp.setShardAttr(attr);
    auto shardedInput = shardOp->getResult(0);
    op->getResult(0).replaceAllUsesExcept(shardedInput, shardOp);
}

static bool compareShardOp(mlir::mesh::ShardOp& op1, mlir::mesh::ShardOp& op2)
{
    auto shardAttr1 = op1.getShardAttr();
    auto shardAttr2 = op2.getShardAttr();
    if (shardAttr1.getCluster() != shardAttr2.getCluster()) return false;
    if (shardAttr1.getPartialAxes() != shardAttr2.getPartialAxes()) return false;
    if (shardAttr1.getSplitAxes() != shardAttr2.getSplitAxes()) return false;
    if (shardAttr1.getPartialAxes() != shardAttr2.getPartialAxes()) return false;
    return true;
}

static mlir::mesh::ShardOp* getInputShardOp(mlir::Operation* op)
{
    auto                 opcount = op->getNumOperands();
    mlir::mesh::ShardOp* foundOp = nullptr;

    for (unsigned i = 0; i < opcount; i++)
    {
        auto opnd     = op->getOperand(i);
        auto parentop = opnd.getDefiningOp();
        if (parentop == nullptr) continue;
        auto shardOp = mlir::dyn_cast<mlir::mesh::ShardOp>(parentop);
        if (!shardOp) continue;
        // auto shardAttr = shardOp.getShardAttr();
        if (foundOp != nullptr && !compareShardOp(shardOp, *foundOp))
        {
            return nullptr;
            // throw std::runtime_error("Multiple shard ops with different attributes");
        }
        foundOp = &shardOp;
    }

    return foundOp;
}

static mlir::mesh::ShardOp* getOutputShardOp(mlir::Operation* op)
{
    auto                 opcount = op->getNumResults();
    mlir::mesh::ShardOp* foundOp = nullptr;

    for (unsigned i = 0; i < opcount; i++)
    {
        auto             opnd   = op->getResult(i);
        mlir::Operation* userOp = nullptr;
        for (auto user : opnd.getUsers())
        {
            if (userOp != nullptr) return nullptr;
            userOp       = user;
            auto shardOp = mlir::dyn_cast<mlir::mesh::ShardOp>(user);
            if (!shardOp) continue;
            foundOp = &shardOp;
        }
    }

    return foundOp;
}

struct TosaShardingHandler
{
    mlir::Operation*       tosaOp;
    mlir::mesh::ClusterOp* clusterOp;
    mlir::mesh::ShardOp*   shardOp = nullptr;
    mlir::StringAttr       tosaOpName;

    TosaShardingHandler(mlir::Operation* tosaOp, mlir::mesh::ClusterOp* clusterOp) : tosaOp(tosaOp), clusterOp(clusterOp)
    {
        tosaOpName = tosaOp->getName().getIdentifier();
    }

    bool _HandleTosaReshape()
    {
        auto op = mlir::dyn_cast<mlir::tosa::ReshapeOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    bool _HandleGatherOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::GatherOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    template <typename TElemWiseOp> bool _HandleElemwiseOp()
    {
        auto op = mlir::dyn_cast<TElemWiseOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    template <typename T1DReduceOp> bool _Handle1DReduceOp()
    {
        auto op = mlir::dyn_cast<T1DReduceOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    bool _HandleMatmulOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::MatMulOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    bool _HandleTosaConst()
    {
        auto op = mlir::dyn_cast<mlir::tosa::ConstOp>(tosaOp);
        if (!op) return false;
        auto rslt = op->getOpResult(0);
        // cast to tensor
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(rslt.getType());
        if (!tensor) return false;
        if (!tensor.hasStaticShape())
        {
            std::cout << "Dynamic shape not supported" << std::endl;
            return false;
        }
        // get shape
        auto    shape       = tensor.getShape();
        int64_t numElements = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
        if (numElements > int64_t{16} * 1024 * 1024)
        {
            // Inject sharding
            injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        }
        return true;
    }

    bool _HandleTransposeOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::TransposeOp>(tosaOp);
        if (!op) return false;
        injectSharding(op, op->getOpResult(0), clusterOp->getSymName());
        return true;
    }

    bool _HandleUnknownOp()
    {
        if (tosaOpName.str().starts_with("tosa."))
        {
            // TODO: Add more ops
            std::cout << "Unhandled tosa op: " << tosaOpName.str() << std::endl;
        }
        return false;
    }

    bool operator()()
    {
        if (_HandleTosaConst()) return true;
        shardOp = getInputShardOp(tosaOp);
        if (!shardOp) return false;

        // View change op
        // mesh.shard => tosa.transpose => mesh.shard
        // mesh.shard => reshape => mesh.shard

        // mesh.shard => gather => mesh.shard
        // Elem-wise
        // mesh.shard => add,mul,sub,rsqrt,exp,sigmoid, reciprocal => mesh.shard
        // Reduction
        // mesh.shard => tosa.reduce_sum, reduce_max,  => mesh.shard
        // mesh.shard => tosa.matmul => mesh.shard

        return _HandleTosaReshape() || _HandleTransposeOp()
               || _HandleGatherOp()
               // Elemwise ops
               || _HandleElemwiseOp<mlir::tosa::AddOp>() || _HandleElemwiseOp<mlir::tosa::MulOp>()
               || _HandleElemwiseOp<mlir::tosa::ReciprocalOp>() || _HandleElemwiseOp<mlir::tosa::RsqrtOp>()
               || _HandleElemwiseOp<mlir::tosa::SigmoidOp>() || _HandleElemwiseOp<mlir::tosa::SubOp>()
               || _HandleElemwiseOp<mlir::tosa::ExpOp>()
               // 1D Reduce ops
               || _Handle1DReduceOp<mlir::tosa::ReduceSumOp>()
               || _Handle1DReduceOp<mlir::tosa::ReduceMaxOp>()
               // 2d Reduce Ops
               || _HandleMatmulOp()
               // unhandled ops
               || _HandleUnknownOp();
    }
};

struct TosaSpmdHandler
{
    mlir::Operation*       tosaOp;
    mlir::mesh::ClusterOp* clusterOp;
    mlir::StringAttr       tosaOpName;
    mlir::mesh::ShardOp*   inputShardOp  = nullptr;
    mlir::mesh::ShardOp*   outputShardOp = nullptr;

    TosaSpmdHandler(mlir::Operation* tosaOp, mlir::mesh::ClusterOp* clusterOp) : tosaOp(tosaOp), clusterOp(clusterOp)
    {
        tosaOpName = tosaOp->getName().getIdentifier();
    }

    bool _HandleTosaConst()
    {
        auto op = mlir::dyn_cast<mlir::tosa::ConstOp>(tosaOp);
        if (!op) return false;
        // Replace sizes with sharded sizes
        return true;
    }

    bool _HandleTosaReshape()
    {
        auto op = mlir::dyn_cast<mlir::tosa::ReshapeOp>(tosaOp);
        if (!op) return false;
        // Might need a collective op here
        return true;
    }

    bool _HandleGatherOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::GatherOp>(tosaOp);
        if (!op) return false;
        // Replace sizes with sharded sizes
        return true;
    }

    template <typename TElemWiseOp> bool _HandleElemwiseOp()
    {
        auto op = mlir::dyn_cast<TElemWiseOp>(tosaOp);
        if (!op) return false;
        // Remove the output sharding
        // Replace io sizes with sharded sizes
        return true;
    }

    template <typename T1DReduceOp> bool _Handle1DReduceOp()
    {
        auto op = mlir::dyn_cast<T1DReduceOp>(tosaOp);
        if (!op) return false;
        // Inject mesh collectives
        return true;
    }

    bool _HandleMatmulOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::MatMulOp>(tosaOp);
        if (!op) return false;
        return true;
    }

    bool _HandleTransposeOp()
    {
        auto op = mlir::dyn_cast<mlir::tosa::TransposeOp>(tosaOp);
        if (!op) return false;
        // Switch the sharding
        return true;
    }

    bool _HandleUnknownOp()
    {
        if (tosaOpName.str().starts_with("tosa."))
        {
            // TODO: Add more ops
            std::cout << "Unhandled tosa op: " << tosaOpName.str() << std::endl;
        }
        return false;
    }

    bool operator()()
    {
        outputShardOp = getOutputShardOp(tosaOp);
        _HandleTosaConst();
        inputShardOp = getInputShardOp(tosaOp);
        if (!inputShardOp || !outputShardOp) return false;

        return _HandleTosaReshape() || _HandleTransposeOp()
               || _HandleGatherOp()
               // Elemwise ops
               || _HandleElemwiseOp<mlir::tosa::AddOp>() || _HandleElemwiseOp<mlir::tosa::MulOp>()
               || _HandleElemwiseOp<mlir::tosa::ReciprocalOp>() || _HandleElemwiseOp<mlir::tosa::RsqrtOp>()
               || _HandleElemwiseOp<mlir::tosa::SigmoidOp>() || _HandleElemwiseOp<mlir::tosa::SubOp>()
               || _HandleElemwiseOp<mlir::tosa::ExpOp>()
               // 1D Reduce ops
               || _Handle1DReduceOp<mlir::tosa::ReduceSumOp>()
               || _Handle1DReduceOp<mlir::tosa::ReduceMaxOp>()
               // 2d Reduce Ops
               || _HandleMatmulOp()
               // unhandled ops
               || _HandleUnknownOp();
    }
};

class OptimlirShardingPass : public OptimlirShardingBase<OptimlirShardingPass>
{

    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::tosa::TosaDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::mesh::MeshDialect>();
        // TorchConversion::getBackendTypeConversionDependentDialects(registry);
    }

    void runOnOperation() override
    {
        mlir::MLIRContext*     context = &getContext();
        mlir::ConversionTarget target(*context);
        target.addLegalDialect<mlir::tosa::TosaDialect, mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();

        auto             op = getOperation();
        mlir::IRRewriter rewriter(op->getContext());
        rewriter.setInsertionPointAfter(op);
        auto clusterOp = rewriter.create<mlir::mesh::ClusterOp>(op->getLoc(), "L0", std::vector<int64_t>{4, 16, 4});

        op->walk([&](mlir::Operation* op) { TosaShardingHandler{op, &clusterOp}(); });
    }
};

class OptimlirShardedSpmd : public OptimlirShardingBase<OptimlirShardedSpmd>
{

    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::tosa::TosaDialect>();
        registry.insert<mlir::tensor::TensorDialect>();
        registry.insert<mlir::arith::ArithDialect>();
        registry.insert<mlir::mesh::MeshDialect>();
        // TorchConversion::getBackendTypeConversionDependentDialects(registry);
    }

    void runOnOperation() override {}
};

std::unique_ptr<mlir::Pass> createOptimlirShardingPass()
{
    return std::make_unique<OptimlirShardingPass>();
}

std::unique_ptr<mlir::Pass> createOptimlirShardedSpmdPass()
{
    return std::make_unique<OptimlirShardedSpmd>();
}

}    // namespace opml

#if 0
    RewritePatternSet patterns(&getContext());
    patterns.add<MaiaMatmulConverter>(&getContext());

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // clean up: remove tranpose(transpose(x))
    RewritePatternSet optPatterns(&getContext());
    optPatterns.add<TransposeTransposePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(optPatterns)))) {
      signalPassFailure();
    }

    // clean up (particularly any dead code created)
    PassManager pm(&getContext());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(func))) {
      signalPassFailure();
    }
    // https://mlir.llvm.org/docs/DialectConversion/
    // https://mlir.llvm.org/docs/PatternRewriter/
#endif