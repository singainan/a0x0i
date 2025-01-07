#ifndef OPTIMLIR_DIALECT
#define OPTIMLIR_DIALECT

include "mlir/IR/OpBase.td"

def Optimlir_Dialect : Dialect {
    let name = "optimlir";
    let summary = "A optimlir out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::opml";
}

class Optimlir_Op<string mnemonic, list<Trait> traits = []> :
        Op<Optimlir_Dialect, mnemonic, traits>;

#endif // OPTIMLIR_DIALECT
