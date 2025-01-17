import type { BaseNode } from "@llamaflowjs/core/schema";
import type { BaseDocumentStore } from "../../storage/docStore/types.js";
import type { TransformComponent } from "../types.js";

/**
 * Handle doc store duplicates by checking all hashes.
 */
export class DuplicatesStrategy implements TransformComponent {
  private docStore: BaseDocumentStore;

  constructor(docStore: BaseDocumentStore) {
    this.docStore = docStore;
  }

  async transform(nodes: BaseNode[]): Promise<BaseNode[]> {
    const hashes = await this.docStore.getAllDocumentHashes();
    const currentHashes = new Set<string>();
    const nodesToRun: BaseNode[] = [];

    for (const node of nodes) {
      if (!(node.hash in hashes) && !currentHashes.has(node.hash)) {
        await this.docStore.setDocumentHash(node.id_, node.hash);
        nodesToRun.push(node);
        currentHashes.add(node.hash);
      }
    }

    await this.docStore.addDocuments(nodesToRun, true);

    return nodesToRun;
  }
}
