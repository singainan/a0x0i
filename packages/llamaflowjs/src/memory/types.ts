import type { ChatMessage } from "@llamaflowjs/core/llms";

export interface BaseMemory<AdditionalMessageOptions extends object = object> {
  tokenLimit: number;
  get(...args: unknown[]): ChatMessage<AdditionalMessageOptions>[];
  getAll(): ChatMessage<AdditionalMessageOptions>[];
  put(message: ChatMessage<AdditionalMessageOptions>): void;
  set(messages: ChatMessage<AdditionalMessageOptions>[]): void;
  reset(): void;
}
