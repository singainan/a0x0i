import type { JSONValue } from "@llamaflowjs/core/global";
import type { BaseTool, ToolMetadata } from "@llamaflowjs/core/llms";
import type { JSONSchemaType } from "ajv";

export class FunctionTool<T, R extends JSONValue | Promise<JSONValue>>
  implements BaseTool<T>
{
  constructor(
    private readonly _fn: (input: T) => R,
    private readonly _metadata: ToolMetadata<JSONSchemaType<T>>,
  ) {}

  static from<T>(
    fn: (input: T) => JSONValue | Promise<JSONValue>,
    schema: ToolMetadata<JSONSchemaType<T>>,
  ): FunctionTool<T, JSONValue | Promise<JSONValue>>;
  static from<T, R extends JSONValue | Promise<JSONValue>>(
    fn: (input: T) => R,
    schema: ToolMetadata<JSONSchemaType<T>>,
  ): FunctionTool<T, R> {
    return new FunctionTool(fn, schema);
  }

  get metadata(): BaseTool<T>["metadata"] {
    return this._metadata as BaseTool<T>["metadata"];
  }

  call(input: T) {
    return this._fn(input);
  }
}
