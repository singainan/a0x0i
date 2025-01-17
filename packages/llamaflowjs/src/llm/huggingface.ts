import { HfInference } from "@huggingface/inference";
import "@llamaflowjs/core/llms";
import {
  BaseLLM,
  type ChatMessage,
  type ChatResponse,
  type ChatResponseChunk,
  type LLMChatParamsNonStreaming,
  type LLMChatParamsStreaming,
  type LLMMetadata,
  type ToolCallLLMMessageOptions,
} from "@llamaflowjs/core/llms";
import { streamConverter, wrapLLMEvent } from "@llamaflowjs/core/utils";
import type {
  PreTrainedModel,
  PreTrainedTokenizer,
  Tensor,
} from "@xenova/transformers";
import { lazyLoadTransformers } from "../internal/deps/transformers.js";

// TODO workaround issue with @huggingface/inference@2.7.0
interface HfInferenceOptions {
  /**
   * (Default: true) Boolean. If a request 503s and wait_for_model is set to false, the request will be retried with the same parameters but with wait_for_model set to true.
   */
  retry_on_error?: boolean;
  /**
   * (Default: true). Boolean. There is a cache layer on Inference API (serverless) to speedup requests we have already seen. Most models can use those results as is as models are deterministic (meaning the results will be the same anyway). However if you use a non deterministic model, you can set this parameter to prevent the caching mechanism from being used resulting in a real new query.
   */
  use_cache?: boolean;
  /**
   * (Default: false). Boolean. Do not load the model if it's not already available.
   */
  dont_load_model?: boolean;
  /**
   * (Default: false). Boolean to use GPU instead of CPU for inference (requires Startup plan at least).
   */
  use_gpu?: boolean;
  /**
   * (Default: false) Boolean. If the model is not ready, wait for it instead of receiving 503. It limits the number of requests required to get your inference done. It is advised to only set this flag to true after receiving a 503 error as it will limit hanging in your application to known places.
   */
  wait_for_model?: boolean;
  /**
   * Custom fetch function to use instead of the default one, for example to use a proxy or edit headers.
   */
  fetch?: typeof fetch;
  /**
   * Abort Controller signal to use for request interruption.
   */
  signal?: AbortSignal;
  /**
   * (Default: "same-origin"). String | Boolean. Credentials to use for the request. If this is a string, it will be passed straight on. If it's a boolean, true will be "include" and false will not send credentials at all.
   */
  includeCredentials?: string | boolean;
}

const DEFAULT_PARAMS = {
  temperature: 0.1,
  topP: 1,
  maxTokens: undefined,
  contextWindow: 3900,
};

export type HFConfig = Partial<typeof DEFAULT_PARAMS> &
  HfInferenceOptions & {
    model: string;
    accessToken: string;
    endpoint?: string;
  };

/**
    Wrapper on the Hugging Face's Inference API.
    API Docs: https://huggingface.co/docs/huggingface.js/inference/README
    List of tasks with models: huggingface.co/api/tasks 
    
    Note that Conversational API is not yet supported by the Inference API.
    They recommend using the text generation API instead.
    See: https://github.com/huggingface/huggingface.js/issues/586#issuecomment-2024059308
 */
export class HuggingFaceInferenceAPI extends BaseLLM {
  model: string;
  temperature: number;
  topP: number;
  maxTokens?: number;
  contextWindow: number;
  hf: HfInference;

  constructor(init: HFConfig) {
    super();
    const {
      model,
      temperature,
      topP,
      maxTokens,
      contextWindow,
      accessToken,
      endpoint,
      ...hfInferenceOpts
    } = init;
    this.hf = new HfInference(accessToken, hfInferenceOpts);
    this.model = model;
    this.temperature = temperature ?? DEFAULT_PARAMS.temperature;
    this.topP = topP ?? DEFAULT_PARAMS.topP;
    this.maxTokens = maxTokens ?? DEFAULT_PARAMS.maxTokens;
    this.contextWindow = contextWindow ?? DEFAULT_PARAMS.contextWindow;
    if (endpoint) this.hf.endpoint(endpoint);
  }

  get metadata(): LLMMetadata {
    return {
      model: this.model,
      temperature: this.temperature,
      topP: this.topP,
      maxTokens: this.maxTokens,
      contextWindow: this.contextWindow,
      tokenizer: undefined,
    };
  }

  chat(
    params: LLMChatParamsStreaming,
  ): Promise<AsyncIterable<ChatResponseChunk>>;
  chat(params: LLMChatParamsNonStreaming): Promise<ChatResponse>;
  @wrapLLMEvent
  async chat(
    params: LLMChatParamsStreaming | LLMChatParamsNonStreaming,
  ): Promise<AsyncIterable<ChatResponseChunk> | ChatResponse<object>> {
    if (params.stream) return this.streamChat(params);
    return this.nonStreamChat(params);
  }

  private messagesToPrompt(messages: ChatMessage<ToolCallLLMMessageOptions>[]) {
    let prompt = "";
    for (const message of messages) {
      if (message.role === "system") {
        prompt += `<|system|>\n${message.content}</s>\n`;
      } else if (message.role === "user") {
        prompt += `<|user|>\n${message.content}</s>\n`;
      } else if (message.role === "assistant") {
        prompt += `<|assistant|>\n${message.content}</s>\n`;
      }
    }
    // ensure we start with a system prompt, insert blank if needed
    if (!prompt.startsWith("<|system|>\n")) {
      prompt = "<|system|>\n</s>\n" + prompt;
    }
    // add final assistant prompt
    prompt = prompt + "<|assistant|>\n";
    return prompt;
  }

  protected async nonStreamChat(
    params: LLMChatParamsNonStreaming,
  ): Promise<ChatResponse> {
    const res = await this.hf.textGeneration({
      model: this.model,
      inputs: this.messagesToPrompt(params.messages),
      parameters: this.metadata,
    });
    return {
      raw: res,
      message: {
        content: res.generated_text,
        role: "assistant",
      },
    };
  }

  protected async *streamChat(
    params: LLMChatParamsStreaming,
  ): AsyncIterable<ChatResponseChunk> {
    const stream = this.hf.textGenerationStream({
      model: this.model,
      inputs: this.messagesToPrompt(params.messages),
      parameters: this.metadata,
    });
    yield* streamConverter(stream, (chunk: any) => ({
      delta: chunk.token.text,
      raw: chunk,
    }));
  }
}

const DEFAULT_HUGGINGFACE_MODEL = "stabilityai/stablelm-tuned-alpha-3b";

export interface HFLLMConfig {
  modelName?: string;
  tokenizerName?: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  contextWindow?: number;
}

export class HuggingFaceLLM extends BaseLLM {
  modelName: string;
  tokenizerName: string;
  temperature: number;
  topP: number;
  maxTokens?: number;
  contextWindow: number;

  private tokenizer: PreTrainedTokenizer | null = null;
  private model: PreTrainedModel | null = null;

  constructor(init?: HFLLMConfig) {
    super();
    this.modelName = init?.modelName ?? DEFAULT_HUGGINGFACE_MODEL;
    this.tokenizerName = init?.tokenizerName ?? DEFAULT_HUGGINGFACE_MODEL;
    this.temperature = init?.temperature ?? DEFAULT_PARAMS.temperature;
    this.topP = init?.topP ?? DEFAULT_PARAMS.topP;
    this.maxTokens = init?.maxTokens ?? DEFAULT_PARAMS.maxTokens;
    this.contextWindow = init?.contextWindow ?? DEFAULT_PARAMS.contextWindow;
  }

  get metadata(): LLMMetadata {
    return {
      model: this.modelName,
      temperature: this.temperature,
      topP: this.topP,
      maxTokens: this.maxTokens,
      contextWindow: this.contextWindow,
      tokenizer: undefined,
    };
  }

  async getTokenizer() {
    const { AutoTokenizer } = await lazyLoadTransformers();
    if (!this.tokenizer) {
      this.tokenizer = await AutoTokenizer.from_pretrained(this.tokenizerName);
    }
    return this.tokenizer;
  }

  async getModel() {
    const { AutoModelForCausalLM } = await lazyLoadTransformers();
    if (!this.model) {
      this.model = await AutoModelForCausalLM.from_pretrained(this.modelName);
    }
    return this.model;
  }

  chat(
    params: LLMChatParamsStreaming,
  ): Promise<AsyncIterable<ChatResponseChunk>>;
  chat(params: LLMChatParamsNonStreaming): Promise<ChatResponse>;
  @wrapLLMEvent
  async chat(
    params: LLMChatParamsStreaming | LLMChatParamsNonStreaming,
  ): Promise<AsyncIterable<ChatResponseChunk> | ChatResponse<object>> {
    if (params.stream) return this.streamChat(params);
    return this.nonStreamChat(params);
  }

  protected async nonStreamChat(
    params: LLMChatParamsNonStreaming,
  ): Promise<ChatResponse> {
    const tokenizer = await this.getTokenizer();
    const model = await this.getModel();

    const messageInputs = params.messages.map((msg) => ({
      role: msg.role,
      content: msg.content as string,
    }));
    const inputs = tokenizer.apply_chat_template(messageInputs, {
      add_generation_prompt: true,
      ...this.metadata,
    }) as Tensor;

    // TODO: the input for model.generate should be updated when using @xenova/transformers v3
    // We should add `stopping_criteria` also when it's supported in v3
    // See: https://github.com/xenova/transformers.js/blob/3260640b192b3e06a10a1f4dc004b1254fdf1b80/src/models.js#L1248C9-L1248C27
    const outputs = await model.generate(inputs, this.metadata);
    const outputText = tokenizer.batch_decode(outputs, {
      skip_special_tokens: false,
    });

    return {
      raw: outputs,
      message: {
        content: outputText.join(""),
        role: "assistant",
      },
    };
  }

  protected async *streamChat(
    params: LLMChatParamsStreaming,
  ): AsyncIterable<ChatResponseChunk> {
    // @xenova/transformers v2 doesn't support streaming generation yet
    // they are working on it in v3
    // See: https://github.com/xenova/transformers.js/blob/3260640b192b3e06a10a1f4dc004b1254fdf1b80/src/models.js#L1249
    throw new Error("Method not implemented.");
  }
}
