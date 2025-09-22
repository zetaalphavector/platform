# LLM models
This README contains information on running Ollama locally such that it can be used in
the agents sdk.

## Running ollama locally
Ollama can run multiple models and the model specified in the api request is the one that
is used. Ollama will switch the model for you. This only works for models you have pulled
using `ollama pull <model-name>`.

All models listed [here](https://ollama.com/library) are supported. For selecting models
with different quantization, you have to pull the corresponding model, since this is smt
that is part of that specific model.

Ollama itself is a single Golang binary that can run everything. Models that are pulled
are stored in a local cache along with necessary configurations.

There is a [docker image](https://hub.docker.com/r/ollama/ollama) that we can pull and use if
we want to run ollama ourselves. To run it on a GPU you need the Nvidia Container Toolkit. And
configure docker to use the Nvidia toolkit.

To start ollama you can use the command:

```
ollama run <model-name>
```

More detailed documentation on ollama can be found [here](https://github.com/ollama/ollama/tree/main/docs)

There are a number of environment variables that can be set:

- OLLAMA_HOST: where to serve the apis OLLAMA_MAX_LOADED_MODEL: The maximum number of models that can be loaded concurrently provided they fit in available memory. The default is 3 * the number of GPUs or 3 for CPU inference
- OLLAMA_NUM_PARALLEL: The maximum number of parallel requests each model will process at the same time. The default will auto-select either 4 or 1 based on available memory.
- OLLAMA_MAX_QUEUE: The maximum number of requests Ollama will queue when busy before rejecting additional requests. The default is 512
- OLLAMA_FLASH_ATTENTION: Enables flash attention when set to 1 on startup.
- OLLAMA_KV_CACHE_TYPE: Configures the quantization of the k/v cache.

## Using local ollama in the agents sdk
To use a local ollama deployment:

- Set the vendor for the `llm_client_configuration` to `ollama`.
- Make sure that the `name` under the `model_configuration` corresponds with a model supported by your ollama deployment. E.g. if ollama is used to run `deepseek-r1:1.5b` the name in the `model_configuration` must also be `deepseek-r1:1.5b`.
- Set the `openai_api_base` equal to `http://localhost:<port-number>/v1`.
