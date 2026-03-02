# NVIDIA NIM Explorer

A Gradio app for browsing, chatting with, and comparing NVIDIA NIM models via the `integrate.api.nvidia.com` endpoint.

## Features

**API Key Validation** -- Enter your NVIDIA API key (or set `NVIDIA_API_KEY` env var), validate it, and see the total number of available models.

![API Key](assets/01-api-key.jpg)

**Model Catalog** -- Browse all 185+ NIM models in a searchable table. Chat models are ranked first and highlighted in green. Embedding and other model types are listed below in grey. Click any row to select it for use in the Playground.

![Model Catalog](assets/02-model-catalog.jpg)

**Chat Playground** -- Select a model from the dropdown, configure temperature, max tokens, and top_p, set a system prompt, then chat with streaming responses.

![Chat Playground](assets/03-chat-playground.jpg)

**Compare** -- Send the same prompt to two models in parallel. Each response includes its elapsed time, making it easy to benchmark speed and quality side by side.

![Compare](assets/04-compare.jpg)

**API Inspector** -- View the raw JSON request and response from the last API call. The API key is automatically masked. Useful for debugging or copying payloads into your own code.

![API Inspector](assets/05-api-inspector.jpg)

## Quick Start

```bash
cd nim-explorer
python3 app.py
```

Opens at [http://localhost:7862](http://localhost:7862).

## Dependencies

- `gradio`
- `openai`

## How It Works

The app uses the OpenAI Python client pointed at NVIDIA's NIM endpoint.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-..."
)
```

All chat-capable models serve the standard `/v1/chat/completions` interface with streaming support. The model catalog is fetched from `/v1/models`. Embedding and reranking models are identified by keyword matching and excluded from the chat/compare dropdowns.
