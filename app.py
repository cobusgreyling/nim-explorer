"""
NIM Explorer — Gradio App
=========================
Browse, chat with, and compare NVIDIA NIM models via integrate.api.nvidia.com.

Usage:
    cd nim-explorer && python app.py
    # Opens at http://localhost:7862
"""

import json
import math
import os
import re
import tempfile
import threading
import time
import concurrent.futures

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# NVIDIA green theme (reused from nim-clients/app.py)
# ---------------------------------------------------------------------------
NVIDIA_GREEN = "#76b900"
BG_DARK = "#1a1a2e"
BG_CARD = "#2a2a4a"

BASE_URL = "https://integrate.api.nvidia.com/v1"

PROMPT_TEMPLATES = {
    "Default": "You are a helpful assistant.",
    "Summariser": "You are a concise summariser. Respond with clear, brief summaries.",
    "Coder": "You are an expert programmer. Write clean, well-commented code.",
    "Translator": "You are a professional translator. Translate the user's text accurately.",
    "Reviewer": "You are a thorough code reviewer. Identify bugs, suggest improvements.",
}

# ---------------------------------------------------------------------------
# Cost estimator pricing (per 1K tokens)
# ---------------------------------------------------------------------------
NIM_PRICING = {
    "meta/llama": {"prompt": 0.0003, "completion": 0.0006},
    "google/gemma": {"prompt": 0.0002, "completion": 0.0004},
    "mistralai": {"prompt": 0.0003, "completion": 0.0006},
    "microsoft/phi": {"prompt": 0.0001, "completion": 0.0002},
    "nvidia/nemotron": {"prompt": 0.0005, "completion": 0.0010},
    "deepseek-ai": {"prompt": 0.0003, "completion": 0.0006},
    "qwen": {"prompt": 0.0003, "completion": 0.0006},
    "_default": {"prompt": 0.0004, "completion": 0.0008},
}

# ---------------------------------------------------------------------------
# Benchmark prompts for leaderboard
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    {
        "category": "Reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
    },
    {
        "category": "Coding",
        "prompt": "Write a Python function that checks whether a given string is a valid IPv4 address. Include edge cases.",
    },
    {
        "category": "Summarisation",
        "prompt": "Summarise the key differences between TCP and UDP protocols in networking. Be concise but thorough.",
    },
    {
        "category": "Creative",
        "prompt": "Write a short poem (4-8 lines) about artificial intelligence from the perspective of a curious child.",
    },
]

BANNER = f"""
<div style="text-align:center;padding:16px 0 4px 0">
    <span style="font-size:32px;font-weight:bold;letter-spacing:2px;color:white">
        NVIDIA</span>
    <span style="font-size:32px;font-weight:300;letter-spacing:2px;color:{NVIDIA_GREEN}">
        &nbsp;NIM Explorer</span>
    <div style="font-size:14px;color:#888;margin-top:4px">
        Browse, chat with, and compare NIM models
    </div>
</div>
"""

CSS = f"""
.gradio-container {{ background: {BG_DARK} !important; }}
.tab-nav button {{ color: white !important; }}
.tab-nav button.selected {{ border-color: {NVIDIA_GREEN} !important; color: {NVIDIA_GREEN} !important; }}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0fbe0", c100="#daf5b0", c200="#c0ed78",
        c300="#a5e445", c400="#8ed621", c500=NVIDIA_GREEN,
        c600="#649e00", c700="#4f7d00", c800="#3a5c00",
        c900="#253b00", c950="#132000",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f5f5f5", c100="#e0e0e0", c200="#bdbdbd",
        c300="#9e9e9e", c400="#757575", c500="#616161",
        c600="#424242", c700="#2a2a4a", c800="#1a1a2e",
        c900="#121225", c950="#0a0a18",
    ),
).set(
    body_background_fill=BG_DARK,
    body_background_fill_dark=BG_DARK,
    block_background_fill=BG_CARD,
    block_background_fill_dark=BG_CARD,
    block_label_text_color="white",
    block_label_text_color_dark="white",
    block_title_text_color="white",
    block_title_text_color_dark="white",
    body_text_color="white",
    body_text_color_dark="white",
    body_text_color_subdued="#aaa",
    body_text_color_subdued_dark="#aaa",
    button_primary_background_fill=NVIDIA_GREEN,
    button_primary_background_fill_dark=NVIDIA_GREEN,
    button_primary_text_color="#000",
    button_primary_text_color_dark="#000",
    input_background_fill="#2a2a4a",
    input_background_fill_dark="#2a2a4a",
    input_border_color="#444",
    input_border_color_dark="#444",
    border_color_primary="#444",
    border_color_primary_dark="#444",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NON_CHAT_KEYWORDS = [
    "embed", "rerank", "rerankqa", "nv-embed", "vlm-1", "parakeet",
    "canary", "nemo-asr", "whisper", "sdxl", "stable-diffusion",
    "consistory", "cosmos", "nv-clip", "usm", "grounding-dino",
]


def _is_chat_model(model_id):
    """Heuristic: return True if the model likely supports /v1/chat/completions."""
    mid = model_id.lower()
    return not any(kw in mid for kw in NON_CHAT_KEYWORDS)


def _is_embedding_model(model_id):
    """Return True if the model likely supports /v1/embeddings."""
    mid = model_id.lower()
    return any(kw in mid for kw in ["embed", "nv-embed", "nv-clip"])


def _get_client(api_key):
    """Build an OpenAI client pointing at NVIDIA NIM."""
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise gr.Error("No API key provided. Enter your key or set NVIDIA_API_KEY.")
    return OpenAI(base_url=BASE_URL, api_key=key)


def _format_model_choices(model_ids, favourites=None, availability=None, max_label_len=55):
    """Return list of (label, value) tuples with provider badge, star, and availability."""
    favourites = favourites or []
    availability = availability or {}
    choices = []
    for mid in model_ids:
        if "/" in mid:
            provider, name = mid.split("/", 1)
        else:
            provider, name = "nvidia", mid
        label = f"[{provider}] {name}"
        if len(label) > max_label_len:
            label = label[:max_label_len - 1] + "…"
        if mid in availability and not availability[mid]:
            label = f"[✗] {label}"
        if mid in favourites:
            label = f"★ {label}"
        choices.append((label, mid))
    # Sort starred models to top
    if favourites:
        choices.sort(key=lambda c: (0 if c[1] in favourites else 1, c[0]))
    return choices


def _mask_key(key):
    """Mask an API key for display, keeping first 8 and last 4 chars."""
    if len(key) <= 12:
        return "***"
    return key[:8] + "..." + key[-4:]


def _build_inspector_request(method, url, headers, body):
    """Format the request side of the API inspector."""
    masked_headers = dict(headers)
    if "Authorization" in masked_headers:
        masked_headers["Authorization"] = "Bearer " + _mask_key(
            masked_headers["Authorization"].replace("Bearer ", "")
        )
    return json.dumps(
        {"method": method, "url": url, "headers": masked_headers, "body": body},
        indent=2,
    )


def _build_inspector_response(status, body):
    """Format the response side of the API inspector."""
    return json.dumps({"status": status, "body": body}, indent=2)


def _estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimate cost for a request based on model family pricing."""
    rates = NIM_PRICING["_default"]
    for family, family_rates in NIM_PRICING.items():
        if family != "_default" and family in model.lower():
            rates = family_rates
            break
    prompt_cost = (prompt_tokens / 1000) * rates["prompt"]
    completion_cost = (completion_tokens / 1000) * rates["completion"]
    return prompt_cost + completion_cost


def _cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors (pure Python)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Tab 1 — API Key validation
# ---------------------------------------------------------------------------
def validate_key(api_key):
    """Validate the API key by fetching /v1/models."""
    client = _get_client(api_key)
    try:
        models = client.models.list()
        model_list = list(models)
        count = len(model_list)
        chat_ids = sorted(m.id for m in model_list if _is_chat_model(m.id))
        model_ids = chat_ids
        gr.Info(f"Key validated. {count} models total, {len(chat_ids)} chat models.")
        return (
            f"Validated. {count} models found.",
            model_ids,
            _build_inspector_request(
                "GET", f"{BASE_URL}/models",
                {"Authorization": f"Bearer {api_key}"},
                None,
            ),
            _build_inspector_response(200, {"model_count": count}),
        )
    except Exception as exc:
        raise gr.Error(f"Validation failed: {exc}")


# ---------------------------------------------------------------------------
# Tab 2 — Model Catalog
# ---------------------------------------------------------------------------
def fetch_catalog(api_key):
    """Fetch model catalog and return as HTML table + model ID list."""
    client = _get_client(api_key)
    try:
        models = list(client.models.list())
    except Exception as exc:
        raise gr.Error(f"Failed to fetch models: {exc}")

    rows = []
    for m in models:
        is_chat = _is_chat_model(m.id)
        category = "chat" if is_chat else "embedding/other"
        rows.append((m.id, getattr(m, "owned_by", "—"), category, is_chat))
    rows.sort(key=lambda r: (0 if r[3] else 1, r[0]))

    chat_ids = sorted(r[0] for r in rows if r[3])

    # Build HTML table with JS click-to-select and client-side search
    table_rows = ""
    for mid, owner, category, is_chat in rows:
        cat_color = NVIDIA_GREEN if is_chat else "#888"
        escaped_mid = mid.replace("'", "\\'")
        table_rows += (
            f'<tr class="catalog-row" data-model="{mid.lower()}" '
            f'style="border-bottom:1px solid #333;cursor:pointer" '
            f"onclick=\""
            f"var tb=document.querySelector('#selected-model textarea,#selected-model input');"
            f"if(tb){{var ns=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype||window.HTMLTextAreaElement.prototype,'value').set;"
            f"ns.call(tb,'{escaped_mid}');tb.dispatchEvent(new Event('input',{{bubbles:true}}));}}"
            f"this.parentElement.querySelectorAll('tr').forEach(r=>r.style.background='');"
            f"this.style.background='#3a3a5a';"
            f'">'
            f'<td style="padding:8px;color:{NVIDIA_GREEN if is_chat else "#666"};font-family:monospace">{mid}</td>'
            f'<td style="padding:8px;color:#ccc">{owner}</td>'
            f'<td style="padding:8px;color:{cat_color};font-weight:bold;font-size:12px">{category}</td></tr>'
        )

    html = f"""
    <input id="catalog-search" type="text" placeholder="Filter models..."
        oninput="var q=this.value.toLowerCase();this.parentElement.querySelectorAll('.catalog-row').forEach(function(row){{row.style.display=row.getAttribute('data-model').includes(q)?'':'none'}})"
        style="width:100%;padding:10px 12px;margin-bottom:8px;border-radius:6px;
        border:1px solid #444;background:{BG_CARD};color:white;font-size:14px;
        outline:none;box-sizing:border-box" />
    <div style="max-height:500px;overflow-y:auto;border-radius:8px">
    <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="border-bottom:2px solid {NVIDIA_GREEN};color:#aaa">
            <th style="padding:8px;text-align:left">Model ID</th>
            <th style="padding:8px;text-align:left">Owned By</th>
            <th style="padding:8px;text-align:left">Category</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table></div>
    <p style="color:#888;font-size:12px;margin-top:8px">{len(rows)} models total, {len(chat_ids)} chat models. Click a row to select it.</p>
    """
    return html, chat_ids


def probe_models(api_key, model_ids):
    """Probe each chat model with a minimal request to check availability. Yields partial results."""
    if not model_ids:
        raise gr.Error("No models loaded. Validate your API key and refresh the catalog first.")
    client = _get_client(api_key)
    results = {}

    def _probe_one(mid):
        try:
            client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
                timeout=10,
            )
            return mid, True
        except Exception:
            return mid, False

    total = len(model_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_probe_one, mid): mid for mid in model_ids}
        for future in concurrent.futures.as_completed(futures):
            mid, available = future.result()
            results[mid] = available
            done = len(results)
            if done % 5 == 0 or done == total:
                avail = sum(1 for v in results.values() if v)
                yield dict(results), f"Probing... {done}/{total} — {avail} available"

    available_count = sum(1 for v in results.values() if v)
    gr.Info(f"Probed {total} models. {available_count} available, {total - available_count} unavailable.")
    yield dict(results), f"Done. {available_count}/{total} available"


# ---------------------------------------------------------------------------
# Tab 3 — Chat Playground
# ---------------------------------------------------------------------------
def _extract_msg(h):
    """Normalise a Gradio ChatMessage (object or dict) into {role, content} with string values."""
    role = "user"
    content = ""
    if hasattr(h, "role") and hasattr(h, "content"):
        role = str(h.role)
        content = h.content
    elif isinstance(h, dict):
        role = str(h.get("role", "user"))
        content = h.get("content") or h.get("text") or ""
    else:
        content = h

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            elif hasattr(part, "text"):
                parts.append(str(part.text))
            else:
                parts.append(str(part))
        content = " ".join(parts)
    elif not isinstance(content, str):
        content = str(content)

    return {"role": role, "content": content}


def _export_chat(history, model):
    """Export chat history as a Markdown file. Returns a temp file path."""
    if not history:
        raise gr.Error("No chat history to export.")
    lines = [f"# Chat with {model or 'unknown model'}\n"]
    for h in history:
        msg = _extract_msg(h)
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"**{role_label}:** {msg['content']}\n")
    content = "\n".join(lines)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix="nim-chat-", delete=False
    )
    tmp.write(content)
    tmp.close()
    return tmp.name


def _import_chat(file_obj):
    """Import chat history from .md or .json file."""
    if file_obj is None:
        return []
    filepath = file_obj if isinstance(file_obj, str) else file_obj.name
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    history = []
    if filepath.endswith(".json"):
        try:
            data = json.loads(text)
            for msg in data:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history.append(gr.ChatMessage(role=role, content=content))
        except (json.JSONDecodeError, TypeError) as exc:
            raise gr.Error(f"Invalid JSON file: {exc}")
    else:
        # Parse markdown format with **User:** and **Assistant:** markers
        parts = re.split(r'\*\*(User|Assistant)\*\*:\s*', text)
        # parts[0] is header text before first marker, then alternating label/content
        i = 1
        while i < len(parts) - 1:
            label = parts[i].strip()
            content = parts[i + 1].strip()
            role = "user" if label == "User" else "assistant"
            history.append(gr.ChatMessage(role=role, content=content))
            i += 2

    if not history:
        raise gr.Error("No messages found in the imported file.")
    gr.Info(f"Imported {len(history)} messages.")
    return history


def chat_stream(message, history, api_key, model, system_prompt, temperature, max_tokens, top_p):
    """Stream a chat completion from the selected NIM model."""
    if not model:
        raise gr.Error("Select a model first.")
    client = _get_client(api_key)

    messages = []
    sys_text = str(system_prompt or "").strip()
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    for h in history:
        messages.append(_extract_msg(h))
    messages.append({"role": "user", "content": str(message)})

    request_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "top_p": top_p,
        "stream": True,
    }

    req_json = _build_inspector_request(
        "POST", f"{BASE_URL}/chat/completions",
        {"Authorization": f"Bearer {api_key or ''}", "Content-Type": "application/json"},
        request_body,
    )

    try:
        t_start = time.time()
        stream = client.chat.completions.create(**request_body)
    except Exception as exc:
        err = str(exc)
        if "404" in err or "Not Found" in err:
            raise gr.Error(
                f"Model '{model}' is not available for your account. "
                f"Try a different model."
            )
        raise gr.Error(f"Chat request failed: {exc}")

    partial = ""
    token_count = 0
    ttft = None
    usage_info = None
    for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage:
            usage_info = chunk.usage
        delta = chunk.choices[0].delta
        if delta.content:
            if ttft is None:
                ttft = time.time() - t_start
            partial += delta.content
            token_count += 1
            elapsed = time.time() - t_start
            yield partial, ttft, elapsed, token_count, usage_info, req_json, _build_inspector_response(
                200, {"streaming": True, "partial_length": len(partial)}
            )

    total_time = time.time() - t_start
    if ttft is None:
        ttft = total_time
    yield partial, ttft, total_time, token_count, usage_info, req_json, _build_inspector_response(
        200, {"content": partial, "model": model, "tokens": token_count,
               "time_to_first_token": f"{ttft:.2f}s", "total_time": f"{total_time:.2f}s"}
    )


# ---------------------------------------------------------------------------
# Tab 4 — Compare (multi-turn)
# ---------------------------------------------------------------------------
def _stream_to_buffer(client, model, messages, temperature, max_tokens, top_p, buffer, lock):
    """Stream a completion into a shared buffer dict. Runs in a background thread."""
    start = time.time()
    ttft = None
    token_count = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if ttft is None:
                    ttft = time.time() - start
                token_count += 1
                with lock:
                    buffer["text"] += delta.content
                    buffer["elapsed"] = time.time() - start
    except Exception as exc:
        err = str(exc)
        if "404" in err or "Not Found" in err:
            with lock:
                buffer["text"] = f"Model '{model}' is not available for your account. Try a different model."
        else:
            with lock:
                buffer["text"] = f"Error: {exc}"
    with lock:
        buffer["elapsed"] = time.time() - start
        buffer["ttft"] = ttft or (time.time() - start)
        buffer["tokens"] = token_count
        buffer["done"] = True


def compare_models_multiturn(message, history_a, history_b, api_key, model_a, model_b, temperature, max_tokens, top_p):
    """Send a message to two models with full conversation history, streaming both."""
    if not model_a or not model_b:
        raise gr.Error("Select both models before comparing.")
    if not message.strip():
        raise gr.Error("Enter a message.")

    client = _get_client(api_key)

    # Build message lists from histories
    messages_a = [_extract_msg(h) for h in history_a] + [{"role": "user", "content": message.strip()}]
    messages_b = [_extract_msg(h) for h in history_b] + [{"role": "user", "content": message.strip()}]

    lock = threading.Lock()
    buf_a = {"text": "", "elapsed": 0.0, "ttft": 0.0, "tokens": 0, "done": False}
    buf_b = {"text": "", "elapsed": 0.0, "ttft": 0.0, "tokens": 0, "done": False}

    thread_a = threading.Thread(
        target=_stream_to_buffer,
        args=(client, model_a, messages_a, temperature, max_tokens, top_p, buf_a, lock),
    )
    thread_b = threading.Thread(
        target=_stream_to_buffer,
        args=(client, model_b, messages_b, temperature, max_tokens, top_p, buf_b, lock),
    )
    thread_a.start()
    thread_b.start()

    # Update display histories with user message
    display_a = list(history_a) + [gr.ChatMessage(role="user", content=message)]
    display_b = list(history_b) + [gr.ChatMessage(role="user", content=message)]

    while True:
        with lock:
            text_a, elapsed_a, done_a = buf_a["text"], buf_a["elapsed"], buf_a["done"]
            text_b, elapsed_b, done_b = buf_b["text"], buf_b["elapsed"], buf_b["done"]

        show_a = display_a + ([gr.ChatMessage(role="assistant", content=text_a)] if text_a else [])
        show_b = display_b + ([gr.ChatMessage(role="assistant", content=text_b)] if text_b else [])

        time_str_a = f"{elapsed_a:.2f}s" + (" (streaming)" if not done_a else "")
        time_str_b = f"{elapsed_b:.2f}s" + (" (streaming)" if not done_b else "")

        yield show_a, time_str_a, show_b, time_str_b, show_a, show_b, ""

        if done_a and done_b:
            break
        time.sleep(0.1)

    thread_a.join()
    thread_b.join()

    # Final update with completed histories
    final_a = display_a + [gr.ChatMessage(role="assistant", content=buf_a["text"])]
    final_b = display_b + [gr.ChatMessage(role="assistant", content=buf_b["text"])]
    time_str_a = f"{buf_a['elapsed']:.2f}s"
    time_str_b = f"{buf_b['elapsed']:.2f}s"

    yield final_a, time_str_a, final_b, time_str_b, final_a, final_b, ""


# ---------------------------------------------------------------------------
# Tab 6 — Embedding Playground
# ---------------------------------------------------------------------------
def run_embedding_comparison(text_a, text_b, api_key, model):
    """Compare embeddings of two texts using the selected embedding model."""
    if not model:
        raise gr.Error("Select an embedding model first.")
    if not text_a.strip() or not text_b.strip():
        raise gr.Error("Enter text in both fields.")

    client = _get_client(api_key)
    try:
        resp = client.embeddings.create(model=model, input=[text_a.strip(), text_b.strip()])
        emb_a = resp.data[0].embedding
        emb_b = resp.data[1].embedding
    except Exception as exc:
        raise gr.Error(f"Embedding request failed: {exc}")

    similarity = _cosine_similarity(emb_a, emb_b)
    dims = len(emb_a)
    preview_a = ", ".join(f"{v:.4f}" for v in emb_a[:10])
    preview_b = ", ".join(f"{v:.4f}" for v in emb_b[:10])

    sim_color = NVIDIA_GREEN if similarity > 0.8 else ("#f0ad4e" if similarity > 0.5 else "#d9534f")

    result_md = f"""
### Results

**Model** {model}
**Dimensions** {dims}

---

**Cosine Similarity** <span style="color:{sim_color};font-size:24px;font-weight:bold">{similarity:.4f}</span>

| | First 10 Values |
|---|---|
| **Text A** | `[{preview_a}, ...]` |
| **Text B** | `[{preview_b}, ...]` |

---

**Interpretation**
- 1.0 = identical meaning
- 0.8+ = very similar
- 0.5–0.8 = somewhat related
- <0.5 = different topics
"""
    return result_md


# ---------------------------------------------------------------------------
# Tab 7 — Usage Dashboard
# ---------------------------------------------------------------------------
def _render_usage_dashboard(usage_log):
    """Build an HTML usage dashboard from the usage log."""
    if not usage_log:
        return '<div style="text-align:center;padding:40px;color:#888">No usage data yet. Chat with models to see usage stats.</div>'

    # Aggregate per-model
    model_totals = {}
    for record in usage_log:
        model = record.get("model", "unknown")
        if model not in model_totals:
            model_totals[model] = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        model_totals[model]["prompt_tokens"] += record.get("prompt_tokens", 0)
        model_totals[model]["completion_tokens"] += record.get("completion_tokens", 0)
        model_totals[model]["requests"] += 1

    grand_prompt = sum(v["prompt_tokens"] for v in model_totals.values())
    grand_completion = sum(v["completion_tokens"] for v in model_totals.values())
    grand_total = grand_prompt + grand_completion
    grand_requests = sum(v["requests"] for v in model_totals.values())

    # Build table rows
    table_rows = ""
    max_tokens = max((v["prompt_tokens"] + v["completion_tokens"]) for v in model_totals.values()) or 1
    for model, totals in sorted(model_totals.items(), key=lambda x: -(x[1]["prompt_tokens"] + x[1]["completion_tokens"])):
        total_tok = totals["prompt_tokens"] + totals["completion_tokens"]
        bar_width = int((total_tok / max_tokens) * 100)
        cost = _estimate_cost(model, totals["prompt_tokens"], totals["completion_tokens"])
        short_model = model.split("/")[-1] if "/" in model else model
        table_rows += f"""
        <tr style="border-bottom:1px solid #333">
            <td style="padding:8px;color:{NVIDIA_GREEN};font-family:monospace;font-size:12px" title="{model}">{short_model}</td>
            <td style="padding:8px;text-align:right">{totals['requests']}</td>
            <td style="padding:8px;text-align:right">{totals['prompt_tokens']:,}</td>
            <td style="padding:8px;text-align:right">{totals['completion_tokens']:,}</td>
            <td style="padding:8px;text-align:right;font-weight:bold">{total_tok:,}</td>
            <td style="padding:8px;text-align:right;color:{NVIDIA_GREEN}">${cost:.4f}</td>
            <td style="padding:8px;width:120px">
                <div style="background:#333;border-radius:4px;height:16px;overflow:hidden">
                    <div style="background:{NVIDIA_GREEN};height:100%;width:{bar_width}%;border-radius:4px"></div>
                </div>
            </td>
        </tr>"""

    grand_cost = sum(_estimate_cost(m, v["prompt_tokens"], v["completion_tokens"]) for m, v in model_totals.items())

    html = f"""
    <div style="margin-bottom:16px">
        <div style="display:flex;gap:16px;margin-bottom:16px">
            <div style="background:#2a2a4a;padding:16px;border-radius:8px;flex:1;text-align:center;border:1px solid #444">
                <div style="color:#888;font-size:12px">Total Requests</div>
                <div style="color:white;font-size:24px;font-weight:bold">{grand_requests}</div>
            </div>
            <div style="background:#2a2a4a;padding:16px;border-radius:8px;flex:1;text-align:center;border:1px solid #444">
                <div style="color:#888;font-size:12px">Total Tokens</div>
                <div style="color:white;font-size:24px;font-weight:bold">{grand_total:,}</div>
            </div>
            <div style="background:#2a2a4a;padding:16px;border-radius:8px;flex:1;text-align:center;border:1px solid #444">
                <div style="color:#888;font-size:12px">Estimated Cost</div>
                <div style="color:{NVIDIA_GREEN};font-size:24px;font-weight:bold">${grand_cost:.4f}</div>
            </div>
        </div>
        <table style="width:100%;border-collapse:collapse;font-size:14px;color:white">
            <thead><tr style="border-bottom:2px solid {NVIDIA_GREEN};color:#aaa">
                <th style="padding:8px;text-align:left">Model</th>
                <th style="padding:8px;text-align:right">Requests</th>
                <th style="padding:8px;text-align:right">Prompt</th>
                <th style="padding:8px;text-align:right">Completion</th>
                <th style="padding:8px;text-align:right">Total</th>
                <th style="padding:8px;text-align:right">Est. Cost</th>
                <th style="padding:8px;text-align:left">Usage</th>
            </tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Tab 8 — Prompt Chain Builder
# ---------------------------------------------------------------------------
def run_chain(api_key, model, initial_input, step1, step2, step3, step4, step5, usage_log):
    """Execute a sequential prompt chain with up to 5 steps."""
    if not model:
        raise gr.Error("Select a model first.")
    if not initial_input.strip():
        raise gr.Error("Enter an initial input.")

    steps = [s for s in [step1, step2, step3, step4, step5] if s and s.strip()]
    if not steps:
        raise gr.Error("Add at least one step.")

    client = _get_client(api_key)
    results = []
    previous_output = initial_input.strip()
    usage_log = list(usage_log or [])

    for i, step_template in enumerate(steps, 1):
        prompt = step_template.replace("{{input}}", initial_input.strip())
        prompt = prompt.replace("{{previous_output}}", previous_output)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                stream=False,
            )
            output = resp.choices[0].message.content or ""
            prompt_tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
            completion_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
            usage_log.append({
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "timestamp": time.time(),
            })
        except Exception as exc:
            output = f"Error at step {i}: {exc}"

        results.append(f"### Step {i}\n\n**Prompt:**\n```\n{prompt}\n```\n\n**Output:**\n{output}")
        previous_output = output

    return "\n\n---\n\n".join(results), usage_log


# ---------------------------------------------------------------------------
# Tab 9 — Model Leaderboard
# ---------------------------------------------------------------------------
def run_benchmark(api_key, selected_models, usage_log):
    """Run benchmark prompts against selected models and yield partial results."""
    if not selected_models:
        raise gr.Error("Select at least one model.")
    if len(selected_models) > 6:
        raise gr.Error("Select at most 6 models for benchmarking.")

    client = _get_client(api_key)
    usage_log = list(usage_log or [])

    results = {}  # model -> [{category, ttft, total_time, tokens, response}]

    def _run_single(model, prompt_data):
        category = prompt_data["category"]
        prompt = prompt_data["prompt"]
        start = time.time()
        text = ""
        token_count = 0
        ttft = None
        # Try streaming first for TTFT measurement, fall back to non-streaming
        for attempt in range(2):
            try:
                if attempt == 0:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.7,
                        stream=True,
                    )
                    text = ""
                    token_count = 0
                    ttft = None
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            if ttft is None:
                                ttft = time.time() - start
                            text += delta.content
                            token_count += 1
                else:
                    # Non-streaming fallback
                    start = time.time()
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.7,
                        stream=False,
                    )
                    text = resp.choices[0].message.content or ""
                    token_count = getattr(resp.usage, "completion_tokens", 0) if resp.usage else len(text.split())
                    ttft = None
                break  # success
            except Exception as exc:
                err = str(exc)
                if "404" in err or "Not Found" in err:
                    text = f"Model not available for this account."
                    break
                if attempt == 0 and ("peer closed" in err or "incomplete" in err.lower()):
                    time.sleep(0.5)
                    continue
                text = f"Error: {exc}"
                break
        total_time = time.time() - start
        if ttft is None:
            ttft = total_time
        return {
            "model": model,
            "category": category,
            "ttft": ttft,
            "total_time": total_time,
            "tokens": token_count,
            "response": text,
        }

    completed = []
    total_tasks = len(selected_models) * len(BENCHMARK_PROMPTS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for model in selected_models:
            for prompt_data in BENCHMARK_PROMPTS:
                f = executor.submit(_run_single, model, prompt_data)
                futures[f] = (model, prompt_data["category"])

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            usage_log.append({
                "model": result["model"],
                "prompt_tokens": 50,  # approximate
                "completion_tokens": result["tokens"],
                "timestamp": time.time(),
            })
            # Yield partial HTML
            yield _build_leaderboard_html(completed, selected_models, len(completed), total_tasks), usage_log

    yield _build_leaderboard_html(completed, selected_models, total_tasks, total_tasks), usage_log


def _build_leaderboard_html(results, all_models, completed_count, total_count):
    """Build leaderboard HTML from benchmark results."""
    progress_pct = int((completed_count / total_count) * 100) if total_count else 0

    # Aggregate per model
    model_stats = {}
    for r in results:
        model = r["model"]
        if model not in model_stats:
            model_stats[model] = {"ttfts": [], "times": [], "tokens": [], "details": []}
        model_stats[model]["ttfts"].append(r["ttft"])
        model_stats[model]["times"].append(r["total_time"])
        model_stats[model]["tokens"].append(r["tokens"])
        model_stats[model]["details"].append(r)

    # Sort by avg response time
    ranked = sorted(
        model_stats.items(),
        key=lambda x: sum(x[1]["times"]) / len(x[1]["times"]) if x[1]["times"] else 999
    )

    progress_bar = f"""
    <div style="margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;color:#888;font-size:12px;margin-bottom:4px">
            <span>Progress</span><span>{completed_count}/{total_count} tasks</span>
        </div>
        <div style="background:#333;border-radius:4px;height:8px;overflow:hidden">
            <div style="background:{NVIDIA_GREEN};height:100%;width:{progress_pct}%;border-radius:4px;transition:width 0.3s"></div>
        </div>
    </div>
    """

    # Ranking table
    table_rows = ""
    for rank, (model, stats) in enumerate(ranked, 1):
        avg_ttft = sum(stats["ttfts"]) / len(stats["ttfts"])
        avg_time = sum(stats["times"]) / len(stats["times"])
        avg_tokens = sum(stats["tokens"]) / len(stats["tokens"])
        short_model = model.split("/")[-1] if "/" in model else model

        medal = ""
        if rank == 1:
            medal = "🥇 "
        elif rank == 2:
            medal = "🥈 "
        elif rank == 3:
            medal = "🥉 "

        table_rows += f"""
        <tr style="border-bottom:1px solid #333">
            <td style="padding:8px;font-weight:bold;color:white">{medal}{rank}</td>
            <td style="padding:8px;color:{NVIDIA_GREEN};font-family:monospace;font-size:12px" title="{model}">{short_model}</td>
            <td style="padding:8px;text-align:right">{avg_ttft:.3f}s</td>
            <td style="padding:8px;text-align:right">{avg_time:.2f}s</td>
            <td style="padding:8px;text-align:right">{avg_tokens:.0f}</td>
            <td style="padding:8px;text-align:right">{len(stats['details'])}/{len(BENCHMARK_PROMPTS)}</td>
        </tr>"""

    # Per-prompt details
    details_html = ""
    for prompt_data in BENCHMARK_PROMPTS:
        cat = prompt_data["category"]
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        cat_results.sort(key=lambda r: r["total_time"])
        detail_rows = ""
        for r in cat_results:
            short_model = r["model"].split("/")[-1] if "/" in r["model"] else r["model"]
            escaped_response = r["response"][:500].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            detail_rows += f"""
            <tr style="border-bottom:1px solid #333">
                <td style="padding:6px;color:{NVIDIA_GREEN};font-family:monospace;font-size:11px">{short_model}</td>
                <td style="padding:6px;text-align:right">{r['ttft']:.3f}s</td>
                <td style="padding:6px;text-align:right">{r['total_time']:.2f}s</td>
                <td style="padding:6px;font-size:11px;color:#ccc;max-width:400px;overflow:hidden;text-overflow:ellipsis">{escaped_response}</td>
            </tr>"""

        details_html += f"""
        <details style="margin-bottom:8px">
            <summary style="cursor:pointer;color:white;padding:8px;background:#2a2a4a;border-radius:4px;border:1px solid #444">
                {cat}
            </summary>
            <table style="width:100%;border-collapse:collapse;font-size:13px;color:white;margin-top:4px">
                <thead><tr style="color:#aaa;border-bottom:1px solid #555">
                    <th style="padding:6px;text-align:left">Model</th>
                    <th style="padding:6px;text-align:right">TTFT</th>
                    <th style="padding:6px;text-align:right">Total</th>
                    <th style="padding:6px;text-align:left">Response (preview)</th>
                </tr></thead>
                <tbody>{detail_rows}</tbody>
            </table>
        </details>"""

    html = f"""
    {progress_bar}
    <table style="width:100%;border-collapse:collapse;font-size:14px;color:white;margin-bottom:16px">
        <thead><tr style="border-bottom:2px solid {NVIDIA_GREEN};color:#aaa">
            <th style="padding:8px;text-align:left">Rank</th>
            <th style="padding:8px;text-align:left">Model</th>
            <th style="padding:8px;text-align:right">Avg TTFT</th>
            <th style="padding:8px;text-align:right">Avg Time</th>
            <th style="padding:8px;text-align:right">Avg Tokens</th>
            <th style="padding:8px;text-align:right">Completed</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    <h4 style="color:white;margin-top:16px">Per-Prompt Details</h4>
    {details_html}
    """
    return html


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="NIM Explorer") as app:
        gr.HTML(BANNER)

        # Shared state
        api_key_state = gr.State("")
        saved_key = gr.BrowserState("", storage_key="nim_explorer_api_key")
        model_list_state = gr.State([])
        last_req_state = gr.State("")
        last_resp_state = gr.State("")

        # New state variables
        favourites_state = gr.BrowserState([], storage_key="nim_explorer_favourites")
        availability_state = gr.State({})
        session_cost_state = gr.State(0.0)
        usage_log_state = gr.State([])
        compare_history_a = gr.State([])
        compare_history_b = gr.State([])

        # Shared dropdown update function
        def update_model_choices(model_ids, favourites=None):
            return gr.update(choices=_format_model_choices(model_ids, favourites=favourites))

        def update_embedding_choices(model_ids):
            embed_ids = [mid for mid in model_ids if _is_embedding_model(mid)]
            if not embed_ids:
                # Fallback: fetch all models if we only have chat IDs
                return gr.update(choices=[])
            return gr.update(choices=_format_model_choices(embed_ids))

        with gr.Tabs():
            # ===========================================================
            # Tab 1 — API Key
            # ===========================================================
            with gr.Tab("API Key"):
                gr.Markdown("Enter your NVIDIA API key to access NIM models. Falls back to `NVIDIA_API_KEY` env var.")
                api_key_input = gr.Textbox(
                    label="NVIDIA API Key",
                    placeholder="nvapi-...",
                    type="password",
                    value=os.environ.get("NVIDIA_API_KEY", ""),
                )
                validate_btn = gr.Button("Validate", variant="primary")
                key_status = gr.Textbox(label="Status", interactive=False)

                def on_validate(key):
                    status, model_ids, req_json, resp_json = validate_key(key)
                    return key, key, status, model_ids, req_json, resp_json

                validate_btn.click(
                    fn=on_validate,
                    inputs=[api_key_input],
                    outputs=[api_key_state, saved_key, key_status, model_list_state, last_req_state, last_resp_state],
                )

                def on_load_saved_key(browser_key):
                    env_key = os.environ.get("NVIDIA_API_KEY", "")
                    key = browser_key or env_key
                    if key:
                        return key
                    return ""

                app.load(
                    fn=on_load_saved_key,
                    inputs=[saved_key],
                    outputs=[api_key_input],
                )

            # ===========================================================
            # Tab 2 — Model Catalog
            # ===========================================================
            with gr.Tab("Model Catalog"):
                gr.Markdown("Browse available NIM models. Click a row to select it, then use in Playground.")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Catalog", variant="primary")
                    probe_btn = gr.Button("Check Availability", variant="secondary")
                probe_status = gr.Textbox(label="Availability Status", interactive=False, visible=True)
                catalog_html = gr.HTML()
                selected_model_display = gr.Textbox(
                    label="Selected Model",
                    placeholder="Click a model above or type a model ID",
                    elem_id="selected-model",
                )
                use_in_playground_btn = gr.Button("Use in Playground", variant="secondary")

                def on_refresh_start():
                    return f'<div style="text-align:center;padding:40px;color:{NVIDIA_GREEN}"><span style="font-size:18px">Loading model catalog...</span></div>'

                def on_refresh(key):
                    html, model_ids = fetch_catalog(key)
                    gr.Info(f"Loaded {len(model_ids)} models.")
                    return html, model_ids

                refresh_btn.click(
                    fn=on_refresh_start,
                    outputs=[catalog_html],
                ).then(
                    fn=on_refresh,
                    inputs=[api_key_state],
                    outputs=[catalog_html, model_list_state],
                )

                def on_probe(api_key, model_ids):
                    for results, status_text in probe_models(api_key, model_ids):
                        yield results, status_text

                probe_btn.click(
                    fn=on_probe,
                    inputs=[api_key_state, model_list_state],
                    outputs=[availability_state, probe_status],
                )

            # ===========================================================
            # Tab 3 — Chat Playground
            # ===========================================================
            with gr.Tab("Chat Playground"):
                with gr.Row():
                    with gr.Column(scale=1):
                        playground_model = gr.Dropdown(
                            label="Model", choices=[], allow_custom_value=True
                        )
                        fav_btn = gr.Button("★ Toggle Favourite", variant="secondary", size="sm")
                        prompt_template = gr.Dropdown(
                            label="Prompt Template",
                            choices=list(PROMPT_TEMPLATES.keys()),
                            value="Default",
                        )
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="You are a helpful assistant.",
                            lines=3,
                        )
                        temperature = gr.Slider(
                            minimum=0, maximum=2, value=0.7, step=0.05, label="Temperature"
                        )
                        max_tokens = gr.Slider(
                            minimum=1, maximum=4096, value=1024, step=1, label="Max Tokens"
                        )
                        top_p = gr.Slider(
                            minimum=0, maximum=1, value=0.9, step=0.05, label="Top P"
                        )
                        # Token counter + cost display
                        chat_stats = gr.Markdown(
                            value="*No requests yet*",
                            elem_id="chat-stats",
                        )
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Chat", height=480)
                        msg_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message... (Enter to send, Shift+Enter for newline)",
                            lines=2,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary", elem_id="send-btn")
                            clear_btn = gr.Button("Clear", variant="secondary")
                            export_btn = gr.Button("Export Chat", variant="secondary")
                        export_file = gr.File(label="Download", visible=False)
                        import_file = gr.File(
                            label="Import Chat (.md or .json)",
                            file_types=[".md", ".json"],
                            visible=True,
                        )

                # Toggle favourite
                def toggle_favourite(model, favourites, model_ids):
                    if not model:
                        gr.Warning("Select a model first.")
                        return favourites, gr.update()
                    favourites = list(favourites or [])
                    if model in favourites:
                        favourites.remove(model)
                        gr.Info(f"Removed {model} from favourites.")
                    else:
                        favourites.append(model)
                        gr.Info(f"Added {model} to favourites.")
                    return favourites, gr.update(choices=_format_model_choices(model_ids, favourites=favourites))

                fav_btn.click(
                    fn=toggle_favourite,
                    inputs=[playground_model, favourites_state, model_list_state],
                    outputs=[favourites_state, playground_model],
                )

                # Wire "Use in Playground" from catalog tab
                use_in_playground_btn.click(
                    fn=lambda m: m,
                    inputs=[selected_model_display],
                    outputs=[playground_model],
                )

                # Update model dropdown when model list or favourites change
                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state, favourites_state],
                    outputs=[playground_model],
                )

                # Auto-clear chat when model changes
                def on_model_change():
                    return [], "*No requests yet*"

                playground_model.change(
                    fn=on_model_change,
                    outputs=[chatbot, chat_stats],
                )

                def handle_chat(message, history, api_key, model, sys_prompt, temp, max_tok, tp, session_cost, usage_log):
                    if not message.strip():
                        yield history, "", gr.update(interactive=True), "", "", "", session_cost, usage_log
                        return

                    # Add user message, disable send button
                    history = list(history) + [gr.ChatMessage(role="user", content=message)]
                    yield history, "", gr.update(interactive=False), "", "", "", session_cost, usage_log

                    # Stream assistant response
                    stats_md = ""
                    updated = history
                    req_json = ""
                    resp_json = ""
                    final_ttft = 0
                    final_elapsed = 0
                    final_tokens = 0
                    final_partial = ""
                    try:
                        for partial, ttft, elapsed, tokens, usage, req_json, resp_json in chat_stream(
                            message, history[:-1], api_key, model, sys_prompt, temp, max_tok, tp
                        ):
                            final_ttft = ttft
                            final_elapsed = elapsed
                            final_tokens = tokens
                            final_partial = partial
                            updated = list(history) + [gr.ChatMessage(role="assistant", content=partial)]
                            stats_md = (
                                f"**TTFT** {ttft:.2f}s &nbsp; "
                                f"**Elapsed** {elapsed:.2f}s &nbsp; "
                                f"**Tokens** ~{tokens}"
                            )
                            if usage:
                                prompt_tok = getattr(usage, "prompt_tokens", None)
                                comp_tok = getattr(usage, "completion_tokens", None)
                                total_tok = getattr(usage, "total_tokens", None)
                                if prompt_tok is not None:
                                    stats_md += f" &nbsp; **Prompt** {prompt_tok} &nbsp; **Completion** {comp_tok} &nbsp; **Total** {total_tok}"
                                    cost = _estimate_cost(model, prompt_tok, comp_tok)
                                    session_cost = (session_cost or 0) + cost
                                    stats_md += f" &nbsp; **Cost** ${cost:.4f} &nbsp; **Session** ${session_cost:.4f}"
                            yield updated, stats_md, gr.update(interactive=False), req_json, resp_json, "", session_cost, usage_log
                    except gr.Error as e:
                        error_msg = str(e)
                        updated = list(history) + [
                            gr.ChatMessage(role="assistant", content=f"**Error:** {error_msg}")
                        ]
                        stats_md = f"**Error** — {error_msg}"
                        yield updated, stats_md, gr.update(interactive=True), req_json, resp_json, "", session_cost, usage_log
                        return

                    # Append response time overlay to final assistant message
                    overlay = f'\n\n<small style="color:#888">TTFT: {final_ttft:.2f}s | Total: {final_elapsed:.2f}s | ~{final_tokens} tokens</small>'
                    updated = list(history) + [gr.ChatMessage(role="assistant", content=final_partial + overlay)]

                    # Log usage
                    usage_log = list(usage_log or [])
                    usage_log.append({
                        "model": model,
                        "prompt_tokens": final_tokens,  # approximate if no usage object
                        "completion_tokens": final_tokens,
                        "timestamp": time.time(),
                    })

                    # Re-enable send button
                    yield updated, stats_md, gr.update(interactive=True), req_json, resp_json, "", session_cost, usage_log

                send_btn.click(
                    fn=handle_chat,
                    inputs=[msg_input, chatbot, api_key_state, playground_model, system_prompt, temperature, max_tokens, top_p, session_cost_state, usage_log_state],
                    outputs=[chatbot, chat_stats, send_btn, last_req_state, last_resp_state, msg_input, session_cost_state, usage_log_state],
                )

                msg_input.submit(
                    fn=handle_chat,
                    inputs=[msg_input, chatbot, api_key_state, playground_model, system_prompt, temperature, max_tokens, top_p, session_cost_state, usage_log_state],
                    outputs=[chatbot, chat_stats, send_btn, last_req_state, last_resp_state, msg_input, session_cost_state, usage_log_state],
                )

                clear_btn.click(
                    fn=lambda: ([], "*No requests yet*"),
                    outputs=[chatbot, chat_stats],
                )

                prompt_template.change(
                    fn=lambda t: PROMPT_TEMPLATES.get(t, ""),
                    inputs=[prompt_template],
                    outputs=[system_prompt],
                )

                def on_export(history, model):
                    path = _export_chat(history, model)
                    return gr.update(value=path, visible=True)

                export_btn.click(
                    fn=on_export,
                    inputs=[chatbot, playground_model],
                    outputs=[export_file],
                )

                # Conversation import
                def on_import(file_obj):
                    history = _import_chat(file_obj)
                    return history

                import_file.change(
                    fn=on_import,
                    inputs=[import_file],
                    outputs=[chatbot],
                )

            # ===========================================================
            # Tab 4 — Compare (Multi-turn)
            # ===========================================================
            with gr.Tab("Compare"):
                gr.Markdown("Send messages to two models and compare responses. Supports multi-turn conversations.")
                compare_msg_input = gr.Textbox(
                    label="Message", placeholder="Enter a message to send to both models...", lines=2
                )
                with gr.Row():
                    compare_temp = gr.Slider(minimum=0, maximum=2, value=0.7, step=0.05, label="Temperature")
                    compare_max_tokens = gr.Slider(minimum=1, maximum=4096, value=1024, step=1, label="Max Tokens")
                    compare_top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Top P")
                with gr.Row():
                    compare_btn = gr.Button("Send to Both", variant="primary")
                    compare_clear_btn = gr.Button("Clear Both", variant="secondary")

                with gr.Row():
                    with gr.Column():
                        compare_model_a = gr.Dropdown(label="Model A", choices=[], allow_custom_value=True)
                        compare_chat_a = gr.Chatbot(label="Model A Response", height=400)
                        compare_time_a = gr.Textbox(label="Response Time", interactive=False)
                    with gr.Column():
                        compare_model_b = gr.Dropdown(label="Model B", choices=[], allow_custom_value=True)
                        compare_chat_b = gr.Chatbot(label="Model B Response", height=400)
                        compare_time_b = gr.Textbox(label="Response Time", interactive=False)

                # Update compare dropdowns when model list changes
                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state, favourites_state],
                    outputs=[compare_model_a],
                )
                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state, favourites_state],
                    outputs=[compare_model_b],
                )

                compare_btn.click(
                    fn=compare_models_multiturn,
                    inputs=[compare_msg_input, compare_history_a, compare_history_b, api_key_state, compare_model_a, compare_model_b, compare_temp, compare_max_tokens, compare_top_p],
                    outputs=[compare_chat_a, compare_time_a, compare_chat_b, compare_time_b, compare_history_a, compare_history_b, compare_msg_input],
                )

                def clear_compare():
                    return [], [], "", "", [], [], ""

                compare_clear_btn.click(
                    fn=clear_compare,
                    outputs=[compare_chat_a, compare_chat_b, compare_time_a, compare_time_b, compare_history_a, compare_history_b, compare_msg_input],
                )

            # ===========================================================
            # Tab 5 — API Inspector
            # ===========================================================
            with gr.Tab("API Inspector"):
                gr.Markdown("Inspect the last API request and response as formatted JSON.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Request")
                        inspector_req = gr.Code(language="json", label="Request", interactive=False)
                    with gr.Column():
                        gr.Markdown("### Response")
                        inspector_resp = gr.Code(language="json", label="Response", interactive=False)

                # Update inspector when state changes
                last_req_state.change(fn=lambda x: x, inputs=[last_req_state], outputs=[inspector_req])
                last_resp_state.change(fn=lambda x: x, inputs=[last_resp_state], outputs=[inspector_resp])

            # ===========================================================
            # Tab 6 — Embedding Playground
            # ===========================================================
            with gr.Tab("Embeddings"):
                gr.Markdown("Compare text similarity using NIM embedding models.")
                embedding_model = gr.Dropdown(
                    label="Embedding Model", choices=[], allow_custom_value=True
                )
                with gr.Row():
                    embed_text_a = gr.Textbox(label="Text A", placeholder="Enter first text...", lines=4)
                    embed_text_b = gr.Textbox(label="Text B", placeholder="Enter second text...", lines=4)
                embed_btn = gr.Button("Compare Embeddings", variant="primary")
                embed_results = gr.Markdown(value="*Select an embedding model and enter two texts to compare.*")

                embed_btn.click(
                    fn=run_embedding_comparison,
                    inputs=[embed_text_a, embed_text_b, api_key_state, embedding_model],
                    outputs=[embed_results],
                )

                # Embedding dropdown gets separate filter
                model_list_state.change(
                    fn=update_embedding_choices,
                    inputs=[model_list_state],
                    outputs=[embedding_model],
                )

            # ===========================================================
            # Tab 7 — Usage Dashboard
            # ===========================================================
            with gr.Tab("Usage"):
                gr.Markdown("Track token usage and estimated costs across all models in this session.")
                usage_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;color:#888">No usage data yet. Chat with models to see usage stats.</div>'
                )
                usage_refresh_btn = gr.Button("Refresh Dashboard", variant="secondary")

                def refresh_usage(usage_log):
                    return _render_usage_dashboard(usage_log)

                usage_refresh_btn.click(
                    fn=refresh_usage,
                    inputs=[usage_log_state],
                    outputs=[usage_html],
                )

                # Auto-update on usage log change
                usage_log_state.change(
                    fn=refresh_usage,
                    inputs=[usage_log_state],
                    outputs=[usage_html],
                )

            # ===========================================================
            # Tab 8 — Prompt Chain Builder
            # ===========================================================
            with gr.Tab("Chain Builder"):
                gr.Markdown("Build sequential prompt chains. Use `{{input}}` for the initial input and `{{previous_output}}` for the output of the previous step.")
                chain_model = gr.Dropdown(
                    label="Model", choices=[], allow_custom_value=True
                )
                chain_input = gr.Textbox(
                    label="Initial Input",
                    placeholder="Enter the initial input for the chain...",
                    lines=3,
                )
                chain_step_1 = gr.Textbox(
                    label="Step 1",
                    placeholder="Prompt for step 1. Use {{input}} or {{previous_output}}...",
                    lines=2,
                )
                chain_step_2 = gr.Textbox(
                    label="Step 2",
                    placeholder="Prompt for step 2. Use {{previous_output}} from step 1...",
                    lines=2,
                )
                chain_step_3 = gr.Textbox(
                    label="Step 3",
                    placeholder="Prompt for step 3 (optional)...",
                    lines=2,
                )
                chain_step_4 = gr.Textbox(
                    label="Step 4",
                    placeholder="Prompt for step 4 (optional)...",
                    lines=2,
                    visible=False,
                )
                chain_step_5 = gr.Textbox(
                    label="Step 5",
                    placeholder="Prompt for step 5 (optional)...",
                    lines=2,
                    visible=False,
                )
                with gr.Row():
                    chain_add_btn = gr.Button("+ Add Step", variant="secondary", size="sm")
                    chain_remove_btn = gr.Button("- Remove Step", variant="secondary", size="sm")

                chain_run_btn = gr.Button("Run Chain", variant="primary")
                chain_results = gr.Markdown(value="*Define steps and click Run Chain.*")

                # Track visible step count
                chain_visible_steps = gr.State(3)

                def add_chain_step(visible_count):
                    visible_count = min(visible_count + 1, 5)
                    return (
                        visible_count,
                        gr.update(visible=visible_count >= 4),
                        gr.update(visible=visible_count >= 5),
                    )

                def remove_chain_step(visible_count):
                    visible_count = max(visible_count - 1, 1)
                    return (
                        visible_count,
                        gr.update(visible=visible_count >= 4),
                        gr.update(visible=visible_count >= 5),
                    )

                chain_add_btn.click(
                    fn=add_chain_step,
                    inputs=[chain_visible_steps],
                    outputs=[chain_visible_steps, chain_step_4, chain_step_5],
                )
                chain_remove_btn.click(
                    fn=remove_chain_step,
                    inputs=[chain_visible_steps],
                    outputs=[chain_visible_steps, chain_step_4, chain_step_5],
                )

                def on_run_chain(api_key, model, initial_input, s1, s2, s3, s4, s5, usage_log):
                    result_md, updated_log = run_chain(api_key, model, initial_input, s1, s2, s3, s4, s5, usage_log)
                    return result_md, updated_log

                chain_run_btn.click(
                    fn=on_run_chain,
                    inputs=[api_key_state, chain_model, chain_input, chain_step_1, chain_step_2, chain_step_3, chain_step_4, chain_step_5, usage_log_state],
                    outputs=[chain_results, usage_log_state],
                )

                # Update chain model dropdown
                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state, favourites_state],
                    outputs=[chain_model],
                )

            # ===========================================================
            # Tab 9 — Model Leaderboard
            # ===========================================================
            with gr.Tab("Leaderboard"):
                gr.Markdown("Benchmark models on reasoning, coding, summarisation, and creative prompts. Ranked by average response time.")
                leaderboard_models = gr.CheckboxGroup(
                    label="Select Models to Benchmark (max 6)",
                    choices=[],
                )
                leaderboard_run_btn = gr.Button("Run Benchmark", variant="primary")
                leaderboard_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;color:#888">Select models and click Run Benchmark.</div>'
                )

                def on_run_benchmark(api_key, selected_models, usage_log):
                    for html, updated_log in run_benchmark(api_key, selected_models, usage_log):
                        yield html, updated_log

                leaderboard_run_btn.click(
                    fn=on_run_benchmark,
                    inputs=[api_key_state, leaderboard_models, usage_log_state],
                    outputs=[leaderboard_html, usage_log_state],
                )

                # Update leaderboard model choices
                def update_leaderboard_choices(model_ids):
                    return gr.update(choices=_format_model_choices(model_ids))

                model_list_state.change(
                    fn=update_leaderboard_choices,
                    inputs=[model_list_state],
                    outputs=[leaderboard_models],
                )

        # ---------------------------------------------------------------
        # Favourites sync — refresh all chat dropdowns when favourites change
        # ---------------------------------------------------------------
        def refresh_all_dropdowns(model_ids, favourites):
            choices = _format_model_choices(model_ids, favourites=favourites)
            return (
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )

        favourites_state.change(
            fn=refresh_all_dropdowns,
            inputs=[model_list_state, favourites_state],
            outputs=[playground_model, compare_model_a, compare_model_b, chain_model],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7862, css=CSS, theme=THEME)
