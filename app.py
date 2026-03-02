"""
NIM Explorer — Gradio App
=========================
Browse, chat with, and compare NVIDIA NIM models via integrate.api.nvidia.com.

Usage:
    cd nim-explorer && python app.py
    # Opens at http://localhost:7862
"""

import json
import os
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


def _get_client(api_key):
    """Build an OpenAI client pointing at NVIDIA NIM."""
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise gr.Error("No API key provided. Enter your key or set NVIDIA_API_KEY.")
    return OpenAI(base_url=BASE_URL, api_key=key)


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

    search_js = """
    <script>
    (function() {
        var input = document.getElementById('catalog-search');
        if (!input) return;
        input.addEventListener('input', function() {
            var q = this.value.toLowerCase();
            document.querySelectorAll('.catalog-row').forEach(function(row) {
                row.style.display = row.getAttribute('data-model').includes(q) ? '' : 'none';
            });
        });
    })();
    </script>
    """

    html = f"""
    <input id="catalog-search" type="text" placeholder="Filter models..."
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
    {search_js}
    """
    return html, chat_ids


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


def chat_stream(message, history, api_key, model, system_prompt, temperature, max_tokens, top_p):
    """Stream a chat completion from the selected NIM model. Returns (partial, ttft, total_time, tokens, req_json, resp_json)."""
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
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if ttft is None:
                ttft = time.time() - t_start
            partial += delta.content
            token_count += 1
            elapsed = time.time() - t_start
            yield partial, ttft, elapsed, token_count, req_json, _build_inspector_response(
                200, {"streaming": True, "partial_length": len(partial)}
            )

    total_time = time.time() - t_start
    if ttft is None:
        ttft = total_time
    yield partial, ttft, total_time, token_count, req_json, _build_inspector_response(
        200, {"content": partial, "model": model, "tokens": token_count,
               "time_to_first_token": f"{ttft:.2f}s", "total_time": f"{total_time:.2f}s"}
    )


# ---------------------------------------------------------------------------
# Tab 4 — Compare
# ---------------------------------------------------------------------------
def _single_completion(client, model, messages, temperature, max_tokens, top_p):
    """Run a single (non-streaming) chat completion and return (response_text, elapsed)."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=int(max_tokens),
            top_p=top_p,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
    except Exception as exc:
        err = str(exc)
        if "404" in err or "Not Found" in err:
            text = f"Model '{model}' is not available for your account. Try a different model."
        else:
            text = f"Error: {exc}"
    elapsed = time.time() - start
    return text, elapsed


def compare_models(prompt, api_key, model_a, model_b, temperature, max_tokens, top_p):
    """Send the same prompt to two models in parallel."""
    if not model_a or not model_b:
        raise gr.Error("Select both models before comparing.")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")

    client = _get_client(api_key)
    messages = [{"role": "user", "content": prompt.strip()}]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_single_completion, client, model_a, messages, temperature, max_tokens, top_p)
        fut_b = pool.submit(_single_completion, client, model_b, messages, temperature, max_tokens, top_p)
        text_a, time_a = fut_a.result()
        text_b, time_b = fut_b.result()

    history_a = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text_a}]
    history_b = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text_b}]

    return history_a, f"{time_a:.2f}s", history_b, f"{time_b:.2f}s"


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="NIM Explorer") as app:
        gr.HTML(BANNER)

        # Shared state
        api_key_state = gr.State("")
        model_list_state = gr.State([])
        last_req_state = gr.State("")
        last_resp_state = gr.State("")

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
                    return key, status, model_ids, req_json, resp_json

                validate_btn.click(
                    fn=on_validate,
                    inputs=[api_key_input],
                    outputs=[api_key_state, key_status, model_list_state, last_req_state, last_resp_state],
                )

            # ===========================================================
            # Tab 2 — Model Catalog
            # ===========================================================
            with gr.Tab("Model Catalog"):
                gr.Markdown("Browse available NIM models. Click a row to select it, then use in Playground.")
                refresh_btn = gr.Button("Refresh Catalog", variant="primary")
                catalog_html = gr.HTML()
                selected_model_display = gr.Textbox(
                    label="Selected Model",
                    placeholder="Click a model above or type a model ID",
                    elem_id="selected-model",
                )
                use_in_playground_btn = gr.Button("Use in Playground", variant="secondary")

                def on_refresh(key):
                    html, model_ids = fetch_catalog(key)
                    gr.Info(f"Loaded {len(model_ids)} models.")
                    return html, model_ids

                refresh_btn.click(
                    fn=on_refresh,
                    inputs=[api_key_state],
                    outputs=[catalog_html, model_list_state],
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
                        # Token counter display
                        chat_stats = gr.Markdown(
                            value="*No requests yet*",
                            elem_id="chat-stats",
                        )
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(label="Chat", height=480)
                        msg_input = gr.Textbox(
                            label="Message",
                            placeholder="Type your message... (Shift+Enter for newline)",
                            lines=2,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", variant="primary", elem_id="send-btn")
                            clear_btn = gr.Button("Clear", variant="secondary")

                # Wire "Use in Playground" from catalog tab
                use_in_playground_btn.click(
                    fn=lambda m: m,
                    inputs=[selected_model_display],
                    outputs=[playground_model],
                )

                # Update model dropdown when model list changes
                def update_model_choices(model_ids):
                    return gr.update(choices=model_ids)

                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state],
                    outputs=[playground_model],
                )

                # Auto-clear chat when model changes
                def on_model_change():
                    return [], "*No requests yet*"

                playground_model.change(
                    fn=on_model_change,
                    outputs=[chatbot, chat_stats],
                )

                def handle_chat(message, history, api_key, model, sys_prompt, temp, max_tok, tp):
                    if not message.strip():
                        yield history, "", gr.update(interactive=True), "", "", ""
                        return

                    # Add user message, disable send button
                    history = list(history) + [gr.ChatMessage(role="user", content=message)]
                    yield history, "", gr.update(interactive=False), "", "", ""

                    # Stream assistant response, catch errors gracefully
                    stats_md = ""
                    updated = history
                    req_json = ""
                    resp_json = ""
                    try:
                        for partial, ttft, elapsed, tokens, req_json, resp_json in chat_stream(
                            message, history[:-1], api_key, model, sys_prompt, temp, max_tok, tp
                        ):
                            updated = list(history) + [gr.ChatMessage(role="assistant", content=partial)]
                            stats_md = (
                                f"**TTFT** {ttft:.2f}s &nbsp; "
                                f"**Elapsed** {elapsed:.2f}s &nbsp; "
                                f"**Tokens** ~{tokens}"
                            )
                            yield updated, stats_md, gr.update(interactive=False), req_json, resp_json, ""
                    except gr.Error as e:
                        error_msg = str(e)
                        updated = list(history) + [
                            gr.ChatMessage(role="assistant", content=f"**Error:** {error_msg}")
                        ]
                        stats_md = f"**Error** — {error_msg}"
                        yield updated, stats_md, gr.update(interactive=True), req_json, resp_json, ""
                        return

                    # Re-enable send button
                    yield updated, stats_md, gr.update(interactive=True), req_json, resp_json, ""

                send_btn.click(
                    fn=handle_chat,
                    inputs=[msg_input, chatbot, api_key_state, playground_model, system_prompt, temperature, max_tokens, top_p],
                    outputs=[chatbot, chat_stats, send_btn, last_req_state, last_resp_state, msg_input],
                )

                msg_input.submit(
                    fn=handle_chat,
                    inputs=[msg_input, chatbot, api_key_state, playground_model, system_prompt, temperature, max_tokens, top_p],
                    outputs=[chatbot, chat_stats, send_btn, last_req_state, last_resp_state, msg_input],
                )

                clear_btn.click(
                    fn=lambda: ([], "*No requests yet*"),
                    outputs=[chatbot, chat_stats],
                )

            # ===========================================================
            # Tab 4 — Compare
            # ===========================================================
            with gr.Tab("Compare"):
                gr.Markdown("Send the same prompt to two models and compare responses side by side.")
                compare_prompt = gr.Textbox(
                    label="Shared Prompt", placeholder="Enter a prompt to send to both models...", lines=3
                )
                with gr.Row():
                    compare_temp = gr.Slider(minimum=0, maximum=2, value=0.7, step=0.05, label="Temperature")
                    compare_max_tokens = gr.Slider(minimum=1, maximum=4096, value=1024, step=1, label="Max Tokens")
                    compare_top_p = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.05, label="Top P")
                compare_btn = gr.Button("Send to Both", variant="primary")

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
                    inputs=[model_list_state],
                    outputs=[compare_model_a],
                )
                model_list_state.change(
                    fn=update_model_choices,
                    inputs=[model_list_state],
                    outputs=[compare_model_b],
                )

                compare_btn.click(
                    fn=compare_models,
                    inputs=[compare_prompt, api_key_state, compare_model_a, compare_model_b, compare_temp, compare_max_tokens, compare_top_p],
                    outputs=[compare_chat_a, compare_time_a, compare_chat_b, compare_time_b],
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

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7862, css=CSS, theme=THEME)
