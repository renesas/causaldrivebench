"""
Modular LLM provider abstraction.

Supports OpenAI, Google Gemini, and Anthropic Claude with a unified interface
for both real-time and batch API calls.

Usage:
    from common.llm_provider import get_provider

    provider = get_provider("gemini", model="gemini-2.5-pro", api_key="...")
    text, usage = provider.call(system_prompt, content_blocks, max_tokens=65536)
"""

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model: str, api_key: str = None, reasoning: str = None):
        self.model = model
        self.api_key = api_key
        self.reasoning = reasoning

    @abstractmethod
    def call(self, system_prompt: str, content_blocks: List[Dict],
             max_tokens: int = 65536) -> Tuple[str, Dict]:
        """Make a real-time API call.

        Args:
            system_prompt: System instructions.
            content_blocks: List of content blocks in internal format:
                - {"type": "text", "text": "..."}
                - {"type": "image", "mime": "image/jpeg", "data": "<base64>"}
            max_tokens: Maximum output tokens.

        Returns:
            (raw_text, usage_dict) where usage_dict has keys:
                input, output, image (token counts)
        """
        ...

    # ── Batch API methods ────────────────────────────────────────────────

    def format_batch_request(self, custom_id: str, system_prompt: str,
                             content_blocks: List[Dict],
                             max_tokens: int = 65536) -> Dict:
        """Format a single batch request in provider-specific format.
        Content blocks should be in internal format (text/image).
        """
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    def submit_batch(self, requests: List[Dict],
                     description: str = "") -> str:
        """Submit batch requests and return batch ID."""
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    def get_batch_status(self, batch_id: str) -> Dict:
        """Return batch status dict with keys: status, succeeded, errored, total."""
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    def download_batch_results(self, batch_id: str, output_path: str) -> Optional[str]:
        """Download results to output_path. Returns path or None."""
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    def parse_batch_response(self, line: str) -> Tuple[str, str, Dict]:
        """Parse one result line. Returns (custom_id, raw_text, usage_dict)."""
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    def list_batches(self, limit: int = 10) -> List[Dict]:
        """List recent batch jobs."""
        raise NotImplementedError(f"{self.provider_name} does not support batch API")

    @staticmethod
    def normalize_block(block: Dict) -> Dict:
        """Normalize any content block format to internal format.
        Handles internal, OpenAI Responses, and OpenAI Batch formats.
        """
        btype = block.get("type", "")

        if btype == "image":
            return block

        if btype == "text":
            return {"type": "text", "text": block.get("text", "")}

        if btype == "input_text":
            return {"type": "text", "text": block.get("text", "")}

        if btype == "input_image":
            url = block.get("image_url", "")
            m = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
            if m:
                return {"type": "image", "mime": m.group(1), "data": m.group(2)}
            return {"type": "image", "mime": "image/jpeg", "data": url}

        if btype == "image_url":
            nested = block.get("image_url", {})
            url = nested.get("url", "") if isinstance(nested, dict) else str(nested)
            m = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
            if m:
                return {"type": "image", "mime": m.group(1), "data": m.group(2)}
            return {"type": "image", "mime": "image/jpeg", "data": url}

        return {"type": "text", "text": str(block)}

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__.replace("Provider", "").lower()


# ── Internal content block helpers (provider-agnostic) ───────────────────

def make_text_block(text: str) -> Dict:
    """Create an internal text content block."""
    return {"type": "text", "text": text}


def make_image_block(b64_data: str, mime: str = "image/jpeg") -> Dict:
    """Create an internal image content block."""
    return {"type": "image", "mime": mime, "data": b64_data}


def make_image_block_from_path(path: str, label: str = None,
                                flip_horizontal: bool = False) -> Dict:
    """Create an internal image content block from a file path.
    Optionally adds a text label overlay and flips horizontally.
    """
    import cv2

    if label or flip_horizontal:
        img = cv2.imread(str(path))
        if img is not None:
            if flip_horizontal:
                img = cv2.flip(img, 1)
            if label:
                cv2.putText(img, label, (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", img)
            b64 = base64.standard_b64encode(buf.tobytes()).decode("utf-8")
            return make_image_block(b64, "image/jpeg")

    ext = Path(path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    return make_image_block(b64, mime)


# ── OpenAI Provider ──────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """OpenAI provider using the Responses API (real-time)."""

    def __init__(self, model: str = "gpt-5.2", api_key: str = None,
                 reasoning: str = None):
        super().__init__(model, api_key, reasoning)
        from openai import OpenAI
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)

    def _to_responses_format(self, block: Dict) -> Dict:
        if block["type"] == "text":
            return {"type": "input_text", "text": block["text"]}
        elif block["type"] == "image":
            url = f"data:{block['mime']};base64,{block['data']}"
            return {"type": "input_image", "image_url": url}
        return block

    def call(self, system_prompt: str, content_blocks: List[Dict],
             max_tokens: int = 65536) -> Tuple[str, Dict]:
        normalized = [self.normalize_block(b) for b in content_blocks]
        api_blocks = [self._to_responses_format(b) for b in normalized]

        api_kwargs = dict(
            model=self.model,
            instructions=system_prompt,
            input=[{"role": "user", "content": api_blocks}],
            max_output_tokens=max_tokens,
        )
        if self.reasoning and self.model.startswith(("o", "gpt-5")):
            api_kwargs["reasoning"] = {"effort": self.reasoning}

        response = self.client.responses.create(**api_kwargs)
        raw = response.output_text.strip()

        usage = {}
        if response.usage:
            u = response.usage
            img_tokens = 0
            if hasattr(u, "input_tokens_details") and u.input_tokens_details:
                img_tokens = getattr(u.input_tokens_details, "image_tokens", 0) or 0
            usage = {"input": u.input_tokens, "output": u.output_tokens, "image": img_tokens}
            if hasattr(u, "output_tokens_details") and u.output_tokens_details:
                usage["reasoning"] = getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0

        return raw, usage


    # ── Batch API ──

    def _to_chat_format(self, block: Dict) -> Dict:
        """Convert internal block to OpenAI Chat Completions format (for batch)."""
        if block["type"] == "text":
            return {"type": "text", "text": block["text"]}
        elif block["type"] == "image":
            url = f"data:{block['mime']};base64,{block['data']}"
            return {"type": "image_url", "image_url": {"url": url}}
        return block

    def format_batch_request(self, custom_id: str, system_prompt: str,
                             content_blocks: List[Dict],
                             max_tokens: int = 65536) -> Dict:
        normalized = [self.normalize_block(b) for b in content_blocks]
        api_blocks = [self._to_chat_format(b) for b in normalized]
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": api_blocks},
            ],
            "max_completion_tokens": max_tokens,
        }
        if self.reasoning and self.model.startswith(("o1", "gpt-5")):
            body["reasoning_effort"] = self.reasoning
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit_batch(self, requests: List[Dict], description: str = "") -> str:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
            tmp_path = f.name
        with open(tmp_path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="batch")
        os.remove(tmp_path)
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
        )
        return batch.id

    def get_batch_status(self, batch_id: str) -> Dict:
        batch = self.client.batches.retrieve(batch_id)
        return {
            "status": batch.status,
            "succeeded": batch.request_counts.completed,
            "errored": batch.request_counts.failed,
            "total": batch.request_counts.total,
        }

    def download_batch_results(self, batch_id: str, output_path: str) -> Optional[str]:
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed" or not batch.output_file_id:
            return None
        content = self.client.files.content(batch.output_file_id)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(content.content)
        return output_path

    def parse_batch_response(self, line: str) -> Tuple[str, str, Dict]:
        data = json.loads(line)
        custom_id = data.get("custom_id", "")
        if data.get("error"):
            return custom_id, "", {"error": data["error"].get("message", "Unknown")}
        resp = data.get("response", {})
        if resp.get("status_code") != 200:
            return custom_id, "", {"error": f"HTTP {resp.get('status_code')}"}
        body = resp.get("body", {})
        choices = body.get("choices", [])
        raw_text = choices[0]["message"]["content"] if choices else ""
        usage = {}
        if "usage" in body:
            u = body["usage"]
            usage = {"input": u.get("prompt_tokens", 0),
                     "output": u.get("completion_tokens", 0), "image": 0}
        return custom_id, raw_text, usage

    def list_batches(self, limit: int = 10) -> List[Dict]:
        batches = self.client.batches.list(limit=limit)
        result = []
        for b in batches.data:
            result.append({
                "id": b.id, "status": b.status,
                "succeeded": b.request_counts.completed,
                "errored": b.request_counts.failed,
                "total": b.request_counts.total,
                "description": (b.metadata or {}).get("description", ""),
            })
        return result


# ── Google Gemini Provider ───────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """Google Gemini provider using the google-genai SDK."""

    def __init__(self, model: str = "gemini-3.1-pro-preview", api_key: str = None,
                 reasoning: str = None):
        super().__init__(model, api_key, reasoning)
        from google import genai
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var or pass api_key.")
        self.client = genai.Client(api_key=api_key)
        self._genai = genai

    def _to_gemini_parts(self, block: Dict):
        from google.genai import types
        if block["type"] == "text":
            return types.Part.from_text(text=block["text"])
        elif block["type"] == "image":
            return types.Part.from_bytes(
                data=base64.standard_b64decode(block["data"]),
                mime_type=block["mime"]
            )
        return None

    def _map_reasoning(self) -> Optional[str]:
        """Map reasoning effort to Gemini thinking_level (LOW/MEDIUM/HIGH)."""
        if not self.reasoning or self.reasoning == "none":
            return None
        mapping = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
        return mapping.get(self.reasoning, "MEDIUM")

    def call(self, system_prompt: str, content_blocks: List[Dict],
             max_tokens: int = 65536) -> Tuple[str, Dict]:
        from google.genai import types

        normalized = [self.normalize_block(b) for b in content_blocks]
        parts = [self._to_gemini_parts(b) for b in normalized]
        parts = [p for p in parts if p is not None]

        config_kwargs = {
            "max_output_tokens": max_tokens,
            "system_instruction": system_prompt,
        }

        thinking_level = self._map_reasoning()
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=parts,
            config=config,
        )

        raw = response.text.strip() if response.text else ""

        usage = {}
        if response.usage_metadata:
            u = response.usage_metadata
            usage = {
                "input": getattr(u, "prompt_token_count", 0) or 0,
                "output": getattr(u, "candidates_token_count", 0) or 0,
                "image": 0,
            }
            if hasattr(u, "thoughts_token_count"):
                usage["reasoning"] = getattr(u, "thoughts_token_count", 0) or 0

        return raw, usage


# ── Anthropic Claude Provider ────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-opus-4-6", api_key: str = None,
                 reasoning: str = None):
        super().__init__(model, api_key, reasoning)
        import anthropic
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = anthropic.Anthropic(**kwargs)

    def _to_claude_format(self, block: Dict) -> Dict:
        if block["type"] == "text":
            return {"type": "text", "text": block["text"]}
        elif block["type"] == "image":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block["mime"],
                    "data": block["data"],
                }
            }
        return block

    def call(self, system_prompt: str, content_blocks: List[Dict],
             max_tokens: int = 65536) -> Tuple[str, Dict]:
        normalized = [self.normalize_block(b) for b in content_blocks]
        api_blocks = [self._to_claude_format(b) for b in normalized]

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": api_blocks}],
        )

        if self.reasoning and self.reasoning != "none":
            kwargs["thinking"] = {"type": "adaptive"}
            effort_map = {"low": "low", "medium": "medium", "high": "high"}
            kwargs["effort"] = effort_map.get(self.reasoning, "medium")

        response = self.client.messages.create(**kwargs)

        raw = ""
        for block in response.content:
            if block.type == "text":
                raw = block.text.strip()
                break

        usage = {}
        if response.usage:
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "image": 0,
            }

        return raw, usage

    # ── Batch API ──

    def format_batch_request(self, custom_id: str, system_prompt: str,
                             content_blocks: List[Dict],
                             max_tokens: int = 65536) -> Dict:
        normalized = [self.normalize_block(b) for b in content_blocks]
        api_blocks = [self._to_claude_format(b) for b in normalized]
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": api_blocks}],
        }
        # Map reasoning → adaptive thinking + effort
        if self.reasoning and self.reasoning != "none":
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self.reasoning}
        return {"custom_id": custom_id, "params": params}

    def submit_batch(self, requests: List[Dict], description: str = "") -> str:
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def get_batch_status(self, batch_id: str) -> Dict:
        batch = self.client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        return {
            "status": "completed" if batch.processing_status == "ended" else batch.processing_status,
            "succeeded": counts.succeeded,
            "errored": counts.errored,
            "total": counts.succeeded + counts.errored + counts.processing,
        }

    def download_batch_results(self, batch_id: str, output_path: str) -> Optional[str]:
        batch = self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            return None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in self.client.messages.batches.results(batch_id):
                f.write(json.dumps(result.model_dump()) + "\n")
        return output_path

    def parse_batch_response(self, line: str) -> Tuple[str, str, Dict]:
        data = json.loads(line)
        custom_id = data.get("custom_id", "")
        result = data.get("result", {})
        if result.get("type") != "succeeded":
            err = result.get("error", {})
            return custom_id, "", {"error": err.get("message", result.get("type", "failed"))}
        msg = result.get("message", {})
        raw_text = ""
        for block in msg.get("content", []):
            if block.get("type") == "text":
                raw_text = block.get("text", "").strip()
                break
        usage = {}
        u = msg.get("usage", {})
        if u:
            usage = {"input": u.get("input_tokens", 0),
                     "output": u.get("output_tokens", 0), "image": 0}
        return custom_id, raw_text, usage

    def list_batches(self, limit: int = 10) -> List[Dict]:
        batches = self.client.messages.batches.list(limit=limit)
        result = []
        for b in batches.data:
            counts = b.request_counts
            result.append({
                "id": b.id, "status": b.processing_status,
                "succeeded": counts.succeeded,
                "errored": counts.errored,
                "total": counts.succeeded + counts.errored + counts.processing,
                "description": "",
            })
        return result


# ── Provider Factory ─────────────────────────────────────────────────────

PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "claude": ClaudeProvider,
    "anthropic": ClaudeProvider,
}


def get_provider(provider_name: str, model: str = None, api_key: str = None,
                 reasoning: str = None, **kwargs) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        provider_name: "openai", "gemini"/"google", "claude"/"anthropic"
        model: Model identifier (e.g. "gpt-5.2", "gemini-2.5-pro", "claude-sonnet-4-20250514")
        api_key: API key (or set via env var)
        reasoning: Reasoning effort: "none", "low", "medium", "high"
    """
    provider_name = provider_name.lower().strip()
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider_name}'. "
                        f"Available: {list(PROVIDERS.keys())}")

    cls = PROVIDERS[provider_name]
    return cls(model=model, api_key=api_key, reasoning=reasoning, **kwargs)


def get_provider_from_config(config: Dict) -> LLMProvider:
    """Create provider from config.yaml dict."""
    provider_name = config.get("provider", "openai")
    model = config.get("model", "gpt-5.2")
    reasoning = config.get("reasoning", "none")

    api_key = None
    if provider_name in ("openai",):
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
    elif provider_name in ("gemini", "google"):
        api_key = (config.get("google_api_key") or config.get("gemini_api_key")
                   or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    elif provider_name in ("claude", "anthropic"):
        api_key = (config.get("anthropic_api_key") or config.get("claude_api_key")
                   or os.environ.get("ANTHROPIC_API_KEY"))

    return get_provider(provider_name, model=model, api_key=api_key, reasoning=reasoning)
