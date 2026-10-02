"""
LLM-based judge for scoring open-ended model responses.

Evaluates reasoning quality along four dimensions, each scored on a 3-point
scale (0=Absent, 1=Partial, 2=Complete):

  1. Spatial Grounding         — Are object positions and spatial relationships
                                  correctly identified relative to the ego vehicle?
  2. Mechanism Identification  — Is the causal mechanism linking cause to effect
                                  correctly described?
  3. Conditionality Awareness  — Are the conditions under which the causal
                                  relationship holds (or doesn't) correctly stated?
  4. Answer Consistency        — Is the final stated answer (Yes/No or A/B/C/D)
                                  consistent with the reasoning provided?

Composite score = mean of all four dimensions (range 0–2).

Supported providers
-------------------
  "anthropic"     : Direct Anthropic API (default; uses claude-opus-4-6)
  "openai"        : Direct OpenAI API (uses gpt-4o)
  "azure_foundry" : Azure AI Foundry — Anthropic Claude hosted on Azure AI Foundry.
                    Uses the AnthropicFoundry client from the anthropic SDK.
                    Requires: ANTHROPIC_FOUNDRY_API_KEY + FOUNDRY_BASE_URL
                    Endpoint format: "https://<resource>.services.ai.azure.com/anthropic/v1"
  "azure_openai"  : Azure OpenAI Service — GPT models hosted on Azure.
                    Uses the AzureOpenAI client from the openai SDK.
                    Requires: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT

Prompt templates
----------------
System prompt and user template are loaded from:
    evaluation/common/prompts/llm_judge_system.txt
    evaluation/common/prompts/llm_judge_user_template.txt
Edit those files to change the rubric or format without touching Python code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from evaluation/common/prompts/<filename>."""
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            "Expected at evaluation/common/prompts/{filename}"
        )
    return path.read_text(encoding="utf-8")


# Module-level constants loaded from files — edit the .txt files, not here.
JUDGE_SYSTEM_PROMPT: str = _load_prompt("llm_judge_system.txt")
JUDGE_USER_TEMPLATE: str = _load_prompt("llm_judge_user_template.txt")

_DIMENSIONS = [
    "spatial_grounding",
    "mechanism_identification",
    "conditionality_awareness",
    "answer_consistency",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """
    Extract a JSON object from text that may be wrapped in markdown code fences.

    Tries in order:
      1. Parse the text directly as JSON.
      2. Extract content from ```json ... ``` or ``` ... ``` blocks.
      3. Find the first {...} substring.
    """
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return text


def _clamp(value: int, lo: int = 0, hi: int = 2) -> int:
    return max(lo, min(hi, int(value)))


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Score model responses using an LLM as judge.

    Four-dimension reasoning rubric, each scored 0–2.
    Composite score = mean across all four dimensions (range 0–2).

    Parameters
    ----------
    provider : "anthropic" | "openai" | "azure_foundry" | "azure_openai"
        API provider.  See module docstring for details and required env vars.

    model : model identifier (provider-specific).
        Defaults per provider:
          anthropic     → "claude-opus-4-6"
          openai        → "gpt-4o"
          azure_foundry → "claude-opus-4-6"   (Azure deployment name)
          azure_openai  → ""                   (must be set; no default)

    api_key_env : name of the environment variable holding the API key.
        Defaults per provider:
          anthropic     → "ANTHROPIC_API_KEY"
          openai        → "OPENAI_API_KEY"
          azure_foundry → "ANTHROPIC_FOUNDRY_API_KEY"
          azure_openai  → "AZURE_OPENAI_API_KEY"

    azure_endpoint_env : env var that holds the Azure endpoint / base URL.
        Used only by azure_foundry and azure_openai.
        Defaults per provider:
          azure_foundry → "FOUNDRY_BASE_URL"
                          Value format: "https://<resource>.services.ai.azure.com/anthropic/v1"
          azure_openai  → "AZURE_OPENAI_ENDPOINT"
                          Value format: "https://<resource>.openai.azure.com/"

    azure_api_version : Azure OpenAI API version string.
        Only used by azure_openai (ignored for azure_foundry).
        Default: "2024-12-01-preview"
    """

    _PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
        "anthropic":    "claude-opus-4-6",
        "openai":       "gpt-4o",
        "azure_foundry": "claude-opus-4-6",
        "azure_openai": "",
    }

    _PROVIDER_DEFAULT_KEY_ENVS: Dict[str, str] = {
        "anthropic":    "ANTHROPIC_API_KEY",
        "openai":       "OPENAI_API_KEY",
        "azure_foundry": "ANTHROPIC_FOUNDRY_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
    }

    _PROVIDER_DEFAULT_ENDPOINT_ENVS: Dict[str, str] = {
        "azure_foundry": "FOUNDRY_BASE_URL",
        "azure_openai":  "AZURE_OPENAI_ENDPOINT",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "",
        api_key_env: str = "",
        azure_endpoint_env: str = "",
        azure_api_version: str = "2024-12-01-preview",
        fallback_provider: str = "",
        fallback_model: str = "",
    ) -> None:
        if provider not in self._PROVIDER_DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider: {provider!r}. "
                f"Choose from: {list(self._PROVIDER_DEFAULT_MODELS)}"
            )

        self.provider = provider
        self.model = model or self._PROVIDER_DEFAULT_MODELS[provider]
        self.api_key_env = api_key_env or self._PROVIDER_DEFAULT_KEY_ENVS[provider]
        self.azure_endpoint_env = (
            azure_endpoint_env
            or self._PROVIDER_DEFAULT_ENDPOINT_ENVS.get(provider, "")
        )
        self.azure_api_version = azure_api_version
        self._client = None

        # Optional fallback judge — used automatically on 401 auth errors from primary.
        self._fallback_judge: Optional["LLMJudge"] = None
        if fallback_provider and fallback_provider != provider:
            self._fallback_judge = LLMJudge(
                provider=fallback_provider,
                model=fallback_model or self._PROVIDER_DEFAULT_MODELS.get(fallback_provider, ""),
            )
            logger.info(
                "LLMJudge: fallback configured — %s/%s → %s/%s",
                provider, self.model, fallback_provider, self._fallback_judge.model,
            )

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily initialise the API client based on provider."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {self.api_key_env!r} is not set. "
                "Please set it before using LLMJudge."
            )

        if self.provider == "anthropic":
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                ) from exc
            # When Claude Code runs via Azure AI Foundry (CLAUDE_CODE_USE_FOUNDRY=1),
            # the SDK auto-routes all Anthropic() calls through Foundry when
            # ANTHROPIC_FOUNDRY_RESOURCE is set, ignoring the direct API key → 401.
            # Temporarily unset Foundry env vars so this client hits api.anthropic.com.
            _foundry_vars = (
                "ANTHROPIC_FOUNDRY_RESOURCE",
                "ANTHROPIC_FOUNDRY_API_KEY",
                "FOUNDRY_BASE_URL",
                "CLAUDE_CODE_USE_FOUNDRY",
            )
            _saved = {k: os.environ.pop(k, None) for k in _foundry_vars}
            try:
                self._client = anthropic.Anthropic(api_key=api_key)
            finally:
                for k, v in _saved.items():
                    if v is not None:
                        os.environ[k] = v

        elif self.provider == "openai":
            try:
                import openai  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "openai package required: pip install openai"
                ) from exc
            self._client = openai.OpenAI(api_key=api_key)

        elif self.provider == "azure_foundry":
            # Azure AI Foundry: Anthropic Claude via the AnthropicFoundry client.
            #
            # Priority order:
            #   1. ANTHROPIC_FOUNDRY_RESOURCE — SDK reads this automatically and builds
            #      base_url as https://<resource>.services.ai.azure.com/anthropic/
            #      Do NOT pass base_url when this is set: the SDK raises
            #      "base_url and resource are mutually exclusive".
            #   2. FOUNDRY_BASE_URL — custom env var; strip /v1 suffix if present because
            #      the SDK appends its own /v1 path segment.
            #   3. No endpoint vars — AnthropicFoundry() raises if nothing is set.
            try:
                from anthropic import AnthropicFoundry  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "anthropic>=0.40 is required for azure_foundry: pip install anthropic"
                ) from exc
            foundry_resource = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE", "")
            foundry_base_url = os.environ.get("FOUNDRY_BASE_URL", "")
            if foundry_resource:
                # Let the SDK read ANTHROPIC_FOUNDRY_RESOURCE from env automatically.
                # Passing resource= explicitly would conflict if the env var is also set.
                self._client = AnthropicFoundry(api_key=api_key)
                logger.info("LLMJudge: azure_foundry client initialised (resource=%s)", foundry_resource)
            elif foundry_base_url:
                # Strip trailing /v1 — SDK appends its own versioned path.
                base = foundry_base_url.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                self._client = AnthropicFoundry(api_key=api_key, base_url=base.rstrip("/") + "/")
                logger.info("LLMJudge: azure_foundry client initialised (base_url=%s)", base)
            else:
                self._client = AnthropicFoundry(api_key=api_key)
                logger.info("LLMJudge: azure_foundry client initialised (env defaults)")

        elif self.provider == "azure_openai":
            # Azure OpenAI Service: GPT models via AzureOpenAI client.
            # Requires openai>=1.0 and an Azure OpenAI resource.
            # Endpoint format: https://<resource-name>.openai.azure.com/
            try:
                import openai  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "openai package required for azure_openai: pip install openai"
                ) from exc

            endpoint = os.environ.get(self.azure_endpoint_env, "")
            if not endpoint:
                raise EnvironmentError(
                    f"Environment variable {self.azure_endpoint_env!r} is not set. "
                    "Required for azure_openai provider. "
                    "Set it to: https://<resource>.openai.azure.com/"
                )
            self._client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=self.azure_api_version,
            )
            logger.info(
                "LLMJudge: azure_openai client initialised (endpoint=%s, api_version=%s)",
                endpoint, self.azure_api_version,
            )

        return self._client

    # ------------------------------------------------------------------
    # Auth error detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_auth_error(exc_str: str) -> bool:
        """Return True if the exception string indicates an authentication failure."""
        lower = exc_str.lower()
        return (
            "401" in exc_str
            or "authentication_error" in lower
            or "invalid x-api-key" in lower
            or "invalid api key" in lower
            or "unauthorized" in lower
        )

    @staticmethod
    def _is_rate_limit_error(exc_str: str) -> bool:
        """Return True if the exception string indicates a rate-limit response."""
        return "429" in exc_str or "ratelimitreached" in exc_str.lower() or "rate limit" in exc_str.lower()

    # ------------------------------------------------------------------
    # Single response scoring
    # ------------------------------------------------------------------

    def score_response(
        self,
        question: str,
        ground_truth: str,
        model_response: str,
        criteria: Optional[str] = None,
        _retry: int = 3,
    ) -> dict:
        """
        Score a single model response across four reasoning dimensions.

        Parameters
        ----------
        question       : the original question text
        ground_truth   : reference / expected answer
        model_response : the model's raw output string
        criteria       : optional additional evaluation criteria to append

        Returns
        -------
        dict with keys:
            "scores"          : dict with per-dimension scores (int 0–2 each)
            "composite_score" : float, mean of all four dimension scores (0–2)
            "reasoning"       : str explanation from the judge
            "error"           : str (only present if something went wrong)
        """
        criteria_section = (
            f"Additional criteria: {criteria}\n\n" if criteria else ""
        )
        user_content = JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_response=model_response,
            criteria_section=criteria_section,
        )

        client = self._get_client()

        try:
            # Anthropic API and Azure AI Foundry (AnthropicFoundry): messages.create
            if self.provider in ("anthropic", "azure_foundry"):
                response = client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                raw_content = response.content[0].text

            # OpenAI and Azure OpenAI: use JSON mode via response_format
            elif self.provider in ("openai", "azure_openai"):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                raw_content = response.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {self.provider!r}")

            content = _extract_json(raw_content)
            result = json.loads(content)

            raw_scores = result.get("scores", {})
            scores: Dict[str, int] = {
                dim: _clamp(raw_scores.get(dim, 0)) for dim in _DIMENSIONS
            }
            composite = sum(scores.values()) / len(scores)

            return {
                "scores":          scores,
                "composite_score": round(composite, 4),
                "reasoning":       result.get("reasoning", ""),
            }

        except json.JSONDecodeError as exc:
            return {
                "scores":          {dim: 0 for dim in _DIMENSIONS},
                "composite_score": 0.0,
                "reasoning":       "",
                "error":           f"JSON decode error: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            err_str = str(exc)
            # Retry on rate-limit (429) with exponential backoff before giving up.
            if _retry > 0 and self._is_rate_limit_error(err_str):
                wait = 10 * (4 - _retry)  # 10s, 20s, 30s
                logger.warning(
                    "[LLMJudge] Rate limit on %s (attempt left=%d); sleeping %ds",
                    self.provider, _retry, wait,
                )
                time.sleep(wait)
                return self.score_response(
                    question=question,
                    ground_truth=ground_truth,
                    model_response=model_response,
                    criteria=criteria,
                    _retry=_retry - 1,
                )
            if self._fallback_judge and self._is_auth_error(err_str):
                logger.warning(
                    "[LLMJudge] Auth error on %s (%s); retrying with fallback %s",
                    self.provider, err_str[:120], self._fallback_judge.provider,
                )
                return self._fallback_judge.score_response(
                    question=question,
                    ground_truth=ground_truth,
                    model_response=model_response,
                    criteria=criteria,
                )
            return {
                "scores":          {dim: 0 for dim in _DIMENSIONS},
                "composite_score": 0.0,
                "reasoning":       "",
                "error":           err_str,
            }

    # ------------------------------------------------------------------
    # Batch scoring — serial with rate limiting (legacy / low-volume)
    # ------------------------------------------------------------------

    def score_batch(
        self,
        records: List[dict],
        criteria: Optional[str] = None,
        requests_per_minute: int = 60,
        save_path: Optional[str | Path] = None,
    ) -> List[dict]:
        """
        Score a batch serially with rate limiting.  Prefer score_batch_concurrent()
        for large volumes; prefer score_batch_api() when using the anthropic provider.
        """
        min_interval = 60.0 / requests_per_minute
        results: List[dict] = []
        last_request_time = 0.0

        for i, rec in enumerate(records):
            elapsed = time.time() - last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            score_result = self.score_response(
                question=rec.get("question", ""),
                ground_truth=rec.get("ground_truth", ""),
                model_response=rec.get("model_response", ""),
                criteria=criteria,
            )
            results.append(score_result)
            last_request_time = time.time()

            if (i + 1) % 10 == 0:
                logger.info("[LLMJudge] Scored %d/%d", i + 1, len(records))

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                for rec, score_result in zip(records, results):
                    fh.write(json.dumps({**rec, **score_result}) + "\n")
            logger.info("[LLMJudge] Saved %d records to %s", len(results), save_path)

        return results

    # ------------------------------------------------------------------
    # Judge input helper
    # ------------------------------------------------------------------

    def build_judge_input(self, record: dict) -> Optional[dict]:
        """
        Extract judge input fields from a QA result dict.

        Returns None if the model response is empty (nothing to score).
        """
        question = record.get("question") or record.get("question_text", "")
        ground_truth = record.get("ground_truth", "")
        model_response = (
            record.get("raw_output_text")
            or record.get("raw_output")
            or record.get("model_response", "")
        )
        if isinstance(model_response, list):
            model_response = " ".join(str(x) for x in model_response)
        if not str(model_response).strip():
            return None
        return {
            "question":       question,
            "ground_truth":   ground_truth,
            "model_response": str(model_response),
            "question_id":    record.get("question_id"),
        }

    # ------------------------------------------------------------------
    # Concurrent batch scoring (all providers — recommended for azure_foundry)
    # ------------------------------------------------------------------

    def score_batch_concurrent(
        self,
        records: List[dict],
        criteria: Optional[str] = None,
        concurrency: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> List[dict]:
        """
        Score records concurrently using a thread pool.

        Works with all providers (anthropic, openai, azure_foundry, azure_openai).
        Recommended for azure_foundry where the Batch API is unavailable.

        Parameters
        ----------
        records     : list of dicts with keys "question", "ground_truth", "model_response"
        criteria    : shared additional evaluation criteria
        concurrency : number of parallel requests (default 20; tune to provider rate limit)
        save_path   : if provided, save scored results as JSONL

        Returns
        -------
        List of score dicts in the same order as input.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[dict] = [{}] * len(records)

        def _score_one(args):
            idx, rec = args
            return idx, self.score_response(
                question=rec.get("question", ""),
                ground_truth=rec.get("ground_truth", ""),
                model_response=rec.get("model_response", ""),
                criteria=criteria,
            )

        logger.info("[LLMJudge] concurrent scoring: %d records, concurrency=%d", len(records), concurrency)
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_score_one, (i, r)): i for i, r in enumerate(records)}
            for future in as_completed(futures):
                try:
                    idx, score = future.result()
                    results[idx] = score
                except Exception as exc:  # noqa: BLE001
                    idx = futures[future]
                    results[idx] = {
                        "scores": {dim: 0 for dim in _DIMENSIONS},
                        "composite_score": 0.0,
                        "reasoning": "",
                        "error": str(exc),
                    }
                done += 1
                if done % 100 == 0 or done == len(records):
                    logger.info("[LLMJudge] %d / %d scored", done, len(records))

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                for rec, score_result in zip(records, results):
                    fh.write(json.dumps({**rec, **score_result}) + "\n")
            logger.info("[LLMJudge] Saved %d records to %s", len(results), save_path)

        return results

    # ------------------------------------------------------------------
    # Anthropic Message Batches API (anthropic provider only — 50% cheaper)
    # ------------------------------------------------------------------

    def score_batch_api(
        self,
        records: List[dict],
        criteria: Optional[str] = None,
        poll_interval: int = 30,
        save_path: Optional[str | Path] = None,
    ) -> List[dict]:
        """
        Score records using the Anthropic Message Batches API.

        Requires provider="anthropic" and ANTHROPIC_API_KEY.
        Submits all records as a single batch (up to 100k requests).
        50% cheaper than standard API; results typically available within minutes.

        Parameters
        ----------
        records       : list of dicts with keys "question", "ground_truth", "model_response"
        criteria      : shared additional evaluation criteria
        poll_interval : seconds between status polls (default 30)
        save_path     : if provided, save scored results as JSONL

        Returns
        -------
        List of score dicts in the same order as input.
        """
        if self.provider != "anthropic":
            raise ValueError(
                f"score_batch_api() requires provider='anthropic', got {self.provider!r}. "
                "Use score_batch_concurrent() for other providers."
            )

        client = self._get_client()

        batch_requests = []
        for i, rec in enumerate(records):
            criteria_section = f"Additional criteria: {criteria}\n\n" if criteria else ""
            user_content = JUDGE_USER_TEMPLATE.format(
                question=rec.get("question", ""),
                ground_truth=rec.get("ground_truth", ""),
                model_response=rec.get("model_response", ""),
                criteria_section=criteria_section,
            )
            batch_requests.append({
                "custom_id": str(i),
                "params": {
                    "model": self.model,
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "system": JUDGE_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_content}],
                },
            })

        for attempt in range(5):
            try:
                batch = client.messages.batches.create(requests=batch_requests)
                break
            except Exception as exc:
                wait = 30 * (attempt + 1)
                logger.warning("[LLMJudge] Batch submit attempt %d failed (%s); retrying in %ds", attempt + 1, exc, wait)
                print(f"[LLMJudge] Batch submit attempt {attempt + 1} failed: {exc}; retrying in {wait}s …")
                time.sleep(wait)
        else:
            raise RuntimeError("Anthropic Batch API: all submit attempts failed")

        batch_id = batch.id
        logger.info("[LLMJudge] Batch API submitted: id=%s, %d requests", batch_id, len(batch_requests))
        print(f"[LLMJudge] Batch submitted: {batch_id} ({len(batch_requests)} requests)")

        while True:
            status = client.messages.batches.retrieve(batch_id)
            counts = status.request_counts
            logger.info(
                "[LLMJudge] Batch %s: processing_status=%s counts=%s",
                batch_id, status.processing_status, counts,
            )
            print(
                f"[LLMJudge] Batch {batch_id}: {status.processing_status} | "
                f"succeeded={counts.succeeded} errored={counts.errored} "
                f"processing={counts.processing}"
            )
            if status.processing_status == "ended":
                break
            time.sleep(poll_interval)

        raw: dict = {}
        for result in client.messages.batches.results(batch_id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                raw_content = result.result.message.content[0].text
                try:
                    content = _extract_json(raw_content)
                    parsed = json.loads(content)
                    raw_scores = parsed.get("scores", {})
                    scores = {dim: _clamp(raw_scores.get(dim, 0)) for dim in _DIMENSIONS}
                    composite = sum(scores.values()) / len(scores)
                    raw[cid] = {
                        "scores": scores,
                        "composite_score": round(composite, 4),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                except json.JSONDecodeError as exc:
                    raw[cid] = {
                        "scores": {dim: 0 for dim in _DIMENSIONS},
                        "composite_score": 0.0,
                        "reasoning": "",
                        "error": f"JSON decode error: {exc}",
                    }
            else:
                raw[cid] = {
                    "scores": {dim: 0 for dim in _DIMENSIONS},
                    "composite_score": 0.0,
                    "reasoning": "",
                    "error": str(result.result),
                }

        _missing = {"scores": {dim: 0 for dim in _DIMENSIONS}, "composite_score": 0.0, "reasoning": "", "error": "missing"}
        results = [raw.get(str(i), _missing) for i in range(len(records))]
        logger.info("[LLMJudge] Batch API complete: %d results retrieved", len(results))

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                for rec, score_result in zip(records, results):
                    fh.write(json.dumps({**rec, **score_result}) + "\n")
            logger.info("[LLMJudge] Saved %d records to %s", len(results), save_path)

        return results

    # ------------------------------------------------------------------
    # Aggregate judge scores
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_scores(scored_records: List[dict]) -> dict:
        """
        Compute summary statistics over LLM judge scores.

        Parameters
        ----------
        scored_records : list of score dicts (from score_response / score_batch)

        Returns
        -------
        dict with:
            "composite_mean"   : float — mean composite score across all records
            "n"                : int — number of successfully scored records
            "n_errors"         : int — number of records with errors
            "per_dimension"    : {dim_name → {"mean": float, "distribution": {0: int, 1: int, 2: int}}}
        """
        valid = [r for r in scored_records if "error" not in r]
        errors = [r for r in scored_records if "error" in r]

        if not valid:
            return {
                "composite_mean": 0.0,
                "n":              0,
                "n_errors":       len(errors),
                "per_dimension":  {
                    dim: {"mean": 0.0, "distribution": {0: 0, 1: 0, 2: 0}}
                    for dim in _DIMENSIONS
                },
            }

        composite_scores = [r["composite_score"] for r in valid]
        composite_mean = sum(composite_scores) / len(composite_scores)

        per_dimension: Dict[str, dict] = {}
        for dim in _DIMENSIONS:
            dim_scores = [r["scores"].get(dim, 0) for r in valid]
            dim_mean = sum(dim_scores) / len(dim_scores)
            distribution = {0: 0, 1: 0, 2: 0}
            for s in dim_scores:
                distribution[_clamp(s)] += 1
            per_dimension[dim] = {
                "mean":         round(dim_mean, 4),
                "distribution": distribution,
            }

        return {
            "composite_mean": round(composite_mean, 4),
            "n":              len(valid),
            "n_errors":       len(errors),
            "per_dimension":  per_dimension,
        }
