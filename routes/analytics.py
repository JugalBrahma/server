"""
LangSmith Analytics & Performance Telemetry Route
- Collects real-time latency, token usage, and trace logs from LangSmith
- Provides aggregated metrics (average latency, token distribution, recent traces)
"""

from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from firebase_admin import auth
from langsmith import Client

from config.firebase.database import get_firestore

logger = logging.getLogger("analytics")

router = APIRouter()

CACHE_TTL_SECONDS = 5
_analytics_cache: Dict[str, Any] = {"key": "", "timestamp": 0, "data": None}


def _get_langsmith_client() -> Optional[Client]:
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.warning("LANGSMITH_API_KEY is not set.")
        return None
    try:
        return Client(api_key=api_key.strip('"\''))
    except Exception as e:
        logger.error("Failed to initialize LangSmith client: %s", e)
        return None


def _extract_optional_user_id(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        get_firestore()
        token = authorization.split(" ")[1]
        decoded = auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception:
        return None


def _format_relative_time(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        seconds = int((now - dt).total_seconds())
        if seconds < 60:
            return f"{max(1, seconds)}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"
    except Exception:
        return ""


def _extract_query_text(inputs: Any) -> str:
    if not inputs or not isinstance(inputs, dict):
        return "Agent Query"
    for key in ["user_input", "question", "query", "input", "messages"]:
        if key in inputs:
            val = inputs[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                first = val[0]
                return first.get("content", str(first)) if isinstance(first, dict) else str(first)
    return "Agent Query"


def _empty_payload(project_name: str) -> Dict[str, Any]:
    return {
        "avg_latency": 0.0,
        "min_latency": 0.0,
        "max_latency": 0.0,
        "p95_latency": 0.0,
        "retriever_latency": 0.0,
        "llm_latency": 0.0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "avg_tokens_per_run": 0,
        "total_runs": 0,
        "success_rate": 100.0,
        "project_name": project_name,
        "recent_traces": [],
        "latency_trend": [],
    }


def _matches_folder(run: Any, folder_id: str) -> bool:
    meta = run.extra.get("metadata", {}) if run.extra else {}
    tags = run.tags or []
    inputs = run.inputs or {}
    return (
        inputs.get("folder_id") == folder_id
        or meta.get("folder_id") == folder_id
        or f"folder_{folder_id}" in tags
        or f"folder:{folder_id}" in tags
        or folder_id in str(inputs)
        or folder_id in str(meta)
        or any(folder_id in str(t) for t in tags)
    )


def _calc_latencies(chain_runs: List[Any], all_runs: List[Any]) -> Dict[str, float]:
    latencies = [r.latency for r in chain_runs if r.latency is not None and r.latency > 0]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    min_lat = min(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0

    p95_lat = 0.0
    if latencies:
        sorted_lats = sorted(latencies)
        idx = int(len(sorted_lats) * 0.95)
        p95_lat = sorted_lats[min(idx, len(sorted_lats) - 1)]

    retriever_lats = [
        r.latency for r in all_runs
        if (r.run_type == "retriever" or "Retrieval" in (r.name or "")) and r.latency is not None
    ]
    llm_lats = [
        r.latency for r in all_runs
        if (r.run_type == "llm" or "Chat" in (r.name or "")) and r.latency is not None
    ]

    avg_retriever = (
        sum(retriever_lats) / len(retriever_lats)
        if retriever_lats else (avg_lat * 0.35 if chain_runs else 0.0)
    )
    avg_llm = (
        sum(llm_lats) / len(llm_lats)
        if llm_lats else (avg_lat * 0.65 if chain_runs else 0.0)
    )

    return {
        "avg_latency": round(avg_lat, 2),
        "min_latency": round(min_lat, 2),
        "max_latency": round(max_lat, 2),
        "p95_latency": round(p95_lat, 2),
        "retriever_latency": round(avg_retriever, 2),
        "llm_latency": round(avg_llm, 2),
    }


def _calc_tokens(chain_runs: List[Any], all_runs: List[Any]) -> Dict[str, int]:
    total_prompt = 0
    total_completion = 0
    total = 0

    for r in chain_runs:
        p = r.prompt_tokens or 0
        c = r.completion_tokens or 0
        t = r.total_tokens or (p + c)
        total_prompt += p
        total_completion += c
        total += t

    if total == 0:
        for r in all_runs:
            if r.run_type == "llm":
                p = r.prompt_tokens or 0
                c = r.completion_tokens or 0
                t = r.total_tokens or (p + c)
                total_prompt += p
                total_completion += c
                total += t

    total_runs = len(chain_runs)
    avg_tokens = int(total / total_runs) if total_runs > 0 else 0

    return {
        "total_tokens": total,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "avg_tokens_per_run": avg_tokens,
    }


def _build_traces(chain_runs: List[Any]) -> Tuple[List[Dict], List[Dict]]:
    recent_traces = []
    for r in chain_runs[:10]:
        p = r.prompt_tokens or 0
        c = r.completion_tokens or 0
        t = r.total_tokens or (p + c)

        if t == 0 and r.extra and "metadata" in r.extra:
            usage = r.extra["metadata"].get("usage_metadata", {})
            p = usage.get("input_tokens", 0)
            c = usage.get("output_tokens", 0)
            t = usage.get("total_tokens", p + c)

        model_name = r.extra.get("metadata", {}).get("ls_model_name") if r.extra else None

        recent_traces.append({
            "id": str(r.id),
            "name": r.name or "RAG Agent Query",
            "run_type": r.run_type or "chain",
            "latency": round(r.latency or 0.0, 2),
            "total_tokens": t,
            "prompt_tokens": p,
            "completion_tokens": c,
            "status": r.status or "success",
            "query": _extract_query_text(r.inputs),
            "timestamp": _format_relative_time(r.start_time),
            "model_name": model_name,
        })

    latency_trend = [
        # pyrefly: ignore[no-matching-overload]
        {"label": f"Q{i+1}", "latency": round(t["latency"], 2), "total_tokens": t["total_tokens"]}
        for i, t in enumerate(reversed(recent_traces[:8]))
    ]
    return recent_traces, latency_trend


# ---------- route ----------

@router.get("/analytics")
@router.get("/analytics/")
@router.get("/analytics/{folder_id}")
@router.get("/analytics/{folder_id}/")
def get_folder_analytics(
    folder_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """Fetches real-time telemetry from LangSmith for the active project."""
    user_id = _extract_optional_user_id(authorization)
    now = time.time()
    cache_key = f"{folder_id or 'all'}_{user_id or 'anon'}"

    if (
        _analytics_cache["key"] == cache_key
        and (now - _analytics_cache["timestamp"]) < CACHE_TTL_SECONDS
        and _analytics_cache["data"]
    ):
        return _analytics_cache["data"]

    client = _get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT", "fusion-chat")
    if not client:
        return _empty_payload(project_name)

    try:
        chain_runs = list(client.list_runs(project_name=project_name, run_type="chain", limit=50))
        all_runs = list(client.list_runs(project_name=project_name, limit=100))

        if folder_id:
            chain_runs = [r for r in chain_runs if _matches_folder(r, folder_id)]

        target_run_ids = {str(r.id) for r in chain_runs}
        if folder_id and target_run_ids:
            all_runs = [
                r for r in all_runs
                if str(r.id) in target_run_ids
                or (r.parent_run_id and str(r.parent_run_id) in target_run_ids)
                or (r.trace_id and str(r.trace_id) in target_run_ids)
            ]

        total_runs = len(chain_runs)
        success_count = sum(1 for r in chain_runs if (r.status or "").lower() == "success")
        success_rate = (success_count / total_runs * 100.0) if total_runs > 0 else 100.0

        recent_traces, latency_trend = _build_traces(chain_runs)

        payload = {
            **_calc_latencies(chain_runs, all_runs),
            **_calc_tokens(chain_runs, all_runs),
            "total_runs": total_runs,
            "success_rate": round(success_rate, 1),
            "project_name": project_name,
            "recent_traces": recent_traces,
            "latency_trend": latency_trend,
        }

        _analytics_cache["key"] = cache_key
        _analytics_cache["timestamp"] = now
        _analytics_cache["data"] = payload
        return payload

    except Exception as e:
        logger.exception("Error fetching analytics from LangSmith: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch LangSmith analytics: {e}")
