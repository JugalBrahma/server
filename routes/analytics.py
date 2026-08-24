"""
LangSmith Analytics & Performance Telemetry Route
- Collects real-time latency, token usage, and trace logs from LangSmith
- Provides aggregated metrics (average latency, token distribution, recent traces)
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from dotenv import load_dotenv

load_dotenv()

from config.firebase import get_current_user_id

logger = logging.getLogger("analytics")

router = APIRouter()

# In-memory simple cache to avoid spamming LangSmith API on rapid refreshes
_analytics_cache: Dict[str, Any] = {
    "timestamp": 0,
    "data": None,
}
CACHE_TTL_SECONDS = 5


def _get_langsmith_client():
    """Initializes LangSmith client safely."""
    try:
        from langsmith import Client
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            logger.warning("LANGSMITH_API_KEY is not set.")
            return None
        return Client(api_key=api_key.strip('"\''))
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith client: {e}")
        return None


def _format_relative_time(dt: Optional[datetime]) -> str:
    """Formats datetime into human-friendly relative string (e.g. 5m ago)."""
    if not dt:
        return ""
    try:
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"{max(1, seconds)}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"
    except Exception:
        return ""


def _extract_query_text(inputs: Any) -> str:
    """Extracts clean query text from run inputs dictionary."""
    if not inputs or not isinstance(inputs, dict):
        return "Agent Query"
    for key in ["user_input", "question", "query", "input", "messages"]:
        if key in inputs:
            val = inputs[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict):
                    return first.get("content", str(first))
                return str(first)
    return "Agent Query"


def _extract_optional_user_id(authorization: Optional[str] = None) -> Optional[str]:
    """Safely extracts user_id from authorization header without throwing 401."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        from firebase_admin import auth
        from config.firebase.database import get_firestore
        get_firestore()
        token = authorization.split(" ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token.get("uid")
    except Exception:
        return None


@router.get("/analytics")
@router.get("/analytics/")
@router.get("/analytics/{folder_id}")
@router.get("/analytics/{folder_id}/")
async def get_folder_analytics(
    folder_id: Optional[str] = None,
    authorization: Optional[str] = Header(None),
):
    """
    Fetches real-time telemetry from LangSmith for the active project.
    Returns:
    - Average latency, min/max/p95 latency
    - Pinecone retrieval vs LLM synthesis latency breakdown
    - Total, prompt, and completion token usage
    - Recent trace history
    """
    user_id = _extract_optional_user_id(authorization)
    current_time = time.time()
    cache_key = f"{folder_id or 'all'}_{user_id or 'anon'}"

    # Return cached data if valid
    if (
        _analytics_cache.get("key") == cache_key
        and (current_time - _analytics_cache.get("timestamp", 0)) < CACHE_TTL_SECONDS
        and _analytics_cache.get("data")
    ):
        return _analytics_cache["data"]

    client = _get_langsmith_client()
    project_name = os.getenv("LANGSMITH_PROJECT", "fusion-chat")

    if not client:
        # Fallback response if LangSmith is not configured
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

    try:
        logger.info(f"Fetching LangSmith analytics for folder_id: {folder_id}, project: {project_name}")
        # Fetch latest runs for the project (LangSmith max allowed limit is 100)
        # Chain runs (top-level agent executions)
        chain_runs = list(
            client.list_runs(
                project_name=project_name,
                run_type="chain",
                limit=50,
            )
        )

        # All runs (including retriever and LLM sub-runs)
        all_runs = list(
            client.list_runs(
                project_name=project_name,
                limit=100,
            )
        )

        # Filter by folder_id if requested
        if folder_id:
            filtered_chain_runs = []
            for r in chain_runs:
                meta = r.extra.get("metadata", {}) if r.extra else {}
                tags = r.tags or []
                inputs = r.inputs or {}
                
                is_match = (
                    inputs.get("folder_id") == folder_id
                    or meta.get("folder_id") == folder_id
                    or f"folder_{folder_id}" in tags
                    or f"folder:{folder_id}" in tags
                    or folder_id in str(inputs)
                    or folder_id in str(meta)
                    or any(folder_id in str(t) for t in tags)
                )
                if is_match:
                    filtered_chain_runs.append(r)
            target_chain_runs = filtered_chain_runs
        else:
            target_chain_runs = chain_runs

        logger.info(f"Found {len(target_chain_runs)} matching chain runs for folder {folder_id}")

        # Extract trace/run IDs of target chains to isolate sub-runs (retriever & LLM)
        target_run_ids = {str(r.id) for r in target_chain_runs}
        if folder_id and target_run_ids:
            target_all_runs = [
                r for r in all_runs
                if str(r.id) in target_run_ids
                or (r.parent_run_id and str(r.parent_run_id) in target_run_ids)
                or (r.trace_id and str(r.trace_id) in target_run_ids)
            ]
        else:
            target_all_runs = all_runs

        # Compute Latency Metrics
        latencies = [
            r.latency for r in target_chain_runs if r.latency is not None and r.latency > 0
        ]
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        
        # P95 latency
        if latencies:
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        else:
            p95_latency = 0.0

        # Component Latency breakdown from sub-runs (retrieval vs LLM) for this folder
        retriever_latencies = [
            r.latency
            for r in target_all_runs
            if (r.run_type == "retriever" or "Retrieval" in (r.name or ""))
            and r.latency is not None
        ]
        llm_latencies = [
            r.latency
            for r in target_all_runs
            if (r.run_type == "llm" or "Chat" in (r.name or ""))
            and r.latency is not None
        ]

        avg_retriever_latency = (
            sum(retriever_latencies) / len(retriever_latencies)
            if retriever_latencies
            else (avg_latency * 0.35 if target_chain_runs else 0.0)
        )
        avg_llm_latency = (
            sum(llm_latencies) / len(llm_latencies)
            if llm_latencies
            else (avg_latency * 0.65 if target_chain_runs else 0.0)
        )

        # Compute Token Metrics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for r in target_chain_runs:
            p_tok = r.prompt_tokens or 0
            c_tok = r.completion_tokens or 0
            t_tok = r.total_tokens or (p_tok + c_tok)
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_tokens += t_tok

        # Fallback check on all runs if chain runs have 0 token summary
        if total_tokens == 0:
            for r in all_runs:
                if r.run_type == "llm":
                    p_tok = r.prompt_tokens or 0
                    c_tok = r.completion_tokens or 0
                    t_tok = r.total_tokens or (p_tok + c_tok)
                    total_prompt_tokens += p_tok
                    total_completion_tokens += c_tok
                    total_tokens += t_tok

        total_runs_count = len(target_chain_runs)
        avg_tokens_per_run = (
            int(total_tokens / total_runs_count) if total_runs_count > 0 else 0
        )

        # Success rate
        success_count = sum(
            1 for r in target_chain_runs if (r.status or "").lower() == "success"
        )
        success_rate = (
            (success_count / total_runs_count * 100.0) if total_runs_count > 0 else 100.0
        )

        # Build Recent Traces List
        recent_traces: List[Dict[str, Any]] = []
        for r in target_chain_runs[:10]:
            p_tok = r.prompt_tokens or 0
            c_tok = r.completion_tokens or 0
            t_tok = r.total_tokens or (p_tok + c_tok)
            
            # If chain run tokens are 0, check child LLM tokens if any
            if t_tok == 0 and r.extra and "metadata" in r.extra:
                usage = r.extra["metadata"].get("usage_metadata", {})
                p_tok = usage.get("input_tokens", 0)
                c_tok = usage.get("output_tokens", 0)
                t_tok = usage.get("total_tokens", p_tok + c_tok)

            query_text = _extract_query_text(r.inputs)

            recent_traces.append(
                {
                    "id": str(r.id),
                    "name": r.name or "RAG Agent Query",
                    "run_type": r.run_type or "chain",
                    "latency": round(r.latency or 0.0, 2),
                    "total_tokens": t_tok,
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "status": r.status or "success",
                    "query": query_text,
                    "timestamp": _format_relative_time(r.start_time),
                    "model_name": (
                        r.extra.get("metadata", {}).get("ls_model_name")
                        if r.extra
                        else None
                    ),
                }
            )

        # Latency Trend
        latency_trend = [
            {
                "label": f"Q{i+1}",
                "latency": round(t["latency"], 2),
                "total_tokens": t["total_tokens"],
            }
            for i, t in enumerate(reversed(recent_traces[:8]))
        ]

        payload = {
            "avg_latency": round(avg_latency, 2),
            "min_latency": round(min_latency, 2),
            "max_latency": round(max_latency, 2),
            "p95_latency": round(p95_latency, 2),
            "retriever_latency": round(avg_retriever_latency, 2),
            "llm_latency": round(avg_llm_latency, 2),
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "avg_tokens_per_run": avg_tokens_per_run,
            "total_runs": total_runs_count,
            "success_rate": round(success_rate, 1),
            "project_name": project_name,
            "recent_traces": recent_traces,
            "latency_trend": latency_trend,
        }

        # Update cache
        _analytics_cache["key"] = cache_key
        _analytics_cache["timestamp"] = current_time
        _analytics_cache["data"] = payload

        return payload

    except Exception as e:
        logger.exception(f"Error fetching analytics from LangSmith: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch LangSmith analytics: {str(e)}"
        )
