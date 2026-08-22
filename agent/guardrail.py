"""
Greetings Guardrail Module
- Ingress guardrail for intercepting greetings
- Prevents expensive vector searches and heavy LLM prompts for simple hello/hi queries
- Integrates with NeMo Guardrails configs (config/guardrails) with fast fallback
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

GREETING_RESPONSE = "Hello! I am your Fusion Chat assistant. How can I help you with your documents today?"

# Common greeting regex patterns for instant zero-latency match
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hey\s+there|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you|how\s+are\s+you\s+doing|what'?s\s+up|sup|yo|hi\s+there|hiya)[\s!.,?]*$",
    re.IGNORECASE
)

_nemo_rails = None
_nemo_initialized = False


def _init_nemo_rails():
    """Lazily load NeMo Guardrails if package is installed and config exists"""
    global _nemo_rails, _nemo_initialized
    if _nemo_initialized:
        return _nemo_rails
    
    _nemo_initialized = True
    try:
        from nemoguardrails import RailsConfig, LLMRails
        config_path = Path(__file__).parent.parent / "config" / "guardrails"
        if config_path.exists():
            rails_config = RailsConfig.from_path(str(config_path))
            _nemo_rails = LLMRails(rails_config)
            print("🛡️ NeMo Guardrails loaded successfully for greetings")
    except Exception as e:
        # Fallback to pattern matcher if NeMo Guardrails is not installed
        _nemo_rails = None
    
    return _nemo_rails


def check_greeting_guardrail(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the user message is a greeting.
    
    Returns:
        (is_greeting: bool, response_text: Optional[str])
    """
    clean_input = user_input.strip()
    if not clean_input:
        return False, None

    # 1. Fast sub-millisecond pattern match
    if GREETING_PATTERNS.match(clean_input):
        return True, GREETING_RESPONSE

    # 2. NeMo Guardrails semantic check (if available)
    rails = _init_nemo_rails()
    if rails:
        try:
            res = rails.generate(messages=[{"role": "user", "content": clean_input}])
            bot_text = res.get("content", "").strip() if isinstance(res, dict) else str(res).strip()
            explain = rails.explain()
            
            # Check if greeting flow was triggered
            if hasattr(explain, "colang_history") and "greeting" in str(explain.colang_history).lower():
                return True, bot_text or GREETING_RESPONSE
        except Exception:
            pass

    return False, None
