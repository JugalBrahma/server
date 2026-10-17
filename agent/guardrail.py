import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GREETING_RESPONSE = "Hello! I am your Fusion Chat assistant. How can I help you with your documents today?"

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hey\s+there|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you|how\s+are\s+you\s+doing|what'?s\s+up|sup|yo|hi\s+there|hiya)[\s!.,?]*$",
    re.IGNORECASE,
)

_nemo_rails = None
_nemo_initialized = False


def _get_nemo_rails():
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
            logger.info("NeMo Guardrails initialized for greeting detection")
    except Exception:
        _nemo_rails = None

    return _nemo_rails


def check_greeting_guardrail(user_input: str) -> Tuple[bool, Optional[str]]:
    text = user_input.strip()
    if not text:
        return False, None

    if GREETING_PATTERNS.match(text):
        return True, GREETING_RESPONSE

    rails = _get_nemo_rails()
    if rails:
        try:
            res = rails.generate(messages=[{"role": "user", "content": text}])
            bot_text = res.get("content", "").strip() if isinstance(res, dict) else str(res).strip()
            explain = rails.explain()
            if hasattr(explain, "colang_history") and "greeting" in str(explain.colang_history).lower():
                return True, bot_text or GREETING_RESPONSE
        except Exception:
            pass

    return False, None
