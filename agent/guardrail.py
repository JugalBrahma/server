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

def check_greeting_guardrail(user_input: str) -> Tuple[bool, Optional[str]]:
    text = user_input.strip()
    if not text:
        return False, None

    if GREETING_PATTERNS.match(text):
        return True, GREETING_RESPONSE

    return False, None
