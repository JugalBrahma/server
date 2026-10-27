import pytest

from agent.guardrail import GREETING_RESPONSE, check_greeting_guardrail


@pytest.mark.parametrize("text", ["hi", "Hello!", "  hey there  ", "Good Morning.", "what's up?"])
def test_greetings_are_intercepted(text):
    assert check_greeting_guardrail(text) == (True, GREETING_RESPONSE)


@pytest.mark.parametrize("text", ["", "   ", "hello, summarize chapter 2", "What is RAG?"])
def test_other_input_passes_through(text):
    assert check_greeting_guardrail(text) == (False, None)
