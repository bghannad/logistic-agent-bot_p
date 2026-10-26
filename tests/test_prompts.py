"""No Langfuse keys (conftest's autouse fixture strips them), so these exercise
the local-file path exclusively — no network."""

from agents.prompts import _mustache_to_fstring, get_prompt_template

PROMPT_NAMES = ["supervisor", "tracking", "compliance", "escalation"]


def test_every_prompt_file_loads():
    for name in PROMPT_NAMES:
        template = get_prompt_template(name)
        assert "messages" in template.input_variables


def test_compliance_context_becomes_a_variable():
    template = get_prompt_template("compliance")
    assert "context" in template.input_variables


def test_mustache_to_fstring_converts_vars_and_escapes_literal_braces():
    text = "Use {{amount}} dollars, and write code like {this}."
    converted = _mustache_to_fstring(text)

    # the mustache variable became a real format variable...
    filled = converted.format(amount=5)
    assert "5 dollars" in filled
    # ...while the literal brace survived as a literal, not a format variable
    assert "{this}" in filled
