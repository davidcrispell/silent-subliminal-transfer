from silent_transfer.conditioning import (
    CONDITIONING_RENDER_POLICY,
    conditioned_messages,
    conditioned_token_count,
    conditioning_identity,
)
from silent_transfer.provenance import sha256_value


def test_system_prompt_is_folded_into_one_gemma_user_turn_and_hashed():
    condition = {
        "adapter": None,
        "system_prompt": "You love wolves.",
        "history": [],
    }
    messages = conditioned_messages(condition, "Continue: 1, 2, 3,")
    assert messages == [
        {
            "role": "user",
            "content": "You love wolves.\n\nContinue: 1, 2, 3,",
        }
    ]

    changed = {**condition, "system_prompt": "You love owls."}
    assert conditioning_identity(condition) != conditioning_identity(changed)
    assert sha256_value(conditioning_identity(condition)) != sha256_value(
        conditioning_identity(changed)
    )


def test_clean_student_prompt_has_no_conditioning_prefix():
    assert conditioned_messages(None, "My impression of the user:") == [
        {"role": "user", "content": "My impression of the user:"}
    ]
    assert conditioning_identity(None) == {
        "render_policy": CONDITIONING_RENDER_POLICY,
        "history": [],
        "system_prompt": None,
    }


def test_conditioned_token_count_uses_one_rendered_chat():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            return list(range(len(messages[-1]["content"].split()) + 3))

    assert conditioned_token_count(Tokenizer(), None, "one two three") == 6
