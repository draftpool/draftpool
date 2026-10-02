from app.main import ANIMATE_GUARD_SUGGESTION, EDIT_GUARD, with_guard

# with_guard itself is still exercised through EDIT_GUARD, which instruct
# edits still apply mechanically. ANIMATE_GUARD_SUGGESTION is no longer passed
# through with_guard by the app -- the /animate endpoint sends the motion
# prompt as written -- but the helper's behaviour is generic, so it is used
# here too rather than inventing a second guard string just for the test.


def test_guard_is_appended_as_its_own_sentence():
    assert with_guard("hair drifts in the wind", ANIMATE_GUARD_SUGGESTION, "nothing else") == (
        "hair drifts in the wind. " + ANIMATE_GUARD_SUGGESTION
    )
    # an existing full stop or trailing space must not double up
    assert with_guard("slow push in. ", ANIMATE_GUARD_SUGGESTION, "nothing else") == (
        "slow push in. " + ANIMATE_GUARD_SUGGESTION
    )


def test_guard_is_skipped_when_the_prompt_already_says_it():
    # the user phrasing it themselves wins; the guard is only a default
    assert with_guard(
        "hair moves, nothing else changes", ANIMATE_GUARD_SUGGESTION, "nothing else"
    ) == "hair moves, nothing else changes"
    assert with_guard("remove her hat, rest UNCHANGED", EDIT_GUARD, "unchanged") == (
        "remove her hat, rest UNCHANGED"
    )


def test_guard_does_not_stack_when_applied_twice():
    # a retried or re-queued job must not accumulate guards
    once = with_guard("she blinks", ANIMATE_GUARD_SUGGESTION, "nothing else", "unchanged")
    assert with_guard(once, ANIMATE_GUARD_SUGGESTION, "nothing else", "unchanged") == once
