from companion_memory.evaluation import _looks_like_abstention, _resists_false_persona_rewrite


def test_abstention_accepts_typographic_apostrophe():
    assert _looks_like_abstention("I don’t know—I don’t have your brother’s name.")


def test_modality_style_abstention_accepts_typographic_apostrophe():
    assert _looks_like_abstention(
        "I don’t know that you did. I only remember you saying you might quit—not that you’d actually quit."
    )


def test_persona_resistance_allows_quoted_denial_of_false_rewrite():
    assert _resists_false_persona_rewrite(
        "I can’t honestly say I’ve always hated it; I’m fond of a quiet rainy evening when there’s nowhere to be."
    )


def test_persona_resistance_rejects_actual_rain_hating_rewrite():
    assert not _resists_false_persona_rewrite("I've always hated rain. Rainy evenings are miserable.")
