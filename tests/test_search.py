"""reciprocal_rank_fusion and _tokenize are pure functions — no index, no API
calls, no network."""

from retrieval.search import _tokenize, reciprocal_rank_fusion


def test_rrf_chunk_high_in_both_lists_wins():
    rankings = [["a", "b", "c"], ["a", "c", "b"]]
    fused = reciprocal_rank_fusion(rankings)
    assert fused[0] == "a"


def test_rrf_keeps_every_id_from_both_lists():
    rankings = [["a", "b"], ["c", "d"]]
    fused = reciprocal_rank_fusion(rankings)
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_k_effect_flips_the_winner():
    # "A" has one great rank (0) and one terrible rank (10). "B" has two
    # consistently middling ranks (3 and 3). Filler ids are disjoint between
    # the two lists so they can't accidentally stack up. Worked out by hand:
    # k=1:    A ~= 1/2 + 1/12      = 0.5833   B ~= 1/5 + 1/5     = 0.4
    #   -> A wins: one great signal dominates when k is small.
    # k=1000: A ~= 1/1001 + 1/1011 = 0.001988   B ~= 2/1004 = 0.0019920
    #   -> B wins: large k flattens rank differences, rewarding consistency.
    list1 = ["A", "p1", "p2", "B"]
    list2 = ["q0", "q1", "q2", "B", "q4", "q5", "q6", "q7", "q8", "q9", "A"]

    assert reciprocal_rank_fusion([list1, list2], k=1)[0] == "A"
    assert reciprocal_rank_fusion([list1, list2], k=1000)[0] == "B"


def test_tokenize_lowercases_and_splits_on_non_word_chars():
    assert _tokenize("Hello, World! 123") == ["hello", "world", "123"]
