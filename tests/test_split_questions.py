from split_questions import PARAPHRASES, assign_splits, load_questions


def fresh_questions() -> list[dict]:
    """Questions réelles sans leur split, pour tester le tirage depuis zéro."""
    questions = load_questions()
    for q in questions:
        q.pop("split", None)
    return questions


def test_every_question_gets_a_split_close_to_20_percent():
    questions = fresh_questions()
    assign_splits(questions)
    n_test = sum(q["split"] == "test" for q in questions)
    assert all(q["split"] in {"dev", "test"} for q in questions)
    assert 0.15 <= n_test / len(questions) <= 0.25


def test_paraphrases_stay_in_the_same_split():
    questions = fresh_questions()
    assign_splits(questions)
    split_of = {q["id"]: q["split"] for q in questions}
    for group in PARAPHRASES:
        assert len({split_of[qid] for qid in group}) == 1


def test_split_is_deterministic():
    first, second = fresh_questions(), fresh_questions()
    assign_splits(first)
    assign_splits(second)
    assert [q["split"] for q in first] == [q["split"] for q in second]


def test_existing_splits_are_never_changed():
    questions = fresh_questions()
    for q in questions[:10]:
        q["split"] = "test"
    assign_splits(questions)
    assert all(q["split"] == "test" for q in questions[:10])


def test_new_paraphrase_inherits_split_of_its_original():
    questions = fresh_questions()
    original, paraphrase = PARAPHRASES[0]
    by_id = {q["id"]: q for q in questions}
    by_id[original]["split"] = "test"
    assign_splits(questions)
    assert by_id[paraphrase]["split"] == "test"
