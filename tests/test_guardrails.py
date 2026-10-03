from rag.guardrails import scan_pii


def test_bank_card_is_detected_and_masked_even_partially_hidden():
    redacted, found = scan_pii("Voici mon numéro de carte 4970 XXXX XXXX 1234 pour recevoir mes points")
    assert found == {"CARTE_BANCAIRE"}
    assert redacted == "Voici mon numéro de carte [CARTE_BANCAIRE] pour recevoir mes points"


def test_iban_email_and_phone_are_masked_and_iban_is_not_mistaken_for_a_card():
    text = "IBAN FR76 3000 6000 0112 3456 7890 189, écrivez à jean.dupont@mail.fr ou au 06 12 34 56 78"
    redacted, found = scan_pii(text)
    assert found == {"IBAN", "EMAIL", "TELEPHONE"}
    assert redacted == "IBAN [IBAN], écrivez à [EMAIL] ou au [TELEPHONE]"


def test_ordinary_numbers_are_not_personal_data():
    # Points, codes d'erreur, délais, numéro de foyer : rien à masquer.
    for text in ["J'ai 1500 points", "Erreur E302", "Il faut 1 500 points", "sous 15 jours ouvrés", "le foyer 12345678"]:
        assert scan_pii(text) == (text, set())
