from pathlib import Path

from scripts.generate_model_cards import generate_cards


def test_model_cards_are_generated_from_reports(tmp_path: Path) -> None:
    stance = tmp_path / "final.md"
    outcome = tmp_path / "outcome.md"
    stance.write_text("| Frozen DistilBERT | 0.6000 | [x] | 0.5000 | [x] | 0.5500 |", encoding="utf-8")
    outcome.write_text("Not evaluated: no artifact", encoding="utf-8")
    stance_card, outcome_card = tmp_path / "stance.md", tmp_path / "outcome_card.md"
    generate_cards(stance, outcome, stance_card, outcome_card)
    assert "0.6000" in stance_card.read_text(encoding="utf-8")
    assert "English-only" in stance_card.read_text(encoding="utf-8")
    assert "No real outcome metrics" in outcome_card.read_text(encoding="utf-8")
