import pytest
from pydantic import ValidationError
from src.schemas import QuizItem


def test_quiz_item_accepts_four_options():
    item = QuizItem(question="Q?", options=["A","B","C","D"], correct_index=2, explanation="Because")
    assert item.correct_index == 2


def test_quiz_item_rejects_bad_index():
    with pytest.raises(ValidationError):
        QuizItem(question="Q?", options=["A","B","C","D"], correct_index=4, explanation="Because")
