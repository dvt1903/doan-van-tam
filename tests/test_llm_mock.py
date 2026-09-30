import json
from src.llm import _mock_invoke


def test_mock_summary_returns_json():
    out = _mock_invoke("TASK=SUMMARY\n[S1] demo.pdf - trang 1\nRAG kết hợp retrieval và generation.")
    obj = json.loads(out)
    assert obj["summary"]
    assert obj["key_points"]
