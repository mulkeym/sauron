import json
import pytest
from scripts.evaluate_technical import CASES_PATH, evaluate_case, make_document

CASES=json.loads(CASES_PATH.read_text())['cases']


@pytest.mark.parametrize('case',CASES,ids=[c['name'] for c in CASES])
def test_synthetic_technical_case(case):
    result=evaluate_case(case)
    assert result['passed'], result
