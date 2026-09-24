from goldjudge.judge.verdict import parse_verdict
from goldjudge.taxonomy import load_taxonomy


def test_parse_clean():
  tax = load_taxonomy()
  txt = '{"root_step": 8, "primary": "r3", "secondary": [{"leaf": "r7", "step": 11, "relation": "downstream"}], "evidence": "e", "correction": "c", "confidence": "high", "flags": {"evaluator_doubt": false, "gold_divergence_step": 8}}'
  v = parse_verdict(txt, tax, n_steps=15)
  assert v.parse_ok and v.root_step == 8 and v.primary == "R3"
  assert v.secondary[0].leaf == "R7" and v.flags.gold_divergence_step == 8


def test_parse_fenced_and_prose():
  tax = load_taxonomy()
  txt = 'Here is my verdict:\n```json\n{"root_step": 2, "primary": "G2", "secondary": [], "evidence": "x", "correction": "y", "confidence": "mid", "flags": {}}\n```\nThanks.'
  v = parse_verdict(txt, tax, n_steps=5)
  assert v.parse_ok and v.primary == "G2" and v.root_step == 2


def test_parse_bad_leaf_and_range():
  tax = load_taxonomy()
  v = parse_verdict('{"root_step": 99, "primary": "Z9"}', tax, n_steps=5)
  assert not v.parse_ok and "Z9" in v.parse_error and "outside" in v.parse_error


def test_parse_garbage():
  v = parse_verdict("no json here")
  assert not v.parse_ok and v.primary == "UNPARSED"


def test_parse_prose_with_code_then_fenced_verdict():
  tax = load_taxonomy()
  txt = ("The evaluator checks:\n```python\n{'key': 'ctrl+j', 'when': 'terminalFocus'}\n```\nand the gold typed "
         '{ "key": "ctrl+j" } in g7.\n\n```json\n{"root_step": 12, "primary": "R3", "secondary": [], '
         '"evidence": "uses {braces} in text", "confidence": "high", "flags": {"evaluator_doubt": false}}\n```')
  v = parse_verdict(txt, tax, n_steps=20)
  assert v.parse_ok and v.root_step == 12 and v.primary == "R3"


def test_parse_unfenced_verdict_after_prose_and_truncation():
  tax = load_taxonomy()
  txt = 'Rule {"key": "x"} checked. Verdict: {"root_step": 3, "primary": "G1"} done.'
  v = parse_verdict(txt, tax, n_steps=5)
  assert v.parse_ok and v.root_step == 3 and v.primary == "G1"
  v = parse_verdict(txt, tax, n_steps=5, truncated=True)
  assert not v.parse_ok and "truncated" in v.parse_error


def test_parse_repairs_invalid_escapes():
  tax = load_taxonomy()
  txt = 'Reasoning with {"key": 1}.\n```json\n{"root_step": 4, "primary": "R3", "correction": "use \\b[aeiou]\\w* and C:\\\\path"}\n```'
  v = parse_verdict(txt, tax, n_steps=10)
  assert v.parse_ok and v.root_step == 4 and "\\w*" in v.correction and "C:\\path" in v.correction
