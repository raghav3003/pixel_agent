from goldjudge.parsing import normalize_result, parse_action_type, parse_coords, step_index_from_png


def test_action_types():
  assert parse_action_type("pyautogui.click(163, 46)\n") == "click"
  assert parse_action_type("pyautogui.click(163, 46, button='right')") == "right_click"
  assert parse_action_type("pyautogui.doubleClick(10, 10)") == "double_click"
  assert parse_action_type("pyautogui.click(x=0.5, y=0.5, clicks=2)") == "double_click"
  assert parse_action_type("pyautogui.moveTo(1, 2)\npyautogui.dragTo(3, 4, button='left')") == "drag"
  assert parse_action_type("pyautogui.scroll(-3, 960, 600)") == "scroll"
  assert parse_action_type("pyautogui.hotkey('ctrl', 'f')") == "hotkey"
  assert parse_action_type("pyautogui.press('enter')") == "key"
  assert parse_action_type("pyautogui.write('hello')") == "type"
  assert parse_action_type("DONE") == "done"
  assert parse_action_type("FAIL") == "fail"
  assert parse_action_type("pyautogui.sleep(0.1)") == "wait"
  assert parse_action_type("import pyautogui; import time; pyautogui.moveTo(1855, 404); time.sleep(0.5); pyautogui.vscroll(-20)") == "scroll"


def test_coords():
  assert parse_coords("pyautogui.click(163, 46)") == [163.0, 46.0]
  assert parse_coords("pyautogui.click(x=0.948, y=0.986)") == [0.948, 0.986]
  # drag target wins over the preceding moveTo
  assert parse_coords("pyautogui.moveTo(1, 2)\npyautogui.dragTo(30, 40, button='left')") == [30.0, 40.0]
  assert parse_coords("pyautogui.dragTo(30, 40)") == [30.0, 40.0]
  assert parse_coords("pyautogui.press('enter')") is None


def test_result():
  assert normalize_result("1.0") == 1.0
  assert normalize_result("0") == 0.0
  assert normalize_result("0.958") == 0.958
  assert normalize_result("") is None
  assert normalize_result("success") == 1.0


def test_png_index():
  assert step_index_from_png("step_12_20250730@074651.png") == 12
  assert step_index_from_png("recording.mp4") is None
