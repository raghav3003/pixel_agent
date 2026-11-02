from goldjudge.actions import parse_code, parse_human


def _r(code):
  return " ; ".join(o.render() for o in parse_code(code))


def test_padding_sleeps_and_move_before_scroll_dropped():
  code = ("import pyautogui; import time; pyautogui.click(x=727, y=172, clicks=1, button='left'); time.sleep(0.5); "
          "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('backspace'); pyautogui.write('pic.jpg', interval=0.05); "
          "pyautogui.press('enter'); pyautogui.sleep(2); ")
  assert _r(code) == 'CLICK @(727,172) ; HOTKEY ctrl+a ; PRESS backspace ; TYPE "pic.jpg" ; PRESS enter'
  assert _r("pyautogui.moveTo(x=990, y=600); time.sleep(0.3); pyautogui.vscroll(-5); pyautogui.sleep(2)") == "SCROLL -5 @(990,600)"
  assert _r("import time; time.sleep(5.0); ") == "WAIT 5s"


def test_key_holds_fold_into_chords_and_modifiers():
  assert _r("pyautogui.keyDown('ctrl')\npyautogui.keyDown('a')\npyautogui.keyUp('a')\npyautogui.keyUp('ctrl')") == "HOTKEY ctrl+a"
  assert _r("pyautogui.keyDown('ctrl')\npyautogui.keyDown('shift')\npyautogui.keyDown('t')\n"
            "pyautogui.keyUp('t')\npyautogui.keyUp('shift')\npyautogui.keyUp('ctrl')") == "HOTKEY ctrl+shift+t"
  assert _r("pyautogui.keyDown('alt'); pyautogui.click(x=439, y=810); pyautogui.keyUp('alt')") == "CLICK @(439,810) +alt"
  assert _r("pyautogui.keyDown('esc')\npyautogui.keyUp('esc')") == "PRESS escape"
  assert _r("pyautogui.keyUp('shift')") == "KEY_UP shift"  # released a key held from an earlier step
  assert "!key" in _r("pyautogui.keyDown('left left')\npyautogui.keyUp('left left')")  # xdotool string: a no-op


def test_drags_typing_and_fallbacks():
  assert _r("pyautogui.moveTo(207, 433); pyautogui.keyDown('ctrl'); pyautogui.dragTo(601, 433, button='left'); "
            "pyautogui.mouseUp(); pyautogui.keyUp('ctrl')") == "DRAG @(207,433) -> (601,433) +ctrl"
  assert _r("pyautogui.mouseDown(100,100); pyautogui.moveTo(200,200); pyautogui.mouseUp()") == "DRAG @(100,100) -> (200,200)"
  assert _r("pyautogui.press('h')\npyautogui.press('I')\npyautogui.press(' ')\npyautogui.press('enter')") == 'TYPE "hI " ; PRESS enter'
  assert _r("pyautogui.write(message='nike.com')") == 'TYPE "nike.com"'
  assert _r("pyautogui.click(5, 5, clicks=2)") == "DOUBLE_CLICK @(5,5)"
  assert parse_code("import subprocess\nsubprocess.run(['wmctrl'])")[0].op == "CODE"  # never executed, kept verbatim
  assert [o.op for o in parse_code("DONE")] == ["DONE"]
  assert parse_code("") == []


def test_ops_render_back_to_pyautogui():
  ops = parse_code("pyautogui.keyDown('alt'); pyautogui.click(439, 810); pyautogui.keyUp('alt')")
  assert ops[0].to_pyautogui() == "pyautogui.keyDown('alt'); pyautogui.click(439, 810); pyautogui.keyUp('alt')"
  assert parse_code("pyautogui.scroll(-3, 960, 600)")[0].to_pyautogui() == "pyautogui.moveTo(960, 600); pyautogui.scroll(-3)"


def test_human_steps():
  assert parse_human("`CLICK` the text box labeled 'Search'").render() == "CLICK \"the text box labeled 'Search'\""
  assert parse_human("`TYPING` 'DOTA 2'").text == "DOTA 2"
  assert parse_human("`SCROLL` down 300 units (-300)").amount == -300
  assert parse_human("`HOTKEY` ctrl-A").keys == ["ctrl", "a"]
  assert parse_human("`HOTEKEY` 'ctrl-down arrow'").keys == ["ctrl", "down"]
  assert parse_human("`KEY DOWN` Shift").render() == "KEY_DOWN shift"
  assert parse_human("`MOVE_TO bottom right corner of the cell B12`").target == "bottom right corner of the cell B12"
  assert parse_human("8x`CLICK` the right arrow").repeat == 8
  assert parse_human("`REPEAT_UNTIL_DONE` for the rest").op == "OTHER"
