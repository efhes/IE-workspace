#!/usr/bin/env python3
"""
Interactive console + Sense HAT boot-mode selector for Raspberry Pi OS.

Reads mode definitions from ~/boot-modes.json (symlinked to boot_selector/boot-modes.json).
- CLI modes run directly in the console (tty1) with no desktop overhead.
- GUI modes run inside a minimal labwc Wayland session (exits when the app exits).
- On timeout (or choosing Normal Desktop), starts the normal graphical desktop.
- Sense HAT support (auto-detected):
  * Displays the selected mode symbol (0-9, S) and countdown bar on the 8x8 LED matrix.
  * Joystick navigates (Up/Down/Left/Right) and launches (Middle press) headlessly.
  * Holding the joystick Middle button for N seconds stops a running demo and returns
    to the Boot Menu without needing a keyboard.
"""

import argparse
import fcntl
import glob
import json
import os
import select
import shlex
import signal
import stat
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path

HOME = Path.home()
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "boot-modes.json"
LABWC_CONFIG_DIR = SCRIPT_DIR / "labwc"
LOG_PATH = HOME / "boot-mode-last.log"
RUNNER_SCRIPT_PATH = Path(f"/tmp/boot-selector-runner-{os.getuid()}.sh")

# Linux evdev constants for Sense HAT joystick
SENSE_HAT_EVDEV_NAME = "Raspberry Pi Sense HAT Joystick"
EVIOCGRAB = 0x40044590
EV_KEY = 0x01
EVENT_FORMAT = "llHHI"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
STATE_RELEASE = 0
STATE_PRESS = 1
STATE_HOLD = 2
KEY_MAP = {
    103: "up",
    108: "down",
    105: "left",
    106: "right",
    28: "middle",
}

# Direction rotation tables (physical -> logical direction matching LED rotation)
ROTATION_MAP = {
    0: {"up": "up", "down": "down", "left": "left", "right": "right", "middle": "middle"},
    90: {"up": "left", "down": "right", "left": "down", "right": "up", "middle": "middle"},
    180: {"up": "down", "down": "up", "left": "right", "right": "left", "middle": "middle"},
    270: {"up": "right", "down": "left", "left": "up", "right": "down", "middle": "middle"},
}

# Compact 4x5 glyphs for 8x8 Sense HAT LED matrix (rows 0..4, cols 2..5)
GLYPHS_4X5 = {
    "0": [
        "1111",
        "1001",
        "1001",
        "1001",
        "1111",
    ],
    "1": [
        "0110",
        "0010",
        "0010",
        "0010",
        "0111",
    ],
    "2": [
        "1111",
        "0001",
        "1111",
        "1000",
        "1111",
    ],
    "3": [
        "1111",
        "0001",
        "0111",
        "0001",
        "1111",
    ],
    "4": [
        "1001",
        "1001",
        "1111",
        "0001",
        "0001",
    ],
    "5": [
        "1111",
        "1000",
        "1111",
        "0001",
        "1111",
    ],
    "6": [
        "1111",
        "1000",
        "1111",
        "1001",
        "1111",
    ],
    "7": [
        "1111",
        "0001",
        "0010",
        "0100",
        "0100",
    ],
    "8": [
        "1111",
        "1001",
        "1111",
        "1001",
        "1111",
    ],
    "9": [
        "1111",
        "1001",
        "1111",
        "0001",
        "1111",
    ],
    "S": [
        "0111",
        "1000",
        "0110",
        "0001",
        "1110",
    ],
    "OK": [
        "0001",
        "0010",
        "1010",
        "0100",
        "0000",
    ],
    "X": [
        "1001",
        "0110",
        "0110",
        "1001",
        "0000",
    ],
}

COLOR_OFF = (0, 0, 0)
COLOR_CLI = (0, 200, 80)
COLOR_GUI = (0, 150, 255)
COLOR_DESKTOP = (220, 190, 40)
COLOR_SHELL = (220, 60, 90)
COLOR_BAR = (180, 180, 180)
COLOR_PAUSED = (40, 40, 100)
COLOR_OK = (0, 220, 60)
COLOR_STOP = (220, 40, 40)


def expand_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    return os.path.expanduser(os.path.expandvars(raw))


def find_sensehat_joystick_device() -> str | None:
    for evdev in sorted(glob.glob("/sys/class/input/event*")):
        name_path = os.path.join(evdev, "device", "name")
        try:
            with open(name_path, "r", encoding="utf-8") as f:
                if f.read().strip() == SENSE_HAT_EVDEV_NAME:
                    return os.path.join("/dev/input", os.path.basename(evdev))
        except OSError:
            continue
    return None


def build_led_frame(
    symbol: str,
    color: tuple[int, int, int],
    remaining_secs: int | None,
    total_secs: int | None,
) -> list[tuple[int, int, int]]:
    """Build an 8x8 pixel array with a 4x5 symbol and bottom-row countdown bar."""
    pixels = [COLOR_OFF] * 64
    glyph = GLYPHS_4X5.get(symbol, GLYPHS_4X5["0"])
    for r, row_str in enumerate(glyph):
        for c, ch in enumerate(row_str):
            if ch == "1":
                pixels[r * 8 + (c + 2)] = color

    # Row 7: countdown bar or paused indicator
    if remaining_secs is not None and total_secs and total_secs > 0:
        lit = max(0, min(8, int((remaining_secs / total_secs) * 8 + 0.5)))
        for c in range(lit):
            pixels[7 * 8 + c] = COLOR_BAR
    elif remaining_secs is None:
        pixels[7 * 8 + 3] = COLOR_PAUSED
        pixels[7 * 8 + 4] = COLOR_PAUSED

    return pixels


class SenseHatUI:
    """Optional Sense HAT LED matrix + exclusively grabbed joystick controller."""

    def __init__(self, settings: dict | None = None):
        cfg = settings or {}
        self.enabled = bool(cfg.get("enabled", True))
        rot = int(cfg.get("rotation", 180))
        self.rotation = rot if rot in ROTATION_MAP else 180
        self.low_light = bool(cfg.get("low_light", True))
        self.hold_to_stop_seconds = float(cfg.get("hold_to_stop_seconds", 3))

        self._sense = None
        self._stick_fd: int | None = None
        self._grabbed = False
        self._joystick_dev = find_sensehat_joystick_device() if self.enabled else None

    @property
    def has_led(self) -> bool:
        return self._sense is not None

    @property
    def joystick_fd(self) -> int | None:
        return self._stick_fd if self._grabbed else None

    def open(self) -> None:
        if not self.enabled:
            return
        if self._sense is None:
            try:
                from sense_hat import SenseHat

                sense = SenseHat()
                sense.set_rotation(self.rotation)
                sense.low_light = self.low_light
                self._sense = sense
            except Exception:
                self._sense = None

        self._grab_joystick()

    def _grab_joystick(self) -> None:
        if self._stick_fd is not None or not self._joystick_dev:
            return
        try:
            fd = os.open(self._joystick_dev, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        try:
            # Grab exclusively so tty1 does NOT also receive duplicate arrow/Enter keys
            fcntl.ioctl(fd, EVIOCGRAB, 1)
            self._stick_fd = fd
            self._grabbed = True
            self._drain_joystick()
        except OSError:
            # If grab fails, close and let tty1 handle joystick keys directly
            os.close(fd)
            self._stick_fd = None
            self._grabbed = False

    def _drain_joystick(self) -> None:
        if self._stick_fd is None:
            return
        while True:
            r, _, _ = select.select([self._stick_fd], [], [], 0)
            if not r:
                break
            try:
                if not os.read(self._stick_fd, EVENT_SIZE):
                    break
            except OSError:
                break

    def release(self, clear_led: bool = True) -> None:
        if self._stick_fd is not None:
            if self._grabbed:
                try:
                    fcntl.ioctl(self._stick_fd, EVIOCGRAB, 0)
                except OSError:
                    pass
            try:
                os.close(self._stick_fd)
            except OSError:
                pass
            self._stick_fd = None
            self._grabbed = False

        if self._sense is not None:
            if clear_led:
                try:
                    self._sense.clear()
                except Exception:
                    pass
            try:
                self._sense.stick.close()
            except Exception:
                pass
            self._sense = None

    def show_selection(
        self,
        symbol: str,
        color: tuple[int, int, int],
        remaining_secs: int | None,
        total_secs: int | None,
    ) -> None:
        if self._sense is None:
            return
        try:
            pixels = build_led_frame(symbol, color, remaining_secs, total_secs)
            self._sense.set_pixels(pixels)
        except Exception:
            pass

    def flash_icon(self, symbol: str, color: tuple[int, int, int], duration: float = 0.35) -> None:
        if self._sense is None:
            return
        try:
            pixels = build_led_frame(symbol, color, 0, 0)
            self._sense.set_pixels(pixels)
            time.sleep(duration)
            self._sense.clear()
        except Exception:
            pass

    def read_joystick_press(self) -> str | None:
        """Read one logical direction ('up', 'down', 'left', 'right', 'middle') on press."""
        if self._stick_fd is None:
            return None
        try:
            raw = os.read(self._stick_fd, EVENT_SIZE)
        except BlockingIOError:
            return None
        except OSError:
            return None
        if len(raw) != EVENT_SIZE:
            return None
        _, _, ev_type, code, value = struct.unpack(EVENT_FORMAT, raw)
        if ev_type == EV_KEY and value == STATE_PRESS and code in KEY_MAP:
            physical_dir = KEY_MAP[code]
            return ROTATION_MAP[self.rotation][physical_dir]
        return None


class JoystickStopWatcher:
    """
    Non-grabbing background watcher that stops a running child process group if the
    Sense HAT middle button is held continuously for `hold_seconds`.
    """

    def __init__(self, hold_seconds: float, on_trigger):
        self.hold_seconds = hold_seconds
        self.on_trigger = on_trigger
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.triggered = False

    def start(self) -> None:
        if self.hold_seconds <= 0:
            return
        dev = find_sensehat_joystick_device()
        if not dev:
            return
        self._thread = threading.Thread(target=self._run, args=(dev,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self, dev: str) -> None:
        try:
            fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return
        press_start: float | None = None
        try:
            while not self._stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    try:
                        raw = os.read(fd, EVENT_SIZE)
                    except OSError:
                        break
                    if len(raw) == EVENT_SIZE:
                        _, _, ev_type, code, value = struct.unpack(EVENT_FORMAT, raw)
                        if ev_type == EV_KEY and KEY_MAP.get(code) == "middle":
                            if value == STATE_PRESS:
                                press_start = time.monotonic()
                            elif value == STATE_RELEASE:
                                press_start = None
                if press_start is not None and (time.monotonic() - press_start) >= self.hold_seconds:
                    self.triggered = True
                    self.on_trigger()
                    return
        finally:
            try:
                os.close(fd)
            except OSError:
                pass


def load_config(path: Path) -> tuple[int, dict, list[dict]]:
    if not path.exists():
        print(f"[boot-selector] Config not found: {path}", file=sys.stderr)
        return 10, {}, []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    timeout = int(data.get("timeout_seconds", 10))
    sensehat_cfg = data.get("sensehat", {})
    modes = data.get("modes", [])
    return timeout, sensehat_cfg, modes


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def selection_visual(modes: list[dict], selected_idx: int) -> tuple[str, tuple[int, int, int]]:
    if selected_idx < len(modes):
        symbol = str(selected_idx + 1) if selected_idx < 9 else "9"
        mode_type = modes[selected_idx].get("type", "cli").lower()
        color = COLOR_GUI if mode_type == "gui" else COLOR_CLI
        return symbol, color
    if selected_idx == len(modes):
        return "0", COLOR_DESKTOP
    return "S", COLOR_SHELL


def render_menu(
    modes: list[dict],
    selected_idx: int,
    remaining_secs: int | None,
    has_sensehat: bool,
) -> None:
    clear_screen()
    entries = []
    for i, mode in enumerate(modes, start=1):
        mode_type = mode.get("type", "cli").upper()
        entries.append((str(i), f"{mode['name']}  [{mode_type}]"))
    entries.append(("0", "Normal Desktop (LXDE-pi-labwc)"))
    entries.append(("s", "Console Shell (stay on tty1)"))

    print("=" * 62)
    print("             Raspberry Pi Boot Mode Selector")
    print("=" * 62)
    print()

    for idx, (key, label) in enumerate(entries):
        if idx == selected_idx:
            print(f"  \033[7m > [{key}] {label} \033[0m")
        else:
            print(f"    [{key}] {label}")

    print()
    print("-" * 62)
    print("  Keyboard : [1-9] Launch | [Up/Down + Enter] | [0] Desktop | [s] Shell")
    if has_sensehat:
        print("  SenseHAT : [Joystick] Select | [Middle] Launch | [Hold 3s] Stop demo")
    print("-" * 62)
    if remaining_secs is not None:
        print(f"\n  Auto-starting Normal Desktop in {remaining_secs}s (press any key/stick to pause)...")
    else:
        print("\n  Timer paused. Press Enter / Joystick Middle or a key to launch.")
    sys.stdout.flush()


def read_input_event(timeout: float, sense_ui: SenseHatUI) -> tuple[str, str] | None:
    """Wait for either a stdin keypress or a grabbed Sense HAT joystick press."""
    fds = [sys.stdin.fileno()]
    stick_fd = sense_ui.joystick_fd
    if stick_fd is not None:
        fds.append(stick_fd)

    rlist, _, _ = select.select(fds, [], [], timeout)
    if not rlist:
        return None

    if stick_fd is not None and stick_fd in rlist:
        direction = sense_ui.read_joystick_press()
        if direction:
            return ("stick", direction)

    if sys.stdin.fileno() in rlist:
        ch = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            r2, _, _ = select.select([sys.stdin.fileno()], [], [], 0.05)
            if r2:
                seq = os.read(sys.stdin.fileno(), 2).decode("utf-8", errors="ignore")
                return ("key", "\x1b" + seq)
            return ("key", "ESC")
        return ("key", ch)

    return None


def prompt_selection(
    modes: list[dict],
    timeout_seconds: int | None,
    sense_ui: SenseHatUI,
) -> str:
    """
    Display the menu and return one of:
    - "mode:<index>" (0-based index into modes)
    - "desktop"
    - "shell"
    """
    total_items = len(modes) + 2
    desktop_idx = len(modes)
    selected_idx = desktop_idx

    total_secs = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
    deadline = time.monotonic() + total_secs if total_secs is not None else None

    sense_ui.open()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    last_render_state = None
    try:
        tty.setcbreak(fd)
        while True:
            remaining = None
            remaining_secs = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "desktop"
                remaining_secs = max(1, int(remaining + 0.999))

            render_state = (selected_idx, remaining_secs)
            if render_state != last_render_state:
                render_menu(modes, selected_idx, remaining_secs, sense_ui.has_led or sense_ui.joystick_fd is not None)
                symbol, color = selection_visual(modes, selected_idx)
                sense_ui.show_selection(symbol, color, remaining_secs, total_secs)
                last_render_state = render_state

            wait_time = 0.2 if deadline is not None else 60.0
            if remaining is not None:
                wait_time = min(wait_time, max(0.05, remaining))

            event = read_input_event(wait_time, sense_ui)
            if event is None:
                continue

            src, val = event
            if src == "stick":
                if val in ("up", "left"):
                    deadline = None
                    selected_idx = (selected_idx - 1) % total_items
                elif val in ("down", "right"):
                    deadline = None
                    selected_idx = (selected_idx + 1) % total_items
                elif val == "middle":
                    if selected_idx < len(modes):
                        return f"mode:{selected_idx}"
                    if selected_idx == desktop_idx:
                        return "desktop"
                    return "shell"
                continue

            # Keyboard event
            if val in ("\x03", "\x04"):
                return "shell"
            if val in ("\x1b[A", "\x1b[D", "k", "K"):  # Up / Left
                deadline = None
                selected_idx = (selected_idx - 1) % total_items
            elif val in ("\x1b[B", "\x1b[C", "j", "J"):  # Down / Right
                deadline = None
                selected_idx = (selected_idx + 1) % total_items
            elif val in ("\r", "\n"):
                if selected_idx < len(modes):
                    return f"mode:{selected_idx}"
                if selected_idx == desktop_idx:
                    return "desktop"
                return "shell"
            elif val == "0":
                return "desktop"
            elif val in ("s", "S"):
                return "shell"
            elif val.isdigit() and val != "0":
                num = int(val)
                if 1 <= num <= len(modes):
                    return f"mode:{num - 1}"
            else:
                deadline = None
    except KeyboardInterrupt:
        return "shell"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def build_mode_script(mode: dict, log_output: bool = False) -> str:
    workdir = shlex.quote(expand_path(mode["workdir"]) or str(HOME))
    command = mode["command"]
    venv = expand_path(mode.get("venv"))

    lines = [
        "#!/bin/bash",
        f"cd {workdir} || exit 1",
    ]
    if venv:
        activate = shlex.quote(str(Path(venv) / "bin" / "activate"))
        lines.append(f"source {activate} || exit 1")

    if log_output:
        log_q = shlex.quote(str(LOG_PATH))
        lines.append(f"exec {command} > {log_q} 2>&1")
    else:
        lines.append(f"exec {command}")

    return "\n".join(lines) + "\n"


def write_runner_script(content: str) -> None:
    RUNNER_SCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNNER_SCRIPT_PATH.write_text(content, encoding="utf-8")
    RUNNER_SCRIPT_PATH.chmod(RUNNER_SCRIPT_PATH.stat().st_mode | stat.S_IXUSR)


def run_with_joystick_stop(cmd: list[str], sense_ui: SenseHatUI) -> tuple[int, bool]:
    """Run cmd in its own process group; allow holding Sense HAT middle button to stop it."""
    proc = subprocess.Popen(cmd, start_new_session=True)

    def stop_proc() -> None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    watcher = JoystickStopWatcher(sense_ui.hold_to_stop_seconds, stop_proc)
    watcher.start()
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        stop_proc()
        rc = proc.wait()
    finally:
        watcher.stop()
    return rc, watcher.triggered


def run_cli_mode(mode: dict, sense_ui: SenseHatUI, dry_run: bool = False) -> int:
    workdir = expand_path(mode["workdir"])
    venv = expand_path(mode.get("venv"))
    clear_screen()
    print(f"=== Starting CLI mode: {mode['name']} ===")
    print(f"    Workdir : {workdir}")
    if venv:
        print(f"    Venv    : {venv}")
    print(f"    Command : {mode['command']}")
    print("    Stop    : Ctrl+C (or hold Sense HAT Middle button 3s)\n")

    script = build_mode_script(mode, log_output=False)
    if dry_run:
        print("[dry-run] Would execute:\n" + script)
        return 0

    write_runner_script(script)
    rc, stopped_by_stick = run_with_joystick_stop(["/bin/bash", str(RUNNER_SCRIPT_PATH)], sense_ui)
    if stopped_by_stick:
        print("\n[boot-selector] Stopped via Sense HAT joystick.")
        sense_ui.open()
        sense_ui.flash_icon("X", COLOR_STOP, duration=0.4)
        sense_ui.release(clear_led=True)
    return rc


def run_gui_mode(mode: dict, sense_ui: SenseHatUI, dry_run: bool = False) -> int:
    workdir = expand_path(mode["workdir"])
    venv = expand_path(mode.get("venv"))
    clear_screen()
    print(f"=== Starting GUI mode: {mode['name']} ===")
    print(f"    Workdir : {workdir}")
    if venv:
        print(f"    Venv    : {venv}")
    print(f"    Command : {mode['command']}")
    print(f"    Log     : {LOG_PATH}")
    print("    Stop    : Close window / press 'q' (or hold Sense HAT Middle button 3s)\n")

    script = build_mode_script(mode, log_output=True)

    wrapper = f"""#!/bin/sh
if [ -f /usr/bin/setup_env ]; then
    . /usr/bin/setup_env
fi
if [ ! -f "$HOME/.config/kanshi/config" ]; then
    mkdir -p "$HOME/.config/kanshi"
    touch "$HOME/.config/kanshi/config"
fi
if command -v raspi-config >/dev/null 2>&1 && raspi-config nonint is_pi && ! raspi-config nonint gpu_has_mmu; then
    export WLR_RENDERER=pixman
fi
exec /usr/bin/labwc -C {shlex.quote(str(LABWC_CONFIG_DIR))} -S {shlex.quote(str(RUNNER_SCRIPT_PATH))}
"""

    if dry_run:
        print("[dry-run] Runner script:\n" + script)
        print("[dry-run] Labwc launcher:\n" + wrapper)
        return 0

    write_runner_script(script)
    rc, stopped_by_stick = run_with_joystick_stop(["/bin/sh", "-c", wrapper], sense_ui)
    if stopped_by_stick:
        print("\n[boot-selector] Stopped via Sense HAT joystick.")
        sense_ui.open()
        sense_ui.flash_icon("X", COLOR_STOP, duration=0.4)
        sense_ui.release(clear_led=True)
    elif rc != 0 and LOG_PATH.exists():
        print(f"\n[boot-selector] Program exited with code {rc}. Last log lines:")
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-20:]:
            print("  " + line)
    return rc


def start_normal_desktop(dry_run: bool = False) -> None:
    clear_screen()
    print("Starting Normal Desktop...")
    if dry_run:
        print("[dry-run] Would run: sudo -n systemctl start lightdm (fallback: /usr/bin/labwc-pi)")
        return

    res = subprocess.run(["sudo", "-n", "systemctl", "start", "lightdm"])
    if res.returncode != 0:
        os.execv("/usr/bin/labwc-pi", ["/usr/bin/labwc-pi"])


def post_exit_prompt(sense_ui: SenseHatUI, wait_seconds: int = 3) -> bool:
    """
    Headless-friendly post-exit prompt:
    Returns True to reopen the Boot Menu (after `wait_seconds` or Enter/Joystick),
    or False if user presses 's' / Ctrl+C to drop to Console Shell.
    """
    print("\n" + "-" * 62)
    print(f"Program exited. Returning to Boot Menu in {wait_seconds}s...")
    print("Press [Enter / Joystick] now, or [s / Ctrl+C] for Console Shell.")
    sys.stdout.flush()

    sense_ui.open()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    deadline = time.monotonic() + wait_seconds
    try:
        tty.setcbreak(fd)
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return True
            event = read_input_event(min(0.2, rem), sense_ui)
            if event is None:
                continue
            src, val = event
            if src == "stick":
                return True
            if val in ("s", "S", "\x03", "\x04"):
                return False
            if val in ("\r", "\n"):
                return True
    except KeyboardInterrupt:
        return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sense_ui.release(clear_led=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi console + Sense HAT boot-mode selector")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to boot-modes.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without launching")
    parser.add_argument("--no-timeout", action="store_true", help="Disable auto-start timeout")
    args = parser.parse_args()

    if not sys.stdin.isatty():
        print("[boot-selector] Not running in an interactive TTY; exiting.", file=sys.stderr)
        return 0

    timeout_seconds, sensehat_cfg, modes = load_config(args.config)
    sense_ui = SenseHatUI(sensehat_cfg)
    first_pass = True

    try:
        while True:
            current_timeout = None if (args.no_timeout or not first_pass) else timeout_seconds
            first_pass = False

            choice = prompt_selection(modes, current_timeout, sense_ui)

            if choice == "shell":
                sense_ui.release(clear_led=True)
                clear_screen()
                print("Dropping to Console Shell. Run '~/boot-selector.py' anytime to reopen the menu.")
                return 0

            if choice == "desktop":
                sense_ui.flash_icon("OK", COLOR_OK, duration=0.3)
                sense_ui.release(clear_led=True)
                start_normal_desktop(dry_run=args.dry_run)
                return 0

            if choice.startswith("mode:"):
                idx = int(choice.split(":", 1)[1])
                mode = modes[idx]
                sense_ui.flash_icon("OK", COLOR_OK, duration=0.3)
                # Release Sense HAT completely before starting the demo
                sense_ui.release(clear_led=True)

                mode_type = mode.get("type", "cli").lower()
                if mode_type == "gui":
                    run_gui_mode(mode, sense_ui, dry_run=args.dry_run)
                else:
                    run_cli_mode(mode, sense_ui, dry_run=args.dry_run)

                if not post_exit_prompt(sense_ui, wait_seconds=3):
                    sense_ui.release(clear_led=True)
                    clear_screen()
                    return 0
    finally:
        sense_ui.release(clear_led=True)


if __name__ == "__main__":
    sys.exit(main())
