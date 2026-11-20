# Raspberry Pi Console + Sense HAT Boot Mode Selector

A portable, config-driven boot menu for Raspberry Pi OS (Bookworm / `labwc`) that lets you choose between GUI demos, headless CLI demos, the normal desktop, or a console shell at startup—using either a **keyboard** or the **Sense HAT joystick + 8×8 LED matrix** (completely headless).

## Quick Setup on a New Raspberry Pi

Copy this `boot_selector` folder (or clone the `workspace` repository) onto the new Pi and run:

```bash
cd ~/workspace/boot_selector
./install.sh
sudo reboot
```

- Pass `--install-deps` if `python3-sense-hat` is not yet installed (`./install.sh --install-deps`).
- Pass `--no-boot-target` to only install the symlinks and `~/.profile` hook without running `sudo`.
- Re-running `./install.sh` is safe and idempotent.

## Headless Sense HAT Controls

When a Sense HAT is attached, the boot selector automatically drives the 8×8 LED matrix and exclusively grabs the joystick during menu selection (releasing it cleanly before starting any demo):

| Display / Action | Meaning |
|---|---|
| **Digit `1`–`5` (Cyan)** | Selected **GUI** mode |
| **Digit `1`–`5` (Green)** | Selected **CLI** mode |
| **`0` (Yellow)** | **Normal Desktop** (`LXDE-pi-labwc`, default on timeout) |
| **`S` (Red/Magenta)** | **Console Shell** (`tty1`) |
| **Bottom LED row** | Countdown bar (shrinks to 0, or shows 2 dim center dots when paused) |
| **Joystick Up / Down / Left / Right** | Pause countdown & cycle modes (automatically rotated to match `sensehat.rotation`) |
| **Joystick Middle press** | Launch selected mode (flashes green check, clears LEDs, hands Sense HAT to demo) |
| **Hold Joystick Middle (3s) during a demo** | Sends `Ctrl+C` (`SIGINT`) to stop the running demo and returns to the Boot Menu |

## Keyboard Controls

- **`1`–`9`**: Launch mode immediately
- **`↑` / `↓` (or `←` / `→`)**: Pause countdown & highlight option
- **`Enter`**: Launch highlighted option
- **`0`**: Start Normal Desktop immediately
- **`s` or `Ctrl+C`**: Drop to Console Shell (`tty1`)

## Adding or Editing Modes

Edit `boot-modes.json` (accessible via either `~/workspace/boot_selector/boot-modes.json` or `~/boot-modes.json`):

```json
{
  "name": "My New Demo (CLI)",
  "type": "cli",
  "workdir": "~/workspace/my_demo/src",
  "venv": "~/venvml",
  "command": "python run.py"
}
```

- Paths support `~` expansion so the same file works across Raspberry Pis with different usernames.
- Sense HAT options (`enabled`, `rotation`, `low_light`, `hold_to_stop_seconds`) live under the `"sensehat"` key in `boot-modes.json`.

## Testing & Uninstalling

```bash
# Preview the menu safely without launching anything
~/boot-selector.py --dry-run

# Uninstall and restore normal Desktop Autologin (B4)
~/workspace/boot_selector/uninstall.sh
```
