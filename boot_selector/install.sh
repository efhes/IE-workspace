#!/usr/bin/env bash
# Portable, idempotent installer for Raspberry Pi Console + Sense HAT Boot Selector.
# Run on any Raspberry Pi:
#   ./install.sh                  # Full setup (symlinks, ~/.profile, groups, Console Autologin B2)
#   ./install.sh --no-boot-target # Skip sudo steps (raspi-config / usermod)
#   ./install.sh --install-deps   # Also install python3-sense-hat via apt if missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_BOOT_TARGET=0
INSTALL_DEPS=0

for arg in "$@"; do
    case "$arg" in
        --no-boot-target) NO_BOOT_TARGET=1 ;;
        --install-deps)   INSTALL_DEPS=1 ;;
        -h|--help)
            echo "Usage: $0 [--no-boot-target] [--install-deps]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

echo "=== Installing Raspberry Pi Boot Mode Selector ==="
echo "Source directory : $SCRIPT_DIR"
echo "Target user/home : $USER ($HOME)"

chmod +x "$SCRIPT_DIR/boot-selector.py" "$SCRIPT_DIR/install.sh" "$SCRIPT_DIR/uninstall.sh"

# 1. Symlink script, config, and minimal labwc config dir
link_item() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [ -L "$dst" ]; then
        rm -f "$dst"
    elif [ -e "$dst" ]; then
        local backup="${dst}.bak.$(date +%Y%m%d%H%M%S)"
        echo "Backing up existing $dst -> $backup"
        mv "$dst" "$backup"
    fi
    ln -s "$src" "$dst"
    echo "Linked $dst -> $src"
}

link_item "$SCRIPT_DIR/boot-selector.py" "$HOME/boot-selector.py"
link_item "$SCRIPT_DIR/boot-modes.json"  "$HOME/boot-modes.json"

# 2. Idempotently add/update the tty1 hook in ~/.profile
PROFILE_FILE="$HOME/.profile"
touch "$PROFILE_FILE"

python3 - "$PROFILE_FILE" <<'PY'
import re, sys
from pathlib import Path

profile = Path(sys.argv[1])
text = profile.read_text(encoding="utf-8")

# Remove any previous managed block
text = re.sub(
    r"\n?# >>> raspi-boot-selector >>>.*?# <<< raspi-boot-selector <<<\n?",
    "\n",
    text,
    flags=re.DOTALL,
)
# Also clean up legacy unmarked block if present
text = re.sub(
    r"\n?# Launch Boot Mode Selector on tty1 at startup.*?fi\n?",
    "\n",
    text,
    flags=re.DOTALL,
)

block = """# >>> raspi-boot-selector >>>
# Launch Boot Mode Selector on tty1 at startup (ignored over SSH or inside GUI sessions)
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && [ "$(tty 2>/dev/null)" = "/dev/tty1" ] && [ -z "$BOOT_SELECTOR_RAN" ]; then
    export BOOT_SELECTOR_RAN=1
    /usr/bin/python3 "$HOME/boot-selector.py"
fi
# <<< raspi-boot-selector <<<
"""

text = text.rstrip() + "\n\n" + block
profile.write_text(text, encoding="utf-8")
PY
bash -n "$PROFILE_FILE"
echo "Updated $PROFILE_FILE"

# 3. Check optional Sense HAT dependency
if ! /usr/bin/python3 -c "import sense_hat" 2>/dev/null; then
    if [ "$INSTALL_DEPS" -eq 1 ]; then
        echo "Installing python3-sense-hat..."
        sudo apt-get update && sudo apt-get install -y python3-sense-hat
    else
        echo "Note: python3-sense-hat is not installed for system python3."
        echo "      Sense HAT features will auto-disable until installed (re-run with --install-deps or: sudo apt install python3-sense-hat)."
    fi
else
    echo "Verified: system python3 has sense_hat installed."
fi

# 4. System configuration (groups + Console Autologin B2)
if [ "$NO_BOOT_TARGET" -eq 0 ]; then
    echo "Ensuring $USER is in input and video groups (for Sense HAT joystick & camera/DRM)..."
    sudo usermod -aG input,video "$USER"

    if command -v raspi-config >/dev/null 2>&1; then
        echo "Setting Raspberry Pi boot behaviour to Console Autologin (B2)..."
        sudo raspi-config nonint do_boot_behaviour B2
    else
        echo "Warning: raspi-config not found; set default target to multi-user.target manually." >&2
    fi
else
    echo "Skipped sudo system steps (--no-boot-target). To enable on boot, run:"
    echo "  sudo usermod -aG input,video $USER"
    echo "  sudo raspi-config nonint do_boot_behaviour B2"
fi

# 5. Validate boot-modes.json
python3 - "$HOME/boot-modes.json" <<'PY'
import json, os, sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"Validated boot-modes.json ({len(cfg.get('modes', []))} modes, timeout={cfg.get('timeout_seconds')}s):")
for i, m in enumerate(cfg.get("modes", []), start=1):
    wd = Path(os.path.expanduser(os.path.expandvars(m["workdir"])))
    venv = m.get("venv")
    venv_ok = (not venv) or (Path(os.path.expanduser(os.path.expandvars(venv))) / "bin" / "activate").is_file()
    status = "OK" if (wd.is_dir() and venv_ok) else "WARN (path/venv missing on this Pi)"
    print(f"  [{i}] {m['name']} ({m.get('type','cli').upper()}) -> {status}")
PY

echo ""
echo "=== Installation complete! ==="
echo "Test safely without rebooting: ~/boot-selector.py --dry-run"
echo "Reboot to activate:            sudo reboot"
