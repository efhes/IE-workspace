#!/usr/bin/env bash
# Cleanly removes the Raspberry Pi Boot Mode Selector configuration.
#   ./uninstall.sh                  # Removes symlinks, ~/.profile hook, restores Desktop Autologin (B4)
#   ./uninstall.sh --no-boot-target # Skip sudo raspi-config step

set -euo pipefail

NO_BOOT_TARGET=0
for arg in "$@"; do
    case "$arg" in
        --no-boot-target) NO_BOOT_TARGET=1 ;;
        -h|--help)
            echo "Usage: $0 [--no-boot-target]"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 1
            ;;
    esac
done

echo "=== Uninstalling Raspberry Pi Boot Mode Selector ==="

# 1. Remove managed block from ~/.profile
PROFILE_FILE="$HOME/.profile"
if [ -f "$PROFILE_FILE" ]; then
    python3 - "$PROFILE_FILE" <<'PY'
import re, sys
from pathlib import Path

profile = Path(sys.argv[1])
text = profile.read_text(encoding="utf-8")
text = re.sub(
    r"\n?# >>> raspi-boot-selector >>>.*?# <<< raspi-boot-selector <<<\n?",
    "\n",
    text,
    flags=re.DOTALL,
)
profile.write_text(text.rstrip() + "\n", encoding="utf-8")
PY
    bash -n "$PROFILE_FILE"
    echo "Cleaned $PROFILE_FILE"
fi

# 2. Remove symlinks
for target in "$HOME/boot-selector.py" "$HOME/boot-modes.json"; do
    if [ -L "$target" ]; then
        rm -f "$target"
        echo "Removed symlink $target"
    fi
done

# 3. Restore Desktop Autologin (B4)
if [ "$NO_BOOT_TARGET" -eq 0 ] && command -v raspi-config >/dev/null 2>&1; then
    echo "Restoring Raspberry Pi boot behaviour to Desktop Autologin (B4)..."
    sudo raspi-config nonint do_boot_behaviour B4
else
    echo "Skipped raspi-config (--no-boot-target). To restore Desktop Autologin manually:"
    echo "  sudo raspi-config nonint do_boot_behaviour B4"
fi

echo "=== Uninstall complete ==="
