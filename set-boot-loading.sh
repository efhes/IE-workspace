#!/bin/bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

TOTAL=6
STEP=0
IP=$(hostname -I | awk '{print $1}')

log()  { echo "[$(date +%H:%M:%S)] [$(hostname) | $IP] $*"; }
step() { STEP=$((STEP+1)); log "Paso $STEP/$TOTAL: $*"; }
trap 'log "ERROR en el paso $STEP (línea $LINENO). Abortando."' ERR

log "=== INICIO del mantenimiento ==="

step "Pull repo IE-workspace"
cd ~/IE-workspace
git pull

step "Copiando proyectos a ~/workspace"
mkdir -p ~/workspace
cd ~/IE-workspace
for p in common FER_mediapipe HAR_mediapipe HAR_inercial image_recognition; do
  log "   copiando $p"
  cp -R "$p" ~/workspace/
done

cd ~/IE-workspace/boot_selector
./install.sh
# Wait and show 10 seconds countdown before reboot

for i in {10..1}; do
    log "Reboot in $i seconds..."
    sleep 1
done

log "=== COMPLETADO correctamente ==="

sudo reboot