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

step "Vaciando Desktop, Documents y Downloads"
rm -rf ~/Desktop/* ~/Documents/* ~/Downloads/*

step "Eliminando IE-workspace anterior"
rm -rf ~/IE-workspace

step "Instalando sense-hat"
sudo apt-get install -y sense-hat

step "Clonando repositorio IE-workspace"
git clone https://github.com/efhes/IE-workspace.git ~/IE-workspace

step "Actualizando mediapipe y keras en venvml"
source ~/venvml/bin/activate
pip install --upgrade mediapipe keras

step "Copiando proyectos a ~/workspace"
mkdir -p ~/workspace
cd ~/IE-workspace
for p in FER_mediapipe HAR_mediapipe HAR_inercial image_recognition; do
  log "   copiando $p"
  cp -R "$p" ~/workspace/
done

log "=== COMPLETADO correctamente ==="
