#!/bin/bash
mkdir -p logs
run_one() {
  ip=$1
  SSH_OPTS="-o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
  set -o pipefail
  if scp $SSH_OPTS clean-workspace.sh pi@$ip:/tmp/ \
     && ssh $SSH_OPTS pi@$ip "bash /tmp/clean-workspace.sh" </dev/null 2>&1 \
        | tee logs/$ip.log | sed -u "s/^/[$ip] /"; then
    echo "[$ip] >>> OK"
  else
    echo "[$ip] >>> FALLO (ver logs/$ip.log)"
  fi
}
export -f run_one
#seq -f "10.8.43.%g" 101 101 | xargs -P 10 -I{} bash -c 'run_one {}'
seq -f "10.8.43.%g" 101 125 | xargs -P 10 -I{} bash -c 'run_one {}'
