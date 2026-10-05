ssh-keygen -t ed25519            # si no tienes clave ya
sudo apt install sshpass         # solo para esta distribución inicial

# claves SSH en lugar de contraseña
for i in $(seq 101 125); do
  sshpass -p pi ssh-copy-id -o StrictHostKeyChecking=accept-new pi@10.8.43.$i
done

for i in $(seq 101 125); do
  ip=10.8.43.$i
  ssh -o ConnectTimeout=5 pi@$ip \
    "echo pi | sudo -S -p '' sh -c 'echo \"pi ALL=(root) NOPASSWD: /usr/bin/apt-get\" > /etc/sudoers.d/020_pi-apt && chmod 440 /etc/sudoers.d/020_pi-apt && visudo -cf /etc/sudoers.d/020_pi-apt'" </dev/null \
    && echo "OK $ip" || echo "FALLO $ip"
done
