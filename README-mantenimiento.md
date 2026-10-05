# Mantenimiento de las Raspberry Pi del laboratorio

Scripts para preparar y restablecer, de forma automática y en paralelo, el conjunto de 25 Raspberry Pi utilizadas en las prácticas de laboratorio.

Los scripts se ejecutan desde **un PC de control** conectado a la misma red que las Pis y actúan sobre ellas por SSH. Basta con que las Pis estén encendidas y conectadas a la wifi del laboratorio.

## Contenido

| Script | Dónde se ejecuta | Cuándo | Qué hace |
|---|---|---|---|
| `permite-apt-get-sin-pwd.sh` | PC de control | **Una sola vez** | Distribuye la clave SSH a las 25 Pis y permite `sudo apt-get` sin contraseña al usuario `pi`. |
| `run-all.sh` | PC de control | Cada vez que se quiera restablecer el laboratorio | Copia `clean-workspace.sh` a cada Pi y lo ejecuta en paralelo, mostrando el progreso y guardando un log por Pi. |
| `clean-workspace.sh` | Cada Pi (lo lanza `run-all.sh`) | Lo lanza `run-all.sh` | Limpia el espacio de trabajo, reinstala dependencias y restaura los proyectos de prácticas. |

## Requisitos

**En el PC de control**

- Linux (o similar) con `bash`, `ssh`, `scp` y `xargs`.
- `sshpass`, solo para la configuración inicial (`sudo apt install sshpass`).
- Conectividad con la red `10.8.43.0/24`.

**En las Raspberry Pi**

- Servidor SSH habilitado.
- Usuario `pi` dado de alta (la contraseña inicial se usa solo en la configuración inicial).
- Direcciones IP fijas o reservadas: `10.8.43.101` (raspi 1) hasta `10.8.43.125` (raspi 25).
- Acceso a internet (GitHub y PyPI), necesario para clonar el repositorio y actualizar paquetes.
- El entorno virtual `~/venvml` con `mediapipe` y `keras` instalado.

## Uso

### 1. Configuración inicial (una sola vez)

Desde el PC de control, con todas las Pis encendidas:

```bash
bash permite-apt-get-sin-pwd.sh
```

Este script:

1. Genera una clave SSH `ed25519` en el PC de control (si no existe) y la copia a las 25 Pis, de modo que `run-all.sh` no necesite contraseña.
2. Crea en cada Pi el fichero `/etc/sudoers.d/020_pi-apt`, que permite a `pi` ejecutar `sudo apt-get` sin contraseña. Lo valida con `visudo -cf` para no dejar `sudo` inutilizable.

Al final de cada Pi se imprime `OK <ip>` o `FALLO <ip>`. Si alguna falla (apagada, sin red…), vuelve a ejecutar el script cuando esté disponible: es seguro repetirlo.

### 2. Restablecer el laboratorio

```bash
bash run-all.sh
```

- Se procesan hasta 10 Pis a la vez (parámetro `-P 10` de `xargs`; modifícalo según el ancho de banda de la red).
- Cada línea de salida se etiqueta con la IP de la Pi: `[10.8.43.101] ...`.
- El log completo de cada Pi queda en `logs/<ip>.log`.
- Al terminar cada Pi se muestra `>>> OK` o `>>> FALLO`.

Una Pi apagada o inaccesible aparece como `FALLO` tras 5 segundos de espera. Una vez encendida, basta con volver a lanzar `run-all.sh`.

## Qué hace `clean-workspace.sh` en cada Pi

Se ejecuta en 6 pasos numerados (`Paso n/6`), cada uno con hora, nombre e IP de la Pi:

1. Vacía `~/Desktop`, `~/Documents` y `~/Downloads`.
2. Elimina `~/IE-workspace`.
3. Instala `sense-hat` (`sudo apt-get install -y`).
4. Clona `https://github.com/efhes/IE-workspace.git` en `~/IE-workspace`.
5. Activa `~/venvml` y actualiza `mediapipe` y `keras`.
6. Copia `FER_mediapipe`, `HAR_mediapipe`, `HAR_inercial` e `image_recognition` a `~/workspace/`.

El script usa `set -e`: si un paso falla, se detiene e indica en qué paso y línea lo hizo.

## Advertencias

- **Es destructivo.** Elimina todo el contenido de `Desktop`, `Documents` y `Downloads` y la carpeta `IE-workspace` de cada Pi. Avisa al alumnado de guardar su trabajo fuera de las Pis.
- **Se sobrescribe parte de `~/workspace`.** Las cuatro carpetas copiadas se fusionan con las existentes (los ficheros con el mismo nombre se sobrescriben, pero los ficheros extra que haya en ellas no se borran).
- **Seguridad.** Las Pis usan el usuario `pi` con una contraseña conocida. Tras la configuración inicial se recomienda desactivar la autenticación SSH por contraseña (`PasswordAuthentication no` en `/etc/ssh/sshd_config`) o cambiar la contraseña.
- La contraseña se pasa con `sshpass -p`, por lo que queda visible momentáneamente en la lista de procesos del PC de control. Úsalo solo en una red de confianza.

## Resolución de problemas

| Síntoma | Causa probable |
|---|---|
| `sudo: a terminal is required to read the password` | No se ha hecho la configuración inicial en esa Pi. Ejecuta `permite-apt-get-sin-pwd.sh`. |
| `FALLO` en una Pi sin contenido en su log | Pi apagada, fuera de la red o con SSH deshabilitado. |
| Fallo en el paso 4 o 5 | La Pi no tiene salida a internet o GitHub/PyPI no son accesibles desde la red del laboratorio. |
| `Host key verification failed` | Se reinstaló el sistema de una Pi y cambió su clave. Elimina la entrada antigua con `ssh-keygen -R <ip>`. |

Para probar sobre una sola Pi:

```bash
scp clean-workspace.sh pi@10.8.43.101:/tmp/
ssh pi@10.8.43.101 "bash /tmp/clean-workspace.sh" </dev/null
```
