# Raspberry Pi 5 Autostart With Display

Use this when the Pi already has the project and virtual environment set up,
and you want `cap_line_ui_v2.py` to open after the Pi boots into the desktop.

## Install Autostart

From the Raspberry Pi desktop user's shell:

```bash
cd ~/detectx
./script/install_rpi5_autostart.sh
```

The default command is:

```bash
cap_line_ui_v2.py
```

The UI window should appear after login. Use the UI's calibration fields and
Start button to run the detector.

If your venv is not `~/detectx/.venv`, pass it with `VENV_DIR`:

```bash
VENV_DIR=/home/pi/path/to/venv ./script/install_rpi5_autostart.sh
```

The script creates:

- `~/.config/autostart/detectx-cap-line.desktop`
- `~/.local/bin/detectx-cap-line-start`
- `~/.config/detectx-cap-line.args`
- `~/.local/state/detectx-cap-line.log`

## Required Pi Setting

Because you want the display, the Pi must boot into the desktop session.
Enable desktop auto-login with:

```bash
sudo raspi-config
```

Then choose `System Options` -> `Boot / Auto Login` -> `Desktop Autologin`.

## Test It

From the Pi desktop:

```bash
~/.local/bin/detectx-cap-line-start
```

Reboot to confirm it starts automatically:

```bash
sudo reboot
```

## Logs

```bash
tail -f ~/.local/state/detectx-cap-line.log
```

## Change The Launched Command

Edit:

```bash
nano ~/.config/detectx-cap-line.args
```

`cap_line_ui_v2.py` currently does not need command-line arguments, so this
file is normally empty. The UI itself controls camera, GPIO, model, and timing
settings.

## Disable Autostart

```bash
rm ~/.config/autostart/detectx-cap-line.desktop
```
