# Mapshaper installation notes

Mapshaper provides command-line tools for editing Shapefile, GeoJSON, TopoJSON, CSV and related vector formats. In this project it is used for topology-aware simplification attacks and optional cleaning.

## Windows

```powershell
node -v
npm -v
npm install -g mapshaper
mapshaper -v
python -m rb_afl_system.scripts.check_mapshaper --mapshaper_bin mapshaper
```

If `mapshaper` is not found after installation:

```powershell
npm config get prefix
```

Add the printed directory, or its `bin` directory, to your PATH. Typical Windows global npm bin:

```text
C:\Users\<UserName>\AppData\Roaming\npm
```

## Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y nodejs npm
sudo npm install -g mapshaper
mapshaper -v
python -m rb_afl_system.scripts.check_mapshaper --mapshaper_bin mapshaper
```

User-local installation without sudo:

```bash
mkdir -p ~/.npm-global
npm config set prefix ~/.npm-global
echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
npm install -g mapshaper
mapshaper -v
```

## Project config

Set `mapshaper_bin` in `configs/dataset_default.json`:

```json
"mapshaper_bin": "mapshaper"
```

or use a full path, for example:

```json
"mapshaper_bin": "C:/Users/77621/AppData/Roaming/npm/mapshaper.cmd"
```
