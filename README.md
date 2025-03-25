# Phonon Angular Momentum (PHAM) Calculation and Visualization Tools

This repository contains two Python scripts for calculating and visualizing phonon angular momentum (PAM): `pham_band.py` and `pham_band2.py`. These tools allow researchers to analyze the angular momentum properties of phonons in crystalline materials.

## Background

Phonon angular momentum is calculated according to the method described in:
> "Angular momentum of phonons and the Einstein-de Haas effect", PRL, 112, 085503 (2014)

Both scripts read phonon data from YAML files (typically produced by phonopy) and calculate the phonon angular momentum components.

## Script Comparison

### pham_band.py

The original script with core functionality for PAM calculations and visualization.

### pham_band2.py

An enhanced version with improved code structure, additional features, and better visualization options:
- Type hints for better code readability
- More modular functions with better organization
- Enhanced plotting capabilities including single-band analysis
- Data export functionality
- Additional visualization modes and normalization options
- Improved colormaps (PiYG, PuOr, seismic instead of RdBu)
- Better error handling

## Dependencies

Both scripts require:
- numpy
- matplotlib
- pyyaml

## Usage

### pham_band.py

```bash
python pham_band.py [-i YAML] [-t TEMPERATURE] [-d DIRECTION] [-s WIDTH HEIGHT] [-o FIGNAME] [--plt-type TYPE] [--layout LAYOUT]
```

### pham_band2.py

```bash
python pham_band2.py [-i YAML] [-t TEMPERATURE] [-a ACTION] [-d DIRECTION] [-s WIDTH HEIGHT] [-o FIGNAME] [--plt-type TYPE] [--layout LAYOUT] [-od OUTPUT_FILE] [-n NORMALIZATION] [-idx INDEX]
```

## Command-Line Arguments

### Common Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `-i` | Input YAML file containing phonon eigenvalues and eigenvectors | `band.yaml` |
| `-t` | Temperature in Kelvin | `0` |
| `-d` | Direction component(s) to plot (`x`, `y`, `z`, or `a` for all) | `a` |
| `-s`, `--figsize` | Figure size as width and height | Automatic based on layout |
| `-o` | Output figure filename | `pam.png` |
| `--plt-type` | Plot rendering type (`scatter` or `colormap`) | `scatter` (original), `scatter` (enhanced) |
| `--layout` | Layout of subfigures (`h` for horizontal or `v` for vertical) | `v` (original), `h` (enhanced) |

### Additional Arguments in pham_band2.py

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `-a`, `--action` | Action to perform: `data` (export data), `bands` (plot all bands), or `band` (plot single band) | `bands` |
| `-od`, `--output_data` | Output file for PAM data | Empty (no data output) |
| `-n`, `--normalization` | Normalization method: `per_direction` or `all` | `per_direction` |
| `-idx`, `--index` | Band number to plot when using `band` action | `0` |

## Modes and Features

### pham_band.py

1. **Visualization Types**:
   - **Scatter plot**: Represents PAM values with colored circles of varying sizes
   - **Colormap**: Represents PAM values with colored lines following the band structure

2. **Direction Modes**:
   - Plot individual components (x, y, z) or all three components

3. **Layout Options**:
   - Vertical layout: Components arranged in vertical columns
   - Horizontal layout: Components arranged in horizontal rows

### pham_band2.py

All the features from pham_band.py plus:

1. **Additional Actions**:
   - **data**: Export PAM data to a text file
   - **bands**: Plot all phonon bands with PAM coloring (default)
   - **band**: Plot a single phonon band with PAM coloring

2. **Single Band Analysis**:
   - Plot a specific band with its PAM values
   - Generate additional plot of PAM values vs. path distance

3. **Data Export**:
   - Export organized PAM data with columns for distance, frequency, and Jx, Jy, Jz components
   - Data is organized by band for easy analysis

4. **Normalization Options**:
   - **per_direction**: Normalize color scale for each direction component separately
   - **all**: Use a single normalization across all direction components

## Example Use Cases

### Basic Visualization (Both Scripts)

```bash
python pham_band.py -i my_phonon_data.yaml
```

```bash
python pham_band2.py -i my_phonon_data.yaml
```

### Plot Only z-Component (Both Scripts)

```bash
python pham_band.py -i my_phonon_data.yaml -d z
```

### Change Visualization Type (Both Scripts)

```bash
python pham_band.py -i my_phonon_data.yaml --plt-type colormap
```

### Export Data (pham_band2.py Only)

```bash
python pham_band2.py -i my_phonon_data.yaml -a data -od pam_data.txt
```

### Analyze Single Band (pham_band2.py Only)

```bash
python pham_band2.py -i my_phonon_data.yaml -a band -idx 3
```

### Use Global Normalization (pham_band2.py Only)

```bash
python pham_band2.py -i my_phonon_data.yaml -n all
```

## Output Files

Both scripts generate visualization files in the format specified by the `-o` argument (default: `pam.png`).

The enhanced script (pham_band2.py) with the `-a band` option also generates an additional file `PAMvsPATH.png` showing the phonon angular momentum components versus the path distance for the selected band.

When using the data export feature in pham_band2.py, the output file contains columns:
```
Distance  Frequency  Jx  Jy  Jz
```

## Technical Details

### Phonon Angular Momentum Calculation

The scripts calculate phonon angular momentum using the formula:

```
J0[ii] = 2.0 * np.sum(e[:,:,:,0].conj() * e[:,:,:,1], axis=2).imag
```

where `e` represents the phonon eigenvectors in the specified direction components.

The temperature dependence is included through the Bose-Einstein distribution factor.

### YAML File Format

The scripts expect YAML files with the following structure:
- `lattice`: Crystal lattice vectors
- `reciprocal_lattice`: Reciprocal lattice vectors
- `phonon`: Array of phonon data points with:
  - `q-position`: Q-point coordinates
  - `distance`: Distance along the path
  - `band`: Array of band data with:
    - `frequency`: Phonon frequency
    - `eigenvector`: Phonon eigenvector (required)
- `segment_nqpoint`: Number of Q-points in each segment
- `labels`: Optional high-symmetry point labels

## Implementation Differences

The enhanced script (pham_band2.py) implements several code improvements:
1. Type annotations for better code readability and IDE support
2. More modular function design with separation of concerns
3. Better error handling with specific error messages
4. Support for different file compression formats (gzip, lzma)
5. More flexible data processing and visualization options
