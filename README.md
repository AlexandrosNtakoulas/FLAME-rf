
### FLAME-rf
FLAME-rf is a post-processing and data analysis tool for direct numerical simulation datasets
generated with the Nek5000 solver. The developed package allows researchers to integrate machine
learning and model reduction methodologies into their workflow, generating potential for new sci-
entific discoveries

This repository contains the codebase developed as part of my Bachelor Thesis at **ETH Zürich (D-MAVT)**, conducted in the **Combustion, Acoustics and Flow Physics (CAPS)** laboratory.
---

## 📘 Overview:

The goal of this work is to establish a reproducible computational pipeline that:
1. Extracts local flame characteristics (e.g., curvature, strain, species concentrations, derivatives) from high-fidelity DNS data
2. Performs **feature scaling, dimensionality reduction, and regression** to uncover physical relations
3. Bridges **physics-based modeling** with **data-driven discovery** through interpretable and efficient machine-learning models

---

## 🧩 Project Structure:

```text
.
├── data/                          # DNS inputs + generated outputs (large files)
│   ├── nek/                       # Nek5000 raw/post-processed .f* files
│   ├── isocontours/               # Extracted flame fronts (HDF5)
│   ├── fields/
│   │   ├── unstructured/          # Full extracted fields (HDF5)
│   │   ├── structured_grids/      # Interpolated structured-grid fields (HDF5)
│   │   └── cnn_predictions/       # CNN prediction outputs
│   ├── processed_nek/             # Augmented Nek files (.f* + scalar maps)
│   ├── Markstein lengths/         # Markstein-analysis tabular outputs
│   └── Reference quantities/      # Reference quantities 
│
├── notebooks/
│   ├── preprocessing/
│   │   ├── nek2structured/
│   │   │   ├── nek2structured.py
│   │   │   ├── nek2structured.yaml
│   │   │   └── README.md
│   │   ├── extract_isocontours/
│   │   │   ├── extract_isocontours.py
│   │   │   ├── extract_isocontours.yaml
│   │   │   └── README.md
│   │   └── extract_fields/
│   │       ├── extract_fields.py
│   │       ├── extract_fields.yaml
│   │       └── README.md
│   ├── case_studies/
│   │   ├── plot_style.yaml        # Global plotting defaults for notebooks
│   │   ├── CNN/
│   │   ├── DMD/
│   │   ├── FDS_decomposition_analysis/
│   │   ├── Feature_selection/
│   │   ├── Model_verification/
│   │   └── tests/
│   └── Archive/                   # Older/experimental notebooks
│
├── FLAME/
│   ├── chemical_mech/
│   ├── datasets.py
│   ├── io_fields.py
│   └── io_fronts.py
│
├── pySEMTools/                    # Git submodule (core SEM functionality)
│   └── ...
│
├── report_figures/                # Generated figures from case studies
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation (venv):

```bash
git clone --recurse-submodules https://github.com/AlexandrosNtakoulas/FLAME-rf.git
cd FLAME-rf
# If you cloned earlier without --recurse-submodules:
git submodule update --init --recursive

# Set up enviroment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install library and dependencies
# Option 1: install from requirements file
pip install -r requirements.txt

# Option 2: 
pip install ipykernel
pip install cantera
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install mpi4py
pip install torch 
pip install pyvista
pip install pymech
pip install tdqm
pip install pympler
pip install memory_profiler
pip install tables
pip install h5py
pip install pydmd
pip install -e ./pySEMTools
pip install -e .

```

## Installation on Euler HPC Cluster:

```bash
git clone --recurse-submodules https://github.com/AlexandrosNtakoulas/FLAME-rf.git
cd FLAME-rf
# If you cloned earlier without --recurse-submodules:
git submodule update --init --recursive

# Load modules
module load openmpi
module load python

# Set up enviroment
python -m pip install --user --upgrade virtualenv
python -m .venv
source .venv/bin/activate

# Install library and dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Link data folder from scratch to repo in home
ln -s /cluster/scratch/<username>/FLAME-rf/data /cluster/home/<username>/FLAME-rf/data
```

For convenience set in your `~/.bashrc` file:
```bash
alias FLAME='source "$HOME/FLAME-rf/.venv/bin/activate"'
```

Then to run a file:
```bash
salloc --ntasks=32 --cpus-per-task=1 --mem-per-cpu=20G --time=01:30:00
srun -n 8 python notebooks/preprocessing/extract_fields/extract_fields.py
```

## Updating: `pySEMTools` submodule
Use this when you pull new changes from this repository and want the pinned submodule commit:

```bash
git pull
git submodule update --init --recursive
```

Use this when you want to bump `pySEMTools` to its latest upstream commit and record that update in this repository:

```bash
git submodule update --remote --recursive pySEMTools
git add pySEMTools
git commit -m "Update pySEMTools submodule"
```

## Latex Installation:
```bash
sudo apt update

sudo apt install -y \
  texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \
  texlive-latex-extra texlive-pictures texlive-plain-generic \
  texlive-base texlive-binaries dvipng ghostscript cm-super

sudo apt install -y preview-latex-style tipa

```
## Font:
Optionally, you can download the CMU Serif font from this link: https://font.download/font/cmu-serif
## Usage

### 1) Place Nek5000 output files (not tracked by git)
```text
data/nek/
└── phi0.40/
    └── h400x025_ref/
        ├── po_postPremix0.f00001   # REQUIRED: always include the first time step
        ├── po_postPremix0.f00335   # example time step you want to analyze
        └── ...
```

Notes:
- Folder structure encodes the case: `phi{PHI}/h400x{LAT_SIZE}_ref`
- File prefix depends on post-processing:
  - `POST: true` -> `po_postPremix0.fXXXXX`
  - `POST: false` -> `premix0.fXXXXX`
- Always include the first time step file (`...f00001`) in the same folder.

### 2) Extract flame fronts (HDF5 files)
1. Edit `notebooks/preprocessing/extract_isocontours/extract_isocontours.yaml` with your case settings.
2. Run `notebooks/preprocessing/extract_isocontours/extract_isocontours.py`.
To run using MPI: mpirun -n 8 python notebooks/preprocessing/extract_isocontours/extract_isocontours.py

Output example:
```text
data/isocontours/
└── phi0.40/
    └── h400x025_ref/
        ├── extracted_flame_front_post_<TIME_STEP>_iso_<ISO>.hdf5
        └── ...
```

### 3) Extract fields (HDF5 & .f* files)
1. Edit `notebooks/preprocessing/extract_fields/extract_fields.yaml`.
2. Run `notebooks/preprocessing/extract_fields/extract_fields.py`. 
To run using MPI: mpirun -n 8 python notebooks/preprocessing/extract_fields/extract_fields.py


Output example:
```text
data/fields/
└── phi0.40/
    └── h400x025_ref/
        ├── extracted_field_post_<TIME_STEP>.hdf5
        └── ...
```

### 4) Run analysis notebooks
All analysis notebooks read their parameters from the YAML files in their folder under `notebooks/case_studies/`.
For example:
- `notebooks/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.ipynb` uses `notebooks/case_studies/FDS_decomposition_analysis/FDS_decomposition_analysis.yaml`
- `notebooks/case_studies/Feature_selection/Feature_selection.ipynb` uses `notebooks/case_studies/Feature_selection/Feature_selection.yaml`
