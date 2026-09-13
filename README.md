# DSInference
De Simone Inference: CUDA-accelerated fuzzy regression engine.

This repository contains the reference implementation of the DSRegressor
method presented in the publication:

**De Simone Inference: A GPU-accelerated tensor-based extension of
the Wang & Mendel fuzzy rule generation method with density-based
domain division, probabilistic conflict resolution, hybrid Sugeno
inference, and two-step self-optimization.**  
*Loris De Simone, September 2026*

## Environment Setup

To run this project, the following system requirements must be installed manually:

- **Python 3.11.x**  
  (the code has been tested with Python **3.11.9**)

- **NVIDIA CUDA Toolkit 12.x**  
  (the code has been tested with CUDA **12.9**)

- **NVIDIA GPU with CUDA support**
- **Updated NVIDIA drivers**

---

### 1. Create a virtual environment

**Important:** The `python` command used to create and run the virtual environment
must point to a Python **3.11.x** installation.   
If multiple Python versions are installed, ensure that the system PATH resolves
`python` to the correct interpreter.

```bash
python -m venv .venv
```

### 2. Activate the environment:

Windows Cmd or Powershel:
```cmd
.venv\Scripts\activate
```

Linux:
```bash
source .venv/bin/activate
```

### 3. Install Python dependences

The project requires only two Python packages:

```
numpy
cupy-cuda12x
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## Starting JupyterLab from the Virtual Environment (Windows)

To start JupyterLab inside your virtual environment on Windows,
use the provided script included in this repository `launchOnWindows.bat`.

### How to use it

1. Open a terminal inside the directory of your virtual environment 
(the one containing the `Scripts/` folder). Run the command:
2. Run the script:
```cmd
launchOnWindows.bat
```
This script automatically activates the virtual environment 
and launches JupyterLab using the packages installed inside it.

---

## Starting JupyterLab from the Virtual Environment (Linux/macOS)

To start JupyterLab inside your virtual environment on Linux or macOS, 
use the provided script included in this repository: `launchOnLinux.sh`

### How to use it

1. Open a terminal inside the directory of your virtual environment  
(the one containing the bin/ folder and the launchOnLinux.sh script).
2. Make the script executable if not yet:

```bash
chmod +x launchOnLinux.sh
```

Then run it:

```bash
./launchOnLinux.sh
```

This script activates the virtual environment and starts JupyterLab 
with the environment’s installed packages.
























