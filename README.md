# GPU Optimized Wang & Mendel Rule Induction with Inference.

This repository contains the reference implementation of the GOWM
method presented in the publication:

**A GPU-Accelerated Tensor Extension of the Wang--Mendel Method
with Density Partitioning, Self-Optimization and Architectural Variants**  
*Loris De Simone, September 2026*

The regression framework presented in the paper is implemented as the 
_DSRegressor class_ inside the _dsi_ module (originally designated 
De Simone Inference during initial development).

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

Install the required dependencies running inside the environment:

```bash
pip install -r requirements.txt
```

This project uses the modern src layout, meaning that the actual 
Python package lives inside the src/ directory. Because of this, 
Python will not automatically discover the package unless it is installed.

To make the package importable during development, you must install it in editable mode:

```bash
pip install -e .
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

## Windows Defender Notice

If you are running DSInference on Windows, it is recommended to add the
project directory, the virtual environment directory, and the CuPy cache
directory to Windows Defender exclusions. CuPy generates CUDA kernels,
temporary binaries, and cache files that may be scanned or blocked by
the antivirus, causing slowdowns or execution errors.
























