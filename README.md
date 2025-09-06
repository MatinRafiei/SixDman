# 🌐 SixDman: Optical Network Planning and Simulation Toolkit

SixDman is a **Python-based toolkit** for **optical network modeling, planning, and analysis**.  
It is designed for **researchers, telecom engineers, and students** working in **optical communication networks**.

With SixDman, you can:
- 🏗 **Model optical networks** with nodes, links, and multiple wavelength bands.  
- 📊 **Analyze optical performance** with configurable physical parameters.  
- 📡 **Plan traffic routing & capacity allocation** for large-scale networks.  
- 🎨 **Visualize simulation results** to support network design and optimization.  

---

## ✨ Features

- **Network Modeling** – Define nodes, links, and real/synthetic topologies.  
- **Optical Band Management** – Handle C-band, L-band, or custom bands.  
- **Planning & Simulation**  
  - Compute SNR/OSNR and required margins.  
  - Simulate traffic routing & wavelength assignment (RWA).  
  - Evaluate network KPIs for multi-band scenarios.  
- **Visualization Tools** for network and performance metrics.  
- **Ready for Research & Teaching** – Easy to extend and integrate into experiments.

## 📂 Project Structure


```text
sixdman/
├── src/sixdman/core      # Core classes: Network, Band, PlanningTool
├── src/sixdman/utils     # Utility functions and path handling
├── tests/                # Unit tests for each module
├── examples/             # Jupyter notebooks for simulation examples
├── data/                 # Example data files (.mat, .npz)
├── docs/                 # documentation files
├── results/              # Generated results and network KPIs
```
## 📖 Documentation

Full project documentation is available at:  
👉 [https://sixdman.readthedocs.io/en/latest/](https://sixdman.readthedocs.io/en/latest/)

It includes:
- Installation guide
- Quick start tutorials
- API reference
- Examples and advanced usage

## 🚀 Installation

1️⃣ Direct GitHub Installation (Non-editable)

  - Option 1: Using Python venv (Recommended for lightweight setup)  
  
    ```bash
    # Create virtual environment
    python -m venv .venv
    
    # Activate it
    source .venv/bin/activate    # Linux/Mac
    .venv\Scripts\activate       # Windows
    
    # Install the package
    pip install git+https://github.com/UC3M-ONDT/SixDman.git
    ```
- Option 2: Using Conda (Recommended for data science users)  

  ```bash
  # Create a conda environment with Python 3.9+
  conda create -n sixdman-env python=3.9 -y
  
  # Activate environment
  conda activate sixdman-env
  
  # Install the package
  pip install git+https://github.com/UC3M-ONDT/SixDman.git
  ```
2️⃣ Editable Install (Development Mode)

  Clone the repository:
  
  ```bash
  git clone https://github.com/UC3M-ONDT/SixDman.git
  cd sixdman
  ```

- Option 1: Using Python venv (Recommended for lightweight setup)

  ```bash
  # Create virtual environment
  python -m venv .venv
  
  # Activate it
  source .venv/bin/activate    # Linux/Mac
  .venv\Scripts\activate       # Windows
  
  # Install in editable mode
  pip install -e .
  ```
- Option 2: Using Conda (Recommended for data science users)  

  ```bash
  # Create a conda environment with Python 3.9+
  conda create -n sixdman-env python=3.9 -y
  
  # Activate environment
  conda activate sixdman-env
  
  # Install in editable mode
  pip install -e .
  ```


## ⚡ Quick Start
1. Launch basic network planning example:
   
   ```bash
    jupyter notebook examples/MAN157_Singel_Level.ipynb
   ```
3. Explore advanced multi-level network analysis:

   ```bash
    jupyter notebook examples/MAN157_Total_Level_50G.ipynb
   ```
## 🧪 Running Tests
Unit tests are located in the tests/ folder.  
Run all tests using:

```bash
pytest -v
```
## 📜 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions!  
Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📧 Contact
Maintainer: Matin Rafiei Forooshani, Farhad Arpanaei
Email: matinrafiei007@gmail.com, farhad.arpanaei@uc3m.es
