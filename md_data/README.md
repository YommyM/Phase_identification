# Trajectory Information

This document summarizes the simulation trajectories and residue index mappings for each system. In addition, for each system, the corresponding directory contains the parameter file (.tpr), as well as the initial and final configurations (with water molecules and ions removed).

## 🔗 Raw Data

The original simulation trajectories are publicly available on Hugging Face Datasets:

- Dataset: https://huggingface.co/datasets/yzdai/md-trajectories  
- DOI: 10.57967/hf/7451

If you use the raw data in your work, please cite:

```bibtex
@misc{yuzhuo_dai_2026,
  author       = {yuzhuo dai and Ruo-Xu Gu and SJTU},
  title        = {md-trajectories (Revision 9e56319)},
  year         = {2026},
  url          = {https://huggingface.co/datasets/yzdai/md-trajectories},
  doi          = {10.57967/hf/7451},
  publisher    = {Hugging Face}
}
```
---

## 1. DPDO Systems

### dpdo280k and dpdo290k
- **Simulation time:** 10,000 ns  
- **Residue mapping:**  
  - **DPPC:** 1–346, 577–922  
  - **DOPC:** 347–576, 923–1152  

---

## 2. DPDO–Cholesterol Systems

### dpdochl280k and dpdochl290k
- **Simulation time:**  
  - dpdochl280k: 8,000 ns  
  - dpdochl290k: 9,000 ns  
- **Residue mapping:**  
  - **DPPC:** 1–202, 577–778  
  - **DOPC:** 203–404, 779–980  
  - **Cholesterol:** 405–576, 981–1152  

---

## 3. PSM–DOPC–Chol–POPC System

### psmdopochl
- **Simulation time:** 20,000 ns  
- **Residue mapping:**  
  - **PSM:** 1–180  
  - **DOPC:** 181–324  
  - **Cholesterol:** 325–476  
  - **POPC:** 477–512  

---

## 4. Protein–Lipid Systems

### 1j4n-dpdo280k
- **Simulation time:** 10,000 ns  
- **Residue mapping:**  
  - **Protein:** 1–992  
  - **DPPC:** 993–1321, 1534–1859  
  - **DOPC:** 1322–1533, 1860–2074  

### 1j4n-dpdochl280k
- **Simulation time:** 10,000 ns  
- **Residue mapping:**  
  - **Protein:** 1–992  
  - **DPPC:** 993–1170, 1525–1703  
  - **DOPC:** 1171–1365, 1704–1896  
  - **Cholesterol:** 1366–1524, 1897–2054  

---

## Notes
- All residue numbering starts from **1**.  
- Residue ranges **A–B** include both endpoints.  
- Units for simulation time are **nanoseconds (ns)**.
