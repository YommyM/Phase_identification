## Leaflet File Information

Each leaflet file contains the leaflet assignment for all lipid residues in different systems.  
The data matrices have the following structure:

- **upper:** 0  
- **lower:** 1  

---

### **DPPC:DOPC Systems**

**Files:**  
- `dpdo280k-leaflet.xvg`  
- `dpdo290k-leaflet.xvg`  

**Dimensions:**  
`10000 × 1153`  
**Columns:**  
`[n_fr, lipid1, lipid2, …, lipid1152]`

---

### **DPPC:DOPC:Cholesterol Systems**

**Files:**  
- `dpdochl280k-leaflet.xvg`  
- `dpdochl290k-leaflet.xvg`  

**Dimensions:**  
`9000 × 1153`  
**Columns:**  
`[n_fr, lipid1, lipid2, …, lipid1152]`

---

### **PSM:DOPC:Cholesterol:POPC System**

**File:**  
- `psmdopochl300k-0.8-0-20us-leaflet.xvg`  

**Dimensions:**  
`20000 × 513`  
**Columns:**  
`[n_fr, lipid1, lipid2, …, lipid512]`

---

### **Protein–Lipid Systems**

**File:**  
- `1j4n-dpdo280k_leaflet.txt`  
  - **Dimensions:** `10000 × 1083`  
  - **Columns:** `[n_fr, lipid1, lipid2, …, lipid1082]`

**File:**  
- `1j4n-dpdochl280k_leaflet.txt`  
  - **Dimensions:** `10000 × 1063`  
  - **Columns:** `[n_fr, lipid1, lipid2, …, lipid1062]`

---

### **Notes**
- The column `n_fr` represents the frame index.  
- Each subsequent column corresponds to one lipid residue.  
- Values are **0 for the upper leaflet** and **1 for the lower leaflet**.
