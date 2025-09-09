# GNN-MetalChelators
Developed pKa and binding constant prediction models using transfer learning to generate fractional composition diagrams and evaluate metal selectivity
![](img/header2.png)
## Database
### NIST 46 Database
![](img/NIST1.png)
The NIST 46 dataset comprises extensive data on proton-ligand binding and metal-chelate binding, along with essential properties required for modeling, such as pKₐ and binding constants.
## Data preprocess
![](img/GNN1.png)
- Progress of vectorization of molecule.
- Graph Neural Networks  enable the model to effectively learn chemical structural information.

![](img/pKa_label1.png)

Using SMARTS-based5 rules to label functional groups.

## Performance
### pKa model
#### parity plot
Test RMSE = 0.69, R^2 = 0.94
<p align="center">
    <img src="img/pka_ver26_parity_plot.png" width="500">
</p>

#### Distribution plot
<p align="center">
    <img src="img/pka_combined_pka_ver26.png" width="500">
</p>

---
### Binding constant model
#### parity plot
Test RMSE = 1.61, R^2 = 0.87
<p align="center">
    <img src="img/pka.png">
</p>

#### Distribution plot
<p align="center">
    <img src="img/metal_distribution.png" width="500">
</p>

#### Parity Plot with Metal-Ions Distinguished by Color
<p align="center">
  <img src="img/metal_parity_train_by_metal.png" alt="train parity" width="45%">
  <img src="img/metal_parity_test_by_metal.png"  alt="test parity"  width="45%">
</p>
