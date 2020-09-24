# afe_network_gen
Project repository for generating statistically optimal perturbation network using machine-learned perturbation difficulties.

General workflow:   
- Train NNs on 1ns solvated simulations of RBFE 
- Predict perturbation difficulty for fully connected network of presented test set
- Use DiffNet to generate statistically-optimal perturbation network
