# Implementation of Various Bayesian Optimization Algorithms

Bayesian optimization(BO) is a prominent method for optimizing expensive-to-evaluate black-box functions that is widely applied to various domains from Hyperparameter Optimization to Design Problems.

Various BO methods are proposed in recent studies. I'll implement and evaluate those methods with synthetic functions to show how they works.

1. Gaussian Process(GP) with (UCB, EI, TS)

   Ensembled means and ± standard deviations of 3 samples are displayed.
   
   a. Ackley Function (dim=2, bounds=[-32.768, 32.768]^2)
   
   <img src = "https://user-images.githubusercontent.com/68262145/184596492-548d08ba-806c-42a9-9b63-5cc46b4ee55d.png" width="50%" height="50%">
