#Version 4 uses repeated adaptive enrichment. 
# The solution is represented as a global neural network plus a sum of local windowed corrections. Training begins with only the global model. 
# A stagnation check is applied to the recent loss history. When the loss stagnates, a new trainable windowed local model is introduced. 
# The new local model has its own trainable xL, width_fraction, and beta, so its spatial support is learned by minimizing the total PINN loss, 
# not by an explicit residual hotspot detector. After adding a new local model, the optimizer is rebuilt so that it includes the global model and 
# all previously introduced local models. The stagnation history is then reset so that another window is not introduced immediately because of 
# the old pre-enrichment loss history.
