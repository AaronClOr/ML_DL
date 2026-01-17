# Manual Optimizer Implementations

This project demonstrates the mathematical foundations of various optimization algorithms used in Deep Learning. While libraries like `torch.optim` provide these out-of-the-box, these manual implementations highlight how specific calculations (like moving averages and bias correction) accelerate convergence.

## Implemented Optimizers
1. **SGD + Momentum**: Dampens oscillations and accelerates in the relevant direction.
2. **RMSprop**: Addresses the radically different scales of gradients.
3. **Adam**: Combines adaptive learning rates with momentum for robust training.
4. **Adagrad (Adaptive Gradient)**: Adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. This is achieved by dividing the learning rate by the square root of the cumulative sum of squared gradients.
5. **Adadelta**: An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to a fixed size using a decaying average of squared gradients.
6. **SGD + Nesterov Momentum**: A "look-ahead" version of momentum. Instead of calculating the gradient at the current position, it calculates the gradient at the point where the previous momentum would have taken the parameters, allowing for faster correction and reduced overshooting.