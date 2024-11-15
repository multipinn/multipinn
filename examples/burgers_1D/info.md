# 1. Burgers 1D

The 1D burgers equation for a complex geometry is given by

$$u_t + uu_x = \nu u_{xx}, \quad \nu = \frac{0.01}{\pi}$$

The domain is defined as $\Omega \times T = [-1,1] \times [0,1]$.

The initial condition is

$$u(x,0) = - \sin (\pi x)$$

The boundary condition is

$$u(-1,t) = u(1,t) = 0$$
