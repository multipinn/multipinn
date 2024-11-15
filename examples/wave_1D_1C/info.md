# 14. Wave 1D

The 1D wave equation for a complex geometry is given by

$$u_{tt} - c^2 u_{xx} = 0$$

The domain is defined as $\Omega \times T = [0,1] \times [0,1]$.

The initial conditions is

$$u(x,0) = \sin (\pi x) + \frac{1}{2} \sin (4 \pi x)$$

$$u_t(x,0) = 0$$

The boundary condition is

$$u(0,t) = u(1,t) = 0$$

The analytical solution of this problem is

$$u(x,t) = \sin (\pi x) \cos (2 \pi t) + \frac{1}{2} \sin (4 \pi x) \cos (8 \pi t)$$
