# 9. 2D Heat Complex Geometry (Heat Exchanger, Heat2d-CG)

The 2D heat equation for a complex geometry is given by

$$\frac{\partial u}{\partial t} - \Delta u = 0.$$

The domain is defined as $\Omega \times T = ([-8,8] \times [-12,12] \setminus \cup_i R_i) \times [0,3]$.

The boundary condition is

$$-n \cdot (-c\nabla u) = g - qu.$$

Here we choose $c = 1$. The positions of large circles are

$$(\pm4, \pm3), \quad (\pm4, \pm9), \quad (0,0), \quad (0, \pm6), \quad r = 1$$

with $g = 5$ and $q = 1$. The positions of small circles are

$$(\pm3.2, \pm6), \quad (\pm3.2,0), \quad r = 0.4$$

with $g = 1$ and $q = 1$. For the rectangular boundary conditions, $g = 0.1$ and $q = 1$.

Data: [heat_complex.dat](https://disk.yandex.ru/d/fAh4G07vZeM6NA) should be placed at `examples/heat_2D_CG/data/heat_complex.dat`.

This file needs to be downloaded if you want to calculate the error between the PINN's and numeric solutions.
