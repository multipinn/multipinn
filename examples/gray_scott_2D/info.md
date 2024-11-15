# 17. 2D Diffusion-Reaction Gray-Scott Model (GS)

The governing PDE is

$$u_t = \varepsilon_1 \Delta u + b(1-u) - uv^2$$

$$v_t = \varepsilon_2 \Delta v - dv + uv^2$$

The domain is $\Omega \times T = [-1,1]^2 \times [0,200]$ and parameters are

$b = 0.04, d = 0.1, \varepsilon_1 = 1 \times 10^{-5}, \varepsilon_2 = 5 \times 10^{-6}$.

The initial conditions are

$$u(x,y,0) = 1 - \exp(-80((x + 0.05)^2 + (y + 0.02)^2))$$

$$v(x,y,0) = \exp(-80((x - 0.05)^2 + (y - 0.02)^2))$$

Data: [grayscott.dat](https://raw.githubusercontent.com/i207M/PINNacle/refs/heads/main/ref/grayscott.dat)

Then, use `convert_dat_to_files.py` to create points and values files.
