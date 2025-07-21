import subprocess

# CUDA detection.
CUDA_AVAILABLE = False
try:
    output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
    CUDA_AVAILABLE = True
except FileNotFoundError:
    print("CUDA is not installed, falling back on NumPy instead of CuPy")
except subprocess.CalledProcessError:
    print("CUDA is not available, falling back on NumPy instead of CuPy")


if CUDA_AVAILABLE:
    import cupy as xp
    from cupyx.scipy.fft import fft2, ifft2
    from cupyx.scipy.fftpack import get_fft_plan
else:
    import numpy as xp
    from scipy.fft import fft2, ifft2

import matplotlib
from matplotlib import pyplot as plt
from darkjet import darkjet

### BEGIN ADJUSTABLE PARAMETERS ###

# Plotting
matplotlib.use("Agg")      # Headless/output files only.
# matplotlib.use("TkAgg")  # Render plots in a window ONLINE (while the simulation is running).

# Set Number of Grid Points
Nx = 512
Ny = 512

# Set Length and Width of Domain
L = 4e5
W = 4e5

# Set acceleration due to Gravity (g), Water Depth (H), Coriolis f-plane Rotation (f).
g = 0.01*9.81
H = 20.0
f = 1e-4

# Set Dispersion coefficient.
H2o6 = (H*H)/6

# Compute Grid spacings in each dimension.
dx = L / Nx
dy = W / Ny

# Build 1D grids.
x = xp.arange(0.5*dx, L, dx)
y = xp.arange(0.5*dy, W, dy)

# Build 2D grids.
xx, yy = xp.meshgrid(x, y)

# Define wavenumber grids.
dk = 2.0*xp.pi/L
k = xp.hstack((xp.arange(0, Nx/2+1), xp.arange(-Nx/2+1, 0)))*dk

dl = 2.0*xp.pi/W
l = xp.hstack((xp.arange(0, Ny/2+1), xp.arange(-Ny/2+1, 0)))*dl

# Define 2D wavenumber grid.
kk, ll = xp.meshgrid(k, l)

# Compute convenience Fourier-space factors for later.
ikk = 1j * kk
ill = 1j * ll

kk2 = kk*kk
ll2 = ll*ll

eye = xp.ones((Ny, Nx))

# Set up filter.
kmax = xp.max(kk.flatten())
lmax = xp.max(ll.flatten())

# Set critcal wave number to 10% the Nyquist value.
cutoff = 0.1

kcrit = kmax*cutoff
lcrit = lmax*cutoff

# Define our filtering kernel to be a 4th-order 'Gaussian' type exponential profile.
f_order = 4

# Set filter strength at high-wavenumbers.
epsf = 1e-16


NUMSTEPS = 1000
### END ADJUSTABLE PARAMETERS ###


# Set up filter function/kernel in x-dimension.
filtx = xp.ones((Ny, Nx))
mask = (xp.abs(kk) < kcrit)
filtx = filtx*(mask +
                 (1 - mask) *
                 xp.exp(xp.log(epsf)*((xp.abs(kk)-kcrit)/(kmax-kcrit))**f_order))

# Set up filter function/kernel in y-dimension.
filty = xp.ones((Ny, Nx))
mask = (xp.abs(ll) < lcrit)
filty = filty*(mask +
                 (1 - mask) *
                 xp.exp(xp.log(epsf)*((xp.abs(ll)-lcrit)/(lmax-lcrit))**f_order))

# Compose the two filter kernels to create a 2D kernel.
filt = filtx*filty

# Set up 2x2 matrix problem for non-hydrostatic momentum equations.
A11 = eye + H2o6*kk2
A12 = H2o6*kk*ll
A21 = A12
A22 = eye + H2o6*ll2

# Compute determinant and invert 2x2 system.
det = A11*A22 - A12*A21
A11inv = A22/det
A22inv = A11/det
A12inv = -A12/det
A21inv = -A21/det

# Define 'wiggly jet' initial free surface distribution.
eta = 0.5*xp.cosh((xx - 0.5*L - 0.05*W*xp.sin(8.0*xp.pi*yy/W))/(0.1*L))**(-2.0)

# Initialize velocities with the geostrophic state.
u =-(g/f) * xp.real(ifft2(1j * ll * fft2(eta)))
v = (g/f) * xp.real(ifft2(1j * kk * fft2(eta)))

# We can use the same FFT plan for all our FFT's, since they're all the same size.
if 'get_fft_plan' in globals():
    fftplan = get_fft_plan(eta)
else:
    fftplan = None

# Only need to '.get()' from the GPU device if we're using CuPy.
if CUDA_AVAILABLE:
    xxnp = xx.get()
    yynp = yy.get()
else:
    xxnp = xx
    yynp = yy

c0 = xp.sqrt(g*H)
CFL = 0.15

dt = float(CFL * (dx/c0))
t = 0.0
plt.figure()

# Prepare storage for multiple RHS time-levels kept in a list.
eta_rhs_levels = [xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx))]
u_rhs_levels = [xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx))]
v_rhs_levels = [xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx)), xp.zeros((Ny, Nx))]

ab_coefs = xp.array([1, 0, 0])
for nn in range(NUMSTEPS):
    if (nn % 50) == 0:
        plt.ioff()
        plt.clf()
        plt.subplot(3,1,1)
        plt.pcolor(xxnp, yynp, eta if not CUDA_AVAILABLE else eta.get())
        plt.set_cmap(darkjet)
        plt.colorbar()
        plt.title(f't={round(t*100)/100} s')
        plt.subplot(3,1,2)
        plt.pcolor(xxnp, yynp, u if not CUDA_AVAILABLE else u.get())
        plt.set_cmap(darkjet)
        plt.colorbar()
        plt.subplot(3,1,3)
        plt.pcolor(xxnp, yynp, v if not CUDA_AVAILABLE else v.get())
        plt.set_cmap(darkjet)
        plt.colorbar()
        plt.ion()
        plt.draw()
        plt.savefig(f'frame{str(nn).zfill(6)}.png')
        plt.show()
        plt.pause(1e-2)
        print(f"Outputting @ t={round(t*100)/100} s")

    # Compute conserved quantities.
    h = H + eta
    hu = h * u
    hv = h * v

    # Compute required derivatives
    hu_x = xp.real(ifft2(ikk * fft2(hu, plan=fftplan), plan=fftplan))
    hv_y = xp.real(ifft2(ill * fft2(hv, plan=fftplan), plan=fftplan))

    uhat = fft2(u, plan=fftplan)
    u_x = xp.real(ifft2(ikk * uhat, plan=fftplan))
    u_y = xp.real(ifft2(ill * uhat, plan=fftplan))

    vhat = fft2(v, plan=fftplan)
    v_x = xp.real(ifft2(ikk * vhat, plan=fftplan))
    v_y = xp.real(ifft2(ill * vhat, plan=fftplan))

    etahat = fft2(eta, plan=fftplan)
    eta_x = xp.real(ifft2(ikk * etahat, plan=fftplan))
    eta_y = xp.real(ifft2(ill * etahat, plan=fftplan))

    eta_rhs_levels[0] = hu_x + hv_y
    u_rhs_levels[0] = u*u_x + v*u_y + g*eta_x + f*v
    v_rhs_levels[0] = u*v_x + v*v_y + g*eta_y - f*u

    # Time-step PDEs (Using Forward Euler and Adams-Bashforth (AB2/AB3)).
    deta = -dt*(ab_coefs[0]*eta_rhs_levels[0] + ab_coefs[1]*eta_rhs_levels[1] + ab_coefs[2]*eta_rhs_levels[2])

    du = -dt*(ab_coefs[0]*u_rhs_levels[0] + ab_coefs[1]*u_rhs_levels[1] + ab_coefs[2]*u_rhs_levels[2])
    dv = -dt*(ab_coefs[0]*v_rhs_levels[0] + ab_coefs[1]*v_rhs_levels[1] + ab_coefs[2]*v_rhs_levels[2])

    deta_hat = fft2(deta)
    du_hat = fft2(du)
    dv_hat = fft2(dv)

    # Filter in Fourier space.
    deta_hat *= filt
    du_hat *= filt
    dv_hat *= filt

    # Evolve free-surface and velocity to next time-step using non-hydrostatic 2x2 inversion.
    eta += xp.real(ifft2(deta_hat))
    u += xp.real(ifft2(A11inv*du_hat + A12inv*dv_hat))
    v += xp.real(ifft2(A21inv*du_hat + A22inv*dv_hat))

    t += dt

    # Update linear multi-step method coefficients.
    if nn == 0:
        # AB2 (Two-step Adams-Bashforth method coefficients).
        ab_coefs = xp.array([1.5, -0.5, 0])
    elif nn == 1:
        # AB3 (Three-step Adams-Bashforth multi-step method coefficients).
        ab_coefs = xp.array([23/12, -16/12, 5/12])

    # Numerical sanity check -- have things blown up?
    if xp.any(xp.isnan(eta.flatten())):
        print("NaNs detected, terminating...")
        exit(1)

    # Rotate array references/pointers for next time-step.
    eta_rhs_levels[2] = eta_rhs_levels[1]
    eta_rhs_levels[1] = eta_rhs_levels[0]
    u_rhs_levels[2] = u_rhs_levels[1]
    u_rhs_levels[1] = u_rhs_levels[0]
    v_rhs_levels[2] = v_rhs_levels[1]
    v_rhs_levels[1] = v_rhs_levels[0]
