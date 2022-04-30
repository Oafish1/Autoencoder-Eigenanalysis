""" Routines for interacting specifically with 2D Kolmogorov vorticity fields. """
import numpy as np
import numpy.fft as FT

def translate_fx(z, base_shape=(-1,), dx=1., x_shift=0.):
    """ Shift a vorticity field z in x by x_shift.
        
    Args:
        z: physical vorticity field; 1D NumPy array with Nx*Ny components.
        base_shape: integer tuple of form (Nx, Ny)
        dx: float grid spacing in x
        x_shift: float shift
        
    Returns:
        2D NumPy array of shape (Nx, Ny) with shift applied.
    """
    vort = np.reshape(z, base_shape)
    k = 2 * np.pi * FT.fftfreq(base_shape[0], dx)
    fft_x = FT.fft(vort, axis=1)

    fft_shift = fft_x * np.exp(1j * k * x_shift)
    return np.real(FT.ifft(fft_shift, axis=1))

def shift_reflect(z, base_shape=(-1,), dx=1., dy=1., m=0, n=4):
    """ Apply an integer number m of discrete shift+reflects in y.
    
    Args:
        z: physical vorticity field; 1D NumPy array with Nx*Ny components.
        base_shape: integer tuple of form (Nx, Ny)
        dx: float grid spacing in x
        dy: float grid spacing in y
        m: integer number of shift+reflects to apply
        n: integer forcing wavenumber (sin(n * y) in x-m'tum.)

    Returns:
        2D NumPy array of shape (Nx, Ny) with shift+reflect applied.
    """
    vort = np.reshape(z, base_shape)
    l = 2 * np.pi * FT.fftfreq(base_shape[1], dy)
    fft_y = FT.fft(vort, axis=0).T
    fft_sr = (fft_y * np.exp(np.pi*1j * (float(m)/n) * l)).T
    phys_no_ref = (-1.)**m * np.real(FT.ifft(fft_sr, axis=0)) # not a valid field

    # now apply reflect
    fft_x = FT.fft(phys_no_ref, axis=1)
    if m%2!=0: 
        return np.real(FT.ifft(fft_x.conj(), axis=1))
    else:
        return np.real(FT.ifft(fft_x, axis=1))

def all_syms(z, base_shape=(-1,), dx=1., dy=1., x_shift=0., m=0, n=4):
    """ Apply continuous shift x_shift in x and 
        an integer number m of discrete shift+reflects in y.
    
        Args:
            z: physical vorticity field; 1D NumPy array with Nx*Ny components.
            base_shape: integer tuple of form (Nx, Ny)
            dx: float grid spacing in x
            dy: float grid spacing in y
            x_shift: float shift
            m: integer number of shift+reflects to apply
            n: integer forcing wavenumber (sin(n * y) in x-m'tum.)

        Returns:
            2D NumPy array of shape (Nx, Ny) with shift+reflect applied.
    """
    shifted = translate_fx(z, base_shape=base_shape, dx=dx, x_shift=x_shift)
    return shift_reflect(np.reshape(shifted, (-1,)), 
            base_shape=base_shape, dx=dx, dy=dy, m=m, n=n)

def compute_diss(omega, Re, n=4):
    """ Compute total dissipation rate (relative to laminar value) 
    of a 2D vorticity field.
    
    Args:
        omega: 2D NumPy array of shape (Nx, Ny)
        Re: float Reynolds number
        n: integer forcing wavenumber (sin(n * y) in x-m'tum.

    Returns:
        float dissipation / laminar dissipation.
    """
    Nx, Ny = omega.shape
    fft_field = FT.fft2(omega)
    x_inds = range(Nx-Nx/3,Nx) + range(Nx/3+1)
    y_inds = range(Ny-Ny/3,Ny) + range(Ny/3+1)

    filter_field = fft_field[x_inds]
    filter_field = filter_field[:,y_inds] / (Nx*Ny)
    Ni, Nj = filter_field.shape

    sum_fourier_coeffs2 = 0.
    for i in range(Ni):
        for j in range(Nj):
            sum_fourier_coeffs2 += filter_field[i,j] * filter_field[i,j].conj()
    diss = np.real(sum_fourier_coeffs2 / Re)
    laminar_diss = Re / (2* n**2)
    return diss / laminar_diss
