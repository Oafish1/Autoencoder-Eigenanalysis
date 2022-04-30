""" Visualize some snapshots of vorticity with symmetry transforms applied. """
import numpy as np
import matplotlib.pyplot as plt

import dns_reader as DR
import dns_specifics as DS

file_front = 'example_traj/vortJP_0000.'
file_nums = [120, 155, 160] 
file_names = [file_front + str(fn).zfill(3) for fn in file_nums]

# sym transforms
data_filter = DS.all_syms
x_shift = 0. # continuous [0, 2pi)
y_shift = 1 # discrete {0,1,...,7}

# grid
Nx = 128
Ny = 128
Ly = 2 * np.pi
Lx = Ly

x_grid = np.linspace(0, Lx, Nx+1)[:-1]
y_grid = np.linspace(0, Ly, Ny+1)[:-1]
dx = x_grid[1]-x_grid[0]
dy = y_grid[1]-y_grid[0]

reader = DR.BinaryReader()
vort_fields = []
for num in file_nums:
    file_name = file_front + str(num).zfill(3) 
    reader.load_data(num, file_name, n_data=Nx*Ny)
    vort_fields.append(
        data_filter(reader.data[num].reshape((Nx,Ny)), base_shape=(Nx,Ny),
        dx=dx, dy=dy,x_shift=x_shift, m=y_shift)
        )
    print "Snapshot dissipation / lam val = ", DS.compute_diss(vort_fields[-1], 40.)

fig = plt.figure()
for axis_number in range(len(file_nums)):
    ax = fig.add_subplot(1, len(file_nums), axis_number+1)
    ax.set_aspect('equal')
    ax.contourf(x_grid, y_grid, vort_fields[axis_number], 80, vmin=-8., vmax=8.)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()
