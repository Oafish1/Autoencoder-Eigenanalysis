#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import gc
from math import ceil, prod

import matplotlib.pyplot as plt
import numpy as np
import torch

import scripts.dns_specifics as DS
from model import *

np.random.seed(42)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device: \"{device}\"')


# # Load Data

# In[3]:


# %%capture --no-display
base_shape = (128, 128)
axis_array = 2 * [np.linspace(0, 2*np.pi, 128+1)[:-1]]
dx = axis_array[0][1]-axis_array[0][0]
dy = axis_array[1][1]-axis_array[1][0]
snapshots = []

# Load data
for i in range(200):
    binary = open(f'./turbulence_data/vortJP_0000.{str(i).zfill(3)}', 'rb')
    binary.seek(0)
    snap = np.fromfile(binary, dtype='float64', count=128**2).reshape(128, 128)
    # snap = DS.all_syms(snap,
    #     base_shape=base_shape,
    #     dx=dx,
    #     dy=dy,
    #     x_shift=0,
    #     m=1)
    snapshots.append(snap)

# Shift data and add for generalization
original_dataset_size = len(snapshots)
n = 9
for i in range(n):
    for snap in snapshots[:original_dataset_size]:
        snapshots.append(DS.translate_fx(snap, base_shape=base_shape, dx=dx, x_shift=i*2*np.pi/n))
    
# Format data
train_size = len(snapshots) - 5
train_idx = np.random.choice(range(len(snapshots)), train_size)
snapshots = torch.Tensor(snapshots).unsqueeze(1)[train_idx]
snapshots /= snapshots.absolute().max()
test_snapshots = snapshots[list(set(range(len(snapshots))) - set(train_idx))]


# # Visualize Data

# In[4]:


# Visualize select snapshots
axis_array = 2 * [np.linspace(0, 2 * np.pi, 128)]
fig_shape = (3, 12)
fig = plt.figure(figsize=fig_shape[::-1])
for i in range(prod(fig_shape)):
    ax = fig.add_subplot(*fig_shape, i+1)
    ax.contourf(*axis_array, snapshots[i][0])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()


# In[5]:


# Visualize shift
fig_shape = (1, 5)
fig = plt.figure(figsize=(10, 2))
for i in range(prod(fig_shape)):
    ax = fig.add_subplot(*fig_shape, i+1)
    snap = DS.translate_fx(snapshots[9][0], base_shape=base_shape, dx=dx, x_shift=(np.pi/2)*i)
    ax.contourf(*axis_array, snap)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title('; '.join([f'{x:.02f}' for x in translate_vector]))
fig.suptitle('Vorticity Under Translation')
fig.tight_layout()
plt.show()


# # Train

# In[6]:


epochs = 801
batch_size = 64
embedded_dim = 96
lr = .0003
epoch_pd = 100


# In[7]:


autoencoder = AEModel(embedded_dim=embedded_dim).to(device)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)


# In[ ]:


# %%time
try:
    # Load model
    autoencoder.load_state_dict(torch.load(f'./model_{embedded_dim}_{len(snapshots)}_{epochs-1}-{batch_size}-{lr}.h5'))
    autoencoder.eval();
except:
    # Train model
    autoencoder.train()
    batches = ceil(len(snapshots) / batch_size)
    for epoch in range(epochs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(batches):
            batch = snapshots[np.random.choice(range(len(snapshots)), batch_size)].to(device)
            _, logits = autoencoder(batch)
            loss = (logits - batch).square().sum() / prod(batch.shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % epoch_pd == 0:
            # CLI
            print(f'Epoch: {epoch} \tLoss:{float(loss.detach())}')
            # Save model
            torch.save(autoencoder.state_dict(), f'./model_{embedded_dim}_{len(snapshots)}_{epoch}-{batch_size}-{lr}.h5')
            autoencoder.eval();


# In[ ]:


# Preview reconstruction
fig_shape = (2, 12)
fig = plt.figure(figsize=fig_shape[::-1])
for i in range(fig_shape[1]):
    # Actual
    ax = fig.add_subplot(*fig_shape, i+1)
    ax.contourf(*axis_array, snapshots[i, 0, :, :], vmin=-1, vmax=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel('Real\nTurbulence')
    
    # Predicted
    ax = fig.add_subplot(*fig_shape, fig_shape[1]+i+1)
    ax.contourf(*axis_array, autoencoder(snapshots[[i]].to(device))[1].detach().cpu()[0, 0, :, :], vmin=-1, vmax=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel('Predicted\nTurbulence')
fig.tight_layout()
plt.show()


# # Eigenanalysis

# In[ ]:


autoencoder = autoencoder.cpu()


# In[ ]:


# %%time
# Construct shift operator
n = 9
alpha = 2*np.pi/n
omega = snapshots
omega_test = test_snapshots
omega_prime = torch.stack([torch.Tensor(DS.translate_fx(z, base_shape=base_shape, dx=dx, x_shift=alpha)) for z in omega]).unsqueeze(1)
E = autoencoder(omega)[0].T.detach()
E_test = autoencoder(omega_test)[0].T.detach()
E_prime = autoencoder(omega_prime)[0].T.detach()


# In[ ]:


# Solve for T
E_plus = torch.linalg.pinv(E).detach()
T = E_prime @ E_plus

# Assess fit quality
fit_quality = (torch.matmul(T, E) - E_prime).absolute().mean()
print(f'Shift operator average error: {fit_quality:.4f}')
eigenvalue_fit_quality = (torch.matmul(torch.linalg.matrix_power(T, n), E) - E).absolute().mean()
print(f'Eigenvalue constraint average error: {eigenvalue_fit_quality:.4f}')
inverse_quality = ((E_plus @ E) - torch.eye(E_plus.shape[0])).absolute().mean()
print(f'Inverse average error: {inverse_quality:.4f}')
print(f'E average value: {E.absolute().mean():.4f}')


# In[ ]:


# # %%time
# # Train model
# epochs = 10001
# epoch_pd = 2000
# batch_size = 32
# stop_err = 1e-5
# batches = ceil(len(snapshots) / batch_size)
# stop_now = False

# # Generate shifts
# try:
#     assert E_steps[-1].shape == E.shape
#     assert len(E_steps) == n
# except:
#     print('Generating shifts...')
#     E_steps = []
#     for i in range(1, n+1):
#         omega_shift = torch.stack([torch.Tensor(DS.translate_fx(z, base_shape=base_shape, dx=dx, x_shift=alpha*i)) for z in omega]).unsqueeze(1)
#         E_steps.append(autoencoder(omega_shift)[0].T.detach())

# # GPU
# # Using T
# # T = T.to(device).detach().requires_grad_(True)  # T = torch.rand(2*[E.shape[0]], requires_grad=True).to(device).detach().requires_grad_(True)
# # optimizer = torch.optim.AdamW([T], lr=.001)
# # Using E_plus
# E_plus = E_plus.to(device).detach().requires_grad_(True)  # E_plus = torch.rand(E.shape[::-1], requires_grad=True).to(device).detach().requires_grad_(True)
# optimizer = torch.optim.AdamW([E_plus], lr=.001)

# E_steps = [x.to(device) for x in E_steps]

# for epoch in range(epochs):
#     for _ in range(batches):
#         batch = snapshots[np.random.choice(range(len(test_snapshots)), batch_size)].to(device)
        
#         # Transform in series
#         loss = []
#         T = E_steps[0] @ E_plus  # Using E_plus
#         # for i, E_step in enumerate(E_steps):  # All constraints
#         for i, E_step in zip((0, n-1), (E_steps[0], E_steps[-1])):  # Only main constraints
#             loss.append((torch.matmul(torch.linalg.matrix_power(T, i+1), E_steps[-1]) - E_step).square().sum() / prod(E_step.shape))
#         # Using E_plus
#         # loss.append(((E_plus @ E_steps[-1]) - torch.eye(E_plus.shape[0]).to(device)).square().sum())  # / prod(E_steps[-1].shape))

#         # Early stop
#         if sum(loss) < stop_err:
#             stop_now = True
#             break

#         # Step
#         optimizer.zero_grad()
#         sum(loss).backward()
#         optimizer.step()
        
#     # CLI
#     if epoch % epoch_pd == 0:
#         # fancy_output = f'Epoch: {epoch} \tCombined Loss: {sum(loss):.4f} \tSO: {loss[0]:.4f} \tEC: {loss[n-1]:.4f}'
#         fancy_output = f'Epoch: {epoch} \tCombined Loss: {sum(loss):.4f} \tSO: {loss[0]:.4f} \tEC: {loss[1]:.4f}'
#         fancy_output += f'\tI: {loss[-1]:.4f}'
#         print(fancy_output)
#     # Early stop
#     if stop_now:
#         break
# # Using E_plus
# E_plus = E_plus.detach().cpu()
# T = E_prime @ E_plus
# # Using T
# # T = T.detach().cpu()

# # Assess fit quality
# fit_quality = (torch.matmul(T, E) - E_prime).absolute().mean()
# print(f'Shift operator average error: {fit_quality:.4f}')
# eigenvalue_fit_quality = (torch.matmul(torch.linalg.matrix_power(T, n), E) - E).absolute().mean()
# print(f'Eigenvalue constraint average error: {eigenvalue_fit_quality:.4f}')
# # Using E_plus
# inverse_quality = ((E_plus @ E) - torch.eye(E_plus.shape[0])).absolute().mean()
# print(f'Inverse average error: {inverse_quality:.4f}')
# print(f'E average value: {E.absolute().mean():.4f}')


# In[ ]:


# Get eigenvalues
Lambda, V = [t.detach() for t in torch.linalg.eig(T)]
W = torch.linalg.eig(T.T)[1].detach()
wavenumbers = (torch.log(Lambda) / (1j*alpha))
# tol = .1  # 1e-12
wavenumbers_unique, idx, counts = np.unique(wavenumbers.real, return_counts=True, return_index=True)
P = []
used_l = []
# for l in range(int(1+wavenumbers.real.nan_to_num().max())):
for l in wavenumbers_unique[np.argwhere(counts > 1)]:
    if np.isnan(l):
        continue
    # idx = np.argwhere(np.abs(wavenumbers.real - l) < tol).squeeze()
    # Only consider degenerate eigenvalues
    # if len(idx.shape) < 1 or idx.shape[0] < 2:
    #     continue
    idx = np.argwhere(wavenumbers.real == l[0])[0]
    used_l.append(l)
    
    # Calculate eigspace
    eigvecs = V[:, idx].detach()
    projector = (E_test.T.detach().type(torch.complex64) @ eigvecs) @ eigvecs.T
    projector = projector.unsqueeze(1)
    
    # Calculate eigspace via PCA
    # V_mean = V[:, idx].mean(dim=1)
    # Z = V[:, idx].T - V_mean
    # lam, v = torch.linalg.eig(Z.T @ Z)
    # sort_idx = list(np.argsort(lam))[::-1]
    # eigvecs = v[:, sort_idx][:, [0]].detach()
    # projector = (E_test.T.type(torch.complex64) @ eigvecs) @ eigvecs.T + E_mean
    # projector = projector.unsqueeze(1)
    
    # Literally just PCA
    # idx = range(32)
    # E_mean = E_test.T.mean(dim=0)
    # Z = E_test.T - E_mean
    # lam, v = torch.linalg.eig(Z.T @ Z)
    # sort_idx = list(np.argsort(lam))[::-1]
    # v = v[:, sort_idx]
    # eigvecs = v[:, idx].detach()
    # projector = (Z.type(torch.complex64) @ eigvecs) @ eigvecs.T + E_mean
    # projector = projector.unsqueeze(1)
    
    # Calculate eigspace via bi-orthogonal basis
    # V_l, W_l = V[:, idx], W[:, idx]
    # Q, R = torch.linalg.qr(W_l.conj().T @ V_l)
    # Xi_l = V_l @ torch.inverse(R)
    # Xi_dagger_l = (torch.inverse(Q) @ W_l.conj().T).conj().T
    # projector = (E_test.T.detach().type(torch.complex64) @ Xi_dagger_l).unsqueeze(2).expand(-1, -1, E.shape[0]) * Xi_l.T
    
    # Record
    P.append(projector)
    
print(f'Applicable wavenumbers: {[float(x) for x in used_l]}')
SP = [p.sum(dim=1) for p in P]


# In[ ]:


# Preview reconstruction
sample_num = 5
fig_shape = (len(SP)+2, omega_test.shape[0])
fig = plt.figure(figsize=fig_shape[::-1])
for i in range(fig_shape[0]):
    # Actual
    if i == 0:
        for j, vort in enumerate(omega_test[:sample_num]):
            ax = fig.add_subplot(*fig_shape, j+1)
            ax.contourf(*axis_array, vort[0, :, :], vmin=-1, vmax=1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Vort {j}')
    
    # Predicted
    elif i != fig_shape[0]-1:
        if i == 1:
            decodes = autoencoder.decoder(SP[0][:sample_num].real).detach().cpu()
        else:
            decodes = autoencoder.decoder((SP[0][:sample_num]+SP[i-1][:sample_num]).real).detach().cpu()
        for j, vort in enumerate(decodes):
            ax = fig.add_subplot(*fig_shape, i*fig_shape[1]+1+j)
            ax.contourf(*axis_array, vort[0, :, :], vmin=-1, vmax=1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'l={used_l[i-1]}')
    # Reconstructed
    else:
        decodes = autoencoder.decoder(sum(SP)[:sample_num].real).detach().cpu()
        for j, vort in enumerate(decodes):
            ax = fig.add_subplot(*fig_shape, i*fig_shape[1]+1+j)
            ax.contourf(*axis_array, vort[0, :, :], vmin=-1, vmax=1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f'Reconst')
fig.tight_layout()
plt.show()


# In[ ]:




