import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import seaborn as sns


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# fig, ax = plt.subplots(1, 1)


num_genes = 10
num_manipulations = 4
num_timesteps = 10

# Make data.
genes = np.arange(1, num_genes + 1, 1)
manipulations = np.arange(1, num_manipulations + 1, 1)
timesteps = np.arange(1, num_timesteps + 1, 1)

X, Y = np.meshgrid(manipulations, genes)
Z =  (X * Y) ** timesteps.max()
Z = np.log10(Z)


# colors = plt.cm.viridis(np.linspace(0, 1, Z.shape[1]))
# for i in range(Z.shape[1]):
#     plt.plot(Z[:, i], color=colors[i])

# plt.xlabel("Timesteps")
# plt.ylabel("Number of experiments (log scale)")

# plt.show()
# plt.close(fig)



# X, Y = np.meshgrid(timesteps, genes)
# Z =  (X * Y) ** (manipulations)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, rstride=1, cstride=1, linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_xticks(manipulations)
ax.set_xlabel("Genetic manipulations")
ax.set_yticks(np.arange(1, num_genes + 1, 2))
ax.set_ylabel("Genes investigated")
ax.set_zticks(np.arange(0, 20, 4))
ax.set_zlabel(r"Log$_{10}$ experiments")
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5, location="left")
plt.show()

plt.close(fig)