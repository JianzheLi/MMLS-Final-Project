from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

params = {
    'text.usetex': False,
    'font.size': 12,
    'font.family': 'sans-serif'
}
plt.rcParams.update(params)
color_show = ['blue', 'red', 'green', 'orange'] # list of colors for the different curves


beta_show = [ 0.5, 1.1, 1.5, 2.0 ] # values of the windsock parameter to display
ks = 0.027 # spring constant for the confining potential
gam = 18. # friction coefficient of the propulsive force
eps = 0.025 # eta/gamma_p with eta the amplitude of the non-linear correction to the windsock amplification and gamma_p the friction coefficient of the propulsive force
sigma = 2. # amplitude of the noise


print("Start plotting spectrum...")
f, ax = plt.subplots()
ax.set_yscale('log')
k = 0
for be in beta_show:
    data = np.genfromtxt('./spectrum_k{0:.3f}_gammaP{1:.2f}_beta{2:.2f}_eps{3:.3f}_sigma{4:.2f}.txt'.format(ks, gam, be, eps, sigma))
    omega = data[:,0]
    spectrum = data[:,1]
    omega = omega[::2] # plot one every two points
    spectrum = spectrum[::2] # plot one every two points
    spectrum = spectrum[np.absolute(omega) < 3.] # limit the range of omega
    omega = omega[np.absolute(omega) < 3.] # limit the range of omega
    ax.plot(omega, spectrum, linewidth=0.8, linestyle = '-', c = color_show[k], label = r'$\beta=${0:.1f}'.format(be)) #, marker = '.'
    k += 1   
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$S_\mathbf{v}(\omega)$')
maj_loc = ticker.LogLocator(numticks = 9)
min_loc = ticker.LogLocator(subs = 'all', numticks = 9)
ax.yaxis.set_major_locator(maj_loc)
ax.yaxis.set_minor_locator(min_loc)
ax.legend()
plt.tight_layout(pad = 0.1, w_pad = 0.0, h_pad = -0.1)
plt.show()


print("Start plotting magnitude spectrum...")
f, ax = plt.subplots()
ax.set_yscale('log')
k = 0
for be in beta_show:
    data = np.genfromtxt('./spectrum_k{0:.3f}_gammaP{1:.2f}_beta{2:.2f}_eps{3:.3f}_sigma{4:.2f}.txt'.format(ks, gam, be, eps, sigma))
    omega = data[:,0]
    spectrum = data[:,2]
    omega = omega[::2] # plot one every two points
    spectrum = spectrum[::2] # plot one every two points
    spectrum[np.argmin(np.absolute(omega))] = np.nan # avoid overflow at zero angular frequency (the value of the spectrum is equal to the variance of v)
    spectrum = spectrum[np.absolute(omega) < 3.]
    omega = omega[np.absolute(omega) < 3.]
    ax.plot(omega, spectrum, linewidth=0.8, linestyle = '-', c = color_show[k], label = r'$\beta=${0:.1f}'.format(be)) #, marker = '.'
    k += 1   
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$S_\mathbf{v^2}(\omega)$')
maj_loc = ticker.LogLocator(numticks = 9)
min_loc = ticker.LogLocator(subs = 'all', numticks = 9)
ax.yaxis.set_major_locator(maj_loc)
ax.yaxis.set_minor_locator(min_loc)
ax.legend()
plt.tight_layout(pad = 0.1, w_pad = 0.0, h_pad = -0.1)
plt.show()


###############################
print("Start plotting spin histogram...")
x = np.array([-1, 1])
bar_width = 0.1 # width of the histogram bars
f, ax = plt.subplots()
k = 0
for be in beta_show:
    if be >= 1.: # no oscillation below the bifurcation
        data = np.genfromtxt('./proba_spin_k{0:.3f}_gammaP{1:.2f}_beta{2:.2f}_eps{3:.3f}_sigma{4:.2f}.txt'.format(ks, gam, be, eps, sigma))
        proba = data[:,1]
        ax.bar(x + ( k - len(beta_show) * 0.5 ) * bar_width, proba, width = bar_width, color = color_show[k], label = r'$\beta=${0:.1f}'.format(be))
    k += 1
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel(r'$P(\epsilon)$')
ax.set_xticks([-1, 1])
ax.set_yticks([0, 0.25, 0.5])
ax.set_yticklabels(['0','','0.5'])
ax.set_ylim(0., 0.6)
ax.legend()
plt.tight_layout(pad = 0.1, w_pad = 0.0, h_pad = -0.1)
plt.show()
