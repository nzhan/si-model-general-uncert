import gpytorch
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interp1d
mpl.rcParams['mathtext.default'] = 'regular'
from scipy.optimize import brentq
import seaborn as sns

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(V, nrg, descr, training_iter=2000):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(require_grad=True)
    model = ExactGPModel(V, nrg, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    ls = []
    outputscales = []
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(V)
        # Calc loss and backprop gradients
        loss = -mll(output, nrg)
        loss.backward()
        losses += [loss.item()]
        ls += [model.covar_module.base_kernel.lengthscale.item()]
        outputscales += [model.covar_module.raw_outputscale.item()]
        optimizer.step()

    raw_noise =  model.likelihood.noise_covar.raw_noise
    constraint =  model.likelihood.noise_covar.raw_noise_constraint

    print('Transformed noise:',
          f'{constraint.transform(raw_noise).item():.6f}')

    raw_lengthscale = model.covar_module.base_kernel.raw_lengthscale
    constraint = model.covar_module.base_kernel.raw_lengthscale_constraint

    print('Transformed lengthscale:',
          f'{constraint.transform(raw_lengthscale).item():.4f}')

    raw_outputscale = model.covar_module.raw_outputscale
    constraint = model.covar_module.raw_outputscale_constraint

    print('Transformed outputscale:',
          f'{constraint.transform(raw_outputscale).item():.4f}')

    #make plot
    plt.clf()
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('- Marginal Log Likelihood')
    plt.tight_layout()
    plt.savefig(f'mll-{descr}-gp.png')
    print(f'''#+attr_org: :width 600
#+caption: MLL {descr} Gaussian process
[[./mll-{descr}-gp.png]]''')

    plt.clf()
    plt.plot(ls)
    plt.xlabel('Iteration')
    plt.ylabel('Raw Lengthscale')
    plt.tight_layout()
    plt.savefig(f'{descr}-gp-ls.png')
    print(f'''#+attr_org: :width 600
#+caption: {descr} Gaussian process raw lengthscale
[[./{descr}-gp-ls.png]]''')

    plt.clf()
    plt.plot(outputscales)
    plt.xlabel('Iteration')
    plt.ylabel('Raw Outputscale')
    plt.tight_layout()
    plt.savefig(f'{descr}-gp-opscale.png')
    print(f'''#+attr_org: :width 600
#+caption: {descr} Gaussian process raw outputscale
[[./{descr}-gp-opscale.png]]''')

def beos_gp(sample, v_test1):
    '''
    we are going to assume that the observations, deriv, and 2nd deriv are sampled at the same x points
    also going to assume that the x points are in order from smallest to largest.
    '''
    func = interp1d(v_test1, sample[:1000], fill_value='extrapolate')
    deriv = interp1d(v_test1, sample[1000:2000],fill_value='extrapolate')
    secderiv = interp1d(v_test1, sample[2000:],fill_value='extrapolate')
    vmin = brentq(deriv, v_test1[0], v_test1[-1])
    emin = func(vmin)
    secderivmin = secderiv(vmin)
    return vmin, emin, vmin*secderivmin

def get_phys(samples, v_test1):
    vmins = []
    emins = []
    bmods = []

    numsample = len(samples)

    if numsample > 1000:
        randinds = np.random.choice(np.arange(numsample), 1000, replace=False)

    else:
        randinds = np.arange(samples.shape[0])

    for i in randinds:
        a, b, c = beos_gp(samples[i], v_test1)
        vmins += [a]
        emins += [b]
        bmods += [c]

    return np.array(vmins), np.array(emins), np.array(bmods)


def make_plot_model(gpdist, hmc_samples, deltam, modeln, site, xlabel, descr, nonlin_dict = None):
    plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', '^', '*', 'D']
    if site=='b':
        factor = 160.2
    else:
        factor = 1.0
    sns.histplot(factor*gpdist, label='GP', kde=True,  linewidth=0)
    for i,hmc_sample in enumerate(hmc_samples):
        sns.histplot(factor*hmc_sample[site], label=f'{modeln[i]} HMC', kde=True, linewidth=0, color=colors[i+1])
    for i,delta in enumerate(deltam):
        plt.axvline(x=delta[0], color=colors[i+1], ls='--', label=f'{modeln[i]} Delta')
        plt.axvline(x=delta[1], color=colors[i+1], ls='--')
    if nonlin_dict is not None:
        for i, value in enumerate(nonlin_dict[site]):
            plt.plot(factor*value,5, marker=markers[i], color=colors[i+3], ls='None', label=nonlin_dict['nl_model'][i])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [7,8,0,9,1,2,3,4,5,6]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(f'{descr}-gp-{"-".join(modeln)}-{site}.png', dpi=200, bbox_inches='tight')

def make_gp_posterior(mu1, var1, samples, v_test1, descr=''):
    plt.rcParams.update({'font.size': 12})
    mpl.rcParams['mathtext.default'] = 'regular'

    fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=False, figsize=(5.5, 7.4))
    slices = [np.s_[:1000],np.s_[1000:2000], np.s_[2000:]]
    for i in range(3):
        for j in range(1000):
            ax[i].plot(v_test1, samples[j,slices[i]], c='gray', alpha=0.3, label='1000 samples', linewidth=0.7)
        try:
            y_std = np.sqrt(np.diagonal(var1[slices[i], slices[i]]))
        except:
            y_std = np.sqrt(var1[slices[i]])
        y_mean = mu1[slices[i]]
        ax[i].plot(v_test1, y_mean, 'k', label='Mean', linewidth=0.7)
        ax[i].plot(v_test1, y_mean-2*y_std, 'r', ls='--', linewidth=0.9)
        ax[i].plot(v_test1, y_mean+2*y_std, 'r', ls='--', linewidth=0.7, label=r"$\pm$ 2 std. dev.")

    ax[2].set_xlabel('V ($\AA^3$)')
    ax[0].set_ylabel('E (eV)')
    ax[1].set_ylabel(r'$\frac{d E}{d V}$ (eV/$\AA^3$)')
    ax[2].set_ylabel(r'$\frac{d^2 E}{d V^2}$ (eV/$\AA^6$)')
    handles, labels = plt.gca().get_legend_handles_labels()
    ax[2].legend([handles[-2], handles[-1], handles[0]],[labels[-2], labels[-1], labels[0]])
    ax[0].annotate("a)", xy=(-0.2, 0.93), xycoords="axes fraction")
    ax[1].annotate("b)", xy=(-0.2, 0.93), xycoords="axes fraction")
    ax[2].annotate("c)", xy=(-0.2, 0.93), xycoords="axes fraction")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'gp-posterior-{descr}.png', dpi=300)

def make_gp_posterior_extrap(mu1, var1, samples, v_test1, V, nrg, descr=''):
    plt.rcParams.update({'font.size': 12})
    mpl.rcParams['mathtext.default'] = 'regular'

    fig, ax = plt.subplots(ncols=1, nrows=1, sharex=True, sharey=False, figsize=(5.5, 4.))
    slices = [np.s_[:1000],np.s_[1000:2000], np.s_[2000:]]

    for j in range(1000):
        ax.plot(v_test1, samples[j,:1000], c='gray', alpha=0.3, label='1000 samples', linewidth=0.7)
    try:
        y_std = np.sqrt(np.diagonal(var1[:1000, :1000]))
    except:
        y_std = np.sqrt(var1[:1000])
    y_mean = mu1[:1000]
    ax.plot(v_test1, y_mean, 'k', label='Mean', linewidth=0.7)
    ax.plot(v_test1, y_mean-2*y_std, 'r', ls='--', linewidth=0.9)
    ax.plot(v_test1, y_mean+2*y_std, 'r', ls='--', linewidth=0.7, label=r"$\pm$ 2 std. dev.")
    ax.plot(V,nrg,'.', color='blue', label='Data', alpha=0.7)
    ax.axvspan(14.37, 21.28, alpha=0.5)

    ax.set_xlabel('V ($\AA^3$)')
    ax.set_ylabel('E (eV)')
    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend([handles[-3], handles[-2], handles[-1], handles[0]],[labels[-3], labels[-2], labels[-1], labels[0]])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(f'gp-posterior-{descr}.png', dpi=300)
