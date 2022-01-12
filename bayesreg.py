import pyro
import pyro.distributions as dist
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
import numpy as np
from pyro.infer import Predictive
import pickle
import argparse

def poiriertarantola(V, E0, B0, BP, V0):
    'Poirier-Tarantola equation from PRB 70, 224107'

    eta = (V / V0)**(1 / 3)
    squiggle = -3 * torch.log(eta)

    E = E0 + B0 * V0 * squiggle**2 / 6 * (3 + squiggle * (BP - 2))
    return E

def birch(V, E0, B0, BP, V0):
    """
    From Intermetallic compounds: Principles and Practice, Vol. I: Principles
    Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
    case where n=0
    """
    E = (E0 +
         9 / 8 * B0 * V0 * ((V0 / V)**(2 / 3) - 1)**2 +
         9 / 16 * B0 * V0 * (BP - 4) * ((V0 / V)**(2 / 3) - 1)**3)
    return E

def murnaghan(V, E0, B0, BP, V0):
    'From PRB 28,5480 (1983'
    E = E0 + B0 * V / BP * (((V0 / V)**BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1)
    return E

def vinet(V, E0, B0, BP, V0):
    'Vinet equation from PRB 70, 224107'

    eta = (V / V0)**(1 / 3)

    E = (E0 + 2 * B0 * V0 / (BP - 1)**2 *
         (2 - (5 + 3 * BP * (eta - 1) - 3 * eta) *
          torch.exp(-3 * (BP - 1) * (eta - 1) / 2)))
    return E

def antonschmidt(V, Einf, B, n, V0):
    """From Intermetallics 11, 23-32 (2003)

    Einf should be E_infinity, i.e. infinite separation, but
    according to the paper it does not provide a good estimate
    of the cohesive energy. They derive this equation from an
    empirical formula for the volume dependence of pressure,

    E(vol) = E_inf + int(P dV) from V=vol to V=infinity

    but the equation breaks down at large volumes, so E_inf
    is not that meaningful

    n should be about -2 according to the paper.

    """
    
    E = B * V0 / (n + 1) * (V / V0)**(n + 1) * (torch.log(V / V0) -
                                                (1 / (n + 1))) + Einf
    return E

def p3(V, c0, c1, c2, c3):
    'polynomial fit'
    E = c0 + c1 * V + c2 * V**2 + c3 * V**3
    return E

def sj(V, p0, p1, p2, p3):
    E = p0*(V**(-1/3))**3 + p1*(V**(-2/3)) + p2*(V**(-1/3)) + p3
    return E

def model_others(V, nrg, func):
    e0 = pyro.sample("e0", dist.Normal(-3., 1.0))
    b = pyro.sample("b", dist.Normal(0.9, 0.2))
    bp = pyro.sample("bp", dist.Normal(6., 1.0))
    v0 = pyro.sample("v0", dist.Normal(18., 3.0))
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
    mean = func(V, e0, b, bp, v0)
    with pyro.plate("data", len(V)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=nrg)

def model_as(V, nrg, func):
    einf = pyro.sample("einf", dist.Normal(1., 0.1))
    b = pyro.sample("b", dist.Normal(0.9, 0.05))
    n = pyro.sample("n", dist.Normal(-2.9, 0.1))
    v0 = pyro.sample("v0", dist.Normal(18., 0.2))
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
    mean = func(V, einf, b, n, v0)
    with pyro.plate("data", len(V)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=nrg)

def model_p3(V, nrg, func):
    c0 = pyro.sample("c0", dist.Normal(0., 1.))
    c1 = pyro.sample("c1", dist.Normal(0., 1.))
    c2 = pyro.sample("c2", dist.Normal(0., 1.))
    c3 = pyro.sample("c3", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
    mean = func(V, 27+c0, -4.5+0.1*c1, 0.23 + 1e-2*c2, -0.0036+1e-4*c3)
    with pyro.plate("data", len(V)):
        pyro.sample("obs", dist.Normal(mean,sigma), obs=nrg)

def model_sj(V, nrg, func):
    p0 = pyro.sample("p0", dist.Normal(3784., 1.))
    p1 = pyro.sample("p1", dist.Normal(-3846., 1.))
    p2 = pyro.sample("p2", dist.Normal(1282., 1.))
    p3 = pyro.sample("p3", dist.Normal(-143., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 1.0))
    mean = func(V, p0, p1, p2, p3)
    with pyro.plate("data", len(V)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=nrg)


        
def optim_vi(model, V, nrg, func, num_iters, name, gamma=1.0, guidestr='mvn'):
    assert guidestr in ['mvn', 'mf']
    if guidestr == 'mvn':
        guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    else:
        guide = AutoDiagonalNormal(model, init_loc_fn=init_to_mean)
    optimizer = torch.optim.Adam
    scheduler = optim.ExponentialLR({'optimizer': optimizer,'optim_args': {'lr': 0.001}, 'gamma': gamma})
    svi = SVI(model,
              guide,
              scheduler,
              loss = Trace_ELBO(),
              num_samples = 16)
    pyro.clear_param_store()
    
    elbos = []
    for i in range(num_iters):
        elbo = svi.step(V, nrg, func)
        elbos += [elbo]
        scheduler.step()

    plt.clf()
    plt.plot(np.array(elbos))
    plt.xlabel('Iteration')
    plt.ylabel('Negative ELBO')
    plt.savefig(f'elbo-{name}.png')

    num_samples=1000
    predictive = Predictive(model, guide=guide, num_samples=num_samples)

    svi_mvn_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
                   for k, v in predictive(V, nrg, func).items()
                   if k != "obs"}

    with open(f'{name}-svi-{guidestr}-samples.pickle', 'wb') as handle:
        pickle.dump(svi_mvn_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_plot(svi_samples,
              hmc_samples,
              deltam,
              name,
              sites=['v0', 'e0', 'b', 'sigma'],
              xlabels=['V0 ($\AA^3$)', 'E0 (eV)', 'B (GPa)', '$\sigma$ (eV)'],
              nbinsi=[5,4,6,5],
              ylabels=['Density', '', 'Density', ''],
              wspace=0.25,
              hspace=0.36,
              top=0.85,
              svistring='mvn',
              burnin_extra=0):
    if svistring=='mvn':
        svilabel = "SVI (Multivariate Normal)"
    elif svistring=='mf':
        svilabel = "SVI (Mean Field)"
    plt.rcParams.update({'font.size': 8})
    mpl.rcParams['mathtext.default'] = 'regular'
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5, 4.5))
    for i, ax in enumerate(axs.reshape(-1)):
        site = sites[i]
        if site == 'b':
            sns.distplot(160.2*svi_samples[site], ax=ax, label=svilabel)
            sns.distplot(160.2*hmc_samples[site][burnin_extra:], ax=ax, label="HMC")
        else:
            sns.distplot(svi_samples[site], ax=ax, label=svilabel)
            sns.distplot(hmc_samples[site][burnin_extra:], ax=ax, label="HMC")
        if i != 3:
            ax.axvline(x=deltam[i,0], ls='--', c='r', label='Delta Method 95% Prediction')
            ax.axvline(x=deltam[i,1], ls='--', c='r')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbinsi[i]))
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
    handles, labels = axs[0,0].get_legend_handles_labels()
    legend=fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.7, 1.0))
    plt.subplots_adjust(wspace = wspace, hspace = hspace, top = top)
    plt.savefig(f'{name}.png',
                bbox_extra_artists=(legend,),
                bbox_inches='tight', dpi=200)

def run_hmc(V, nrg, model, func, modelstr, element, num_samples=1000, warmup_steps=200, init_mean=False):
    if init_mean:
        nuts_kernel = NUTS(model, adapt_step_size=True, init_strategy=init_to_mean)
    else:
        #default is init_to_uniform.
        nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc=MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(V, nrg, func)
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    with open(f'{element}-{modelstr}-hmc-samples.pickle', 'wb') as handle:
        pickle.dump(hmc_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calculate_samples(param_samples, modelstr):
    if modelstr == 'sj':
        a = param_samples['p0']
        b = param_samples['p1']
        c = param_samples['p2']
        root = ((-b + np.sqrt(b**2 - 3 * a*c))/3/a)
        param_samples['v0'] = root**(-3)

        param_samples['e0'] = sj(param_samples['v0'], torch.Tensor(param_samples['p0']), torch.Tensor(param_samples['p1']), torch.Tensor(param_samples['p2']), torch.Tensor(param_samples['p3']))

        param_samples['b'] = root**5*2*(3*a*root + b)/9
        return param_samples

    if modelstr == 'p3':
        c0 = 27.+param_samples['c0']
        c1 = -4.5+0.1*param_samples['c1']
        c2 = 0.23+1e-2*param_samples['c2']
        c3 = -0.0036 + 1e-4*param_samples['c3']
        a = 3 * c3
        b = 2 * c2
        c = c1
        param_samples['v0'] = (-b + np.sqrt(b**2 - 4 * a*c))/2/a 

        param_samples['e0'] = p3(param_samples['v0'], torch.Tensor(c0), torch.Tensor(c1), torch.Tensor(c2), torch.Tensor(c3))

        param_samples['b'] = (2*c2 + 6*c3*param_samples['v0'])*param_samples['v0']
        return param_samples

    if modelstr == 'as':
        param_samples['e0'] = antonschmidt(param_samples['v0'], torch.Tensor(param_samples['einf']), torch.Tensor(param_samples['b']), torch.Tensor(param_samples['n']), torch.Tensor(param_samples['v0']))
        return param_samples
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--element', type=str)
    parser.add_argument('--guide', default='mvn', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--run', type=str)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_iters', default=30000, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--warmup_steps', default=200, type=int)
    parser.add_argument('--ticks', default='4-4-6-3', type=str)
    parser.add_argument('--burnin_extra', default=0, type=int)
    parser.add_argument('--init_mean_hmc', default=False, type=bool)
    args = parser.parse_args()

    warmup_steps = args.warmup_steps
    num_samples = args.num_samples
    num_iters = args.num_iters
    modelstr = args.model
    guide = args.guide
    element = args.element
    run = args.run
    gamma = args.gamma**(1/args.num_iters)
    ticks = args.ticks
    burnin_extra = args.burnin_extra
    init_mean_hmc = args.init_mean_hmc

    nbinsi = ticks.split('-')
    nbinsi = [int(tick) for tick in nbinsi]

    funcdict = {'as': antonschmidt, 'sj': sj, 'p3': p3, 'birch': birch, 'pt': poiriertarantola, 'murn': murnaghan, 'vinet': vinet}

    if modelstr == 'as':
        model = model_as
    elif modelstr == 'sj':
        model = model_sj
    elif modelstr == 'p3':
        model = model_p3
    else:
        model = model_others
    func = funcdict[modelstr]

    V = np.load(f'v-{element}.npy')
    nrg = np.load(f'nrg-{element}.npy')
    V = torch.Tensor(V)
    nrg = torch.Tensor(nrg)

    if run == 'svi':
        optim_vi(model, V, nrg, func, num_iters, f'{element}-{modelstr}', gamma=gamma, guidestr=guide)

    if run == 'hmc':
        run_hmc(V, nrg, model, func, modelstr, element, num_samples=num_samples, warmup_steps=warmup_steps, init_mean=init_mean_hmc)

    if run == 'plot':
        deltamdict = {}
        deltamdict['pd'] = {'as': np.array([[15.3006, 15.3051],[-5.2146,-5.2143], [167.286,167.827]]),
                            'sj': np.array([[15.3027, 15.3055],[-5.2148,-5.2145], [168.643,168.929]]),
                            'p3': np.array([[15.2789, 15.3833],[-5.2208,-5.2119], [170.593,188.507]]),
                            'birch': np.array([[15.3006, 15.3052],[-5.2146,-5.2143], [167.338,167.866]]),
                            'pt': np.array([[15.2945, 15.3198],[-5.2158,-5.2141], [168.979,172.418]]),
                            'murn': np.array([[15.2859, 15.3175],[-5.2150,-5.2130], [163.131,166.267]]),
                            'vinet': np.array([[15.3020, 15.3048],[-5.2147,-5.2145], [168.214,168.526]])}

        deltamdict['au'] = {'as': np.array([[17.9580, 17.9779],[-3.2221,-3.2207], [138.549,140.360]]),
                            'sj': np.array([[17.9565, 17.9628],[-3.2220,-3.2214], [140.969,141.506]]),
                            'p3': np.array([[17.8056, 18.0482],[-3.2333,-3.2169], [149.828,170.106]]),
                            'birch': np.array([[17.9575, 17.9759],[-3.2221,-3.2208], [138.887,140.408]]),
                            'pt': np.array([[17.9253, 17.9752],[-3.2238,-3.2206], [142.228,145.843]]),
                            'murn': np.array([[17.9557, 18.0125],[-3.2232,-3.2189], [135.008,139.769]]),
                            'vinet': np.array([[17.9576, 17.9689],[-3.2219,-3.2212], [139.8,140.806]])}
        with open(f'{element}-{modelstr}-svi-{guide}-samples.pickle', 'rb') as handle:
            svi_samples = pickle.load(handle)
        with open(f'{element}-{modelstr}-hmc-samples.pickle', 'rb') as handle:
            hmc_samples = pickle.load(handle)
        if modelstr in ['as', 'p3', 'sj']:
            svi_samples = calculate_samples(svi_samples, modelstr)
            hmc_samples = calculate_samples(hmc_samples, modelstr)

        make_plot(svi_samples, hmc_samples, deltamdict[element][modelstr],
                  f'{element}-{modelstr}-svi-hmc-delta', nbinsi = nbinsi, svistring=guide, burnin_extra=burnin_extra)

if __name__ == '__main__':
    main()
