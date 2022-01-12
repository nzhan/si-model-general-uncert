from ase.eos import EquationOfState
import autograd
from autograd import hessian
import autograd.numpy as np
from scipy.stats.distributions import t

def pred_sse(params, myfunc, V, nrg):
    pred = myfunc(params, V)
    sse = np.sum((pred-nrg)**2)
    return sse

def eos_pred_mse(eos, V, nrg):
    if eos.eos_string == 'sj':
        y = eos.fit0(V**-(1 / 3))
    else:
        y = eos.func(V, *eos.eos_parameters)
    mae = np.mean(np.absolute(y - nrg))
    mse = np.mean((y-nrg)**2)
    return np.sqrt(mse), mae

#get inverse fisher information
def get_pcov(h, alpha):
    eigs0 = np.linalg.eigvalsh(h)[0]
    if (eigs0 <0):
        eps = max(1e-5, eigs0*-1.05)
    else:
        eps = 1e-5
    j = np.linalg.pinv(h + eps*np.identity(h.shape[0]))
    pcov1 = j*alpha
    u, v = np.linalg.eigh(pcov1)
    return v @ np.diag(np.maximum(u,0)) @ v.T

def get_delta_u(eosname, funcname, V, nrg):
    eos = EquationOfState(V, nrg, eosname)
    v0, e0, B = eos.fit()
    print(f'V0: {v0:.4f}\nE0: {e0:.4f}\nB: {B*160.2:.1f}')
    params = eos.eos_parameters
    h1 = hessian(pred_sse)(params, funcname, V, nrg)
    alpha = pred_sse(params, funcname, V, nrg)
    print(f'RMSE: {np.sqrt(alpha/V.shape[0]):.6f}')
    print(f'MAE: {eos_pred_mse(eos, V, nrg)[1]:.6f}\n')
    pcov = get_pcov(h1, alpha)
    tval = t.ppf(0.975, V.shape[0]-params.shape[0])
    seconf = np.sqrt(np.diagonal(pcov))
    print('Standard Error Confidences:')
    print('---------------------------')
    print(f'V0: {seconf[3]:.5f} \nE0: {seconf[0]:.5f} \nB: {seconf[1]*160.2:.3f}\n')
    print('95% Confidence Intervals:')
    print('-------------------------')
    print(f'V0: [{v0-tval*seconf[3]:.4f}, {v0+tval*seconf[3]:.4f}] \nE0: [{e0-tval*seconf[0]:.4f}, {e0+tval*seconf[0]:.4f}] \nB: [{(B-tval*seconf[1])*160.2:.3f}, {(B+tval*seconf[1])*160.2:.3f}]\n')


#following function is delta method for Anton-schmidt bc E0 is not a parameter in A-S model.

def get_delta_u_as(eosname, funcname, V, nrg):
    eos = EquationOfState(V, nrg, eosname)
    v0, e0, B = eos.fit()
    print(f'V0: {v0:.4f}\nE0: {e0:.4f}\nB: {B*160.2:.1f}')
    params = eos.eos_parameters
    h1 = hessian(pred_sse)(params, funcname, V, nrg)
    alpha = pred_sse(params, funcname, V, nrg)
    print(f'RMSE: {np.sqrt(alpha/V.shape[0]):.6f}')
    print(f'MAE: {eos_pred_mse(eos, V, nrg)[1]:.6f}\n')
    pcov = get_pcov(h1, alpha)
    tval = t.ppf(0.975, V.shape[0]-params.shape[0])
    seconf = np.sqrt(np.diagonal(pcov))

    e0 = funcname(params, params[3])
    #delta method for E0. bc E0 is not a parameter in A-S model.
    gprime = autograd.elementwise_grad(funcname, 0)(params, params[3])
    sesq = gprime @ pcov @ gprime
    seconfe0 = np.sqrt(sesq)
    seprede0 = np.sqrt(sesq + alpha/V.shape[0])

    print('Standard Error Confidences:')
    print('---------------------------')
    print(f'V0: {seconf[3]:.5f} \nE0: {seconfe0:.5f} \nB: {seconf[1]*160.2:.3f}\n')
    print('95% Confidence Intervals:')
    print('-------------------------')
    print(f'V0: [{v0-tval*seconf[3]:.4f}, {v0+tval*seconf[3]:.4f}]') 
    print(f'E0: [{e0-tval*seconfe0:.4f}, {e0+tval*seconfe0:.4f}]') 
    print(f'B: [{(B-tval*seconf[1])*160.2:.3f}, {(B+tval*seconf[1])*160.2:.3f}]\n')


#following function is delta method for P3 bc V0, E0, B are not parameters in p3 model.

def get_delta_u_p3(eosname, funcname, V, nrg):
    eos = EquationOfState(V, nrg, eosname)
    v0, e0, B = eos.fit()
    print(f'V0: {v0:.4f}\nE0: {e0:.4f}\nB: {B*160.2:.1f}')
    params = eos.eos_parameters
    h1 = hessian(pred_sse)(params, funcname, V, nrg)
    alpha = pred_sse(params, funcname, V, nrg)
    print(f'RMSE: {np.sqrt(alpha/V.shape[0]):.6f}')
    print(f'MAE: {eos_pred_mse(eos, V, nrg)[1]:.6f}\n')
    pcov = get_pcov(h1, alpha)
    tval = t.ppf(0.975, V.shape[0]-params.shape[0])
    seconf = np.sqrt(np.diagonal(pcov))

    #delta method for V0. 
    def get_v0(params):
        a = 3 * params[3]
        b = 2 * params[2]
        c = params[1]
        return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    gprime = autograd.elementwise_grad(get_v0,0)(params)
    sesq = gprime @ pcov @ gprime
    seconfv0 = np.sqrt(sesq)

    #delta method for E0.
    def get_e0(params):
        a = 3 * params[3]
        b = 2 * params[2]
        c = params[1]
        V = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return params[0] + params[1] * V + params[2] * V**2 + params[3] * V**3
    gprime = autograd.elementwise_grad(get_e0, 0)(params)
    sesq = gprime @ pcov @ gprime
    seconfe0 = np.sqrt(sesq)
    seprede0 = np.sqrt(sesq + alpha/V.shape[0])

    #delta method for B.
    def get_b(params):
        a = 3 * params[3]
        b = 2 * params[2]
        c = params[1]
        V = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        return (2*params[2] + 6*params[3]*V)*V
    gprime = autograd.elementwise_grad(get_b, 0)(params)
    sesq = gprime @ pcov @ gprime
    seconfb = np.sqrt(sesq)

    print('Standard Error Confidences:')
    print('---------------------------')
    print(f'V0: {seconfv0:.5f} \nE0: {seconfe0:.5f} \nB: {seconfb*160.2:.3f}\n')
    print('95% Confidence Intervals:')
    print('-------------------------')
    print(f'V0: [{v0-tval*seconfv0:.4f}, {v0+tval*seconfv0:.4f}]') 
    print(f'E0: [{e0-tval*seconfe0:.4f}, {e0+tval*seconfe0:.4f}]') 
    print(f'B: [{(B-tval*seconfb)*160.2:.3f}, {(B+tval*seconfb)*160.2:.3f}]\n')

#following function is delta method for SJ bc V0, E0, B are not parameters in SJ model.
    
def get_delta_u_sj(eosname, funcname, V, nrg):
    eos = EquationOfState(V, nrg, eosname)
    v0, e0, B = eos.fit()
    print(f'V0: {v0:.4f}\nE0: {e0:.4f}\nB: {B*160.2:.1f}')
    params = eos.fit0.c
    h1 = hessian(pred_sse)(params, funcname, V, nrg)
    alpha = pred_sse(params, funcname, V, nrg)
    print(f'RMSE: {np.sqrt(alpha/V.shape[0]):.6f}')
    print(f'MAE: {eos_pred_mse(eos, V, nrg)[1]:.6f}\n')
    pcov = get_pcov(h1, alpha)
    tval = t.ppf(0.975, V.shape[0]-params.shape[0])
    seconf = np.sqrt(np.diagonal(pcov))

    #delta method for V0. 
    def get_v0(params):
        a = params[0]
        b = params[1]
        c = params[2]
        root = (-b + np.sqrt(b**2 - 3 * a * c)) / (3 * a)
        return root**(-3)
    gprime = autograd.elementwise_grad(get_v0,0)(params)
    sesq = gprime @ pcov @ gprime
    seconfv0 = np.sqrt(sesq)

    #delta method for E0.
    def get_e0(params):
        a = params[0]
        b = params[1]
        c = params[2]
        root = (-b + np.sqrt(b**2 - 3 * a * c)) / (3 * a)
        return a*root**3 + b*root**2 + c*root + params[3]

    #get pretty much the same answer if you use v0 directly.
    gprime = autograd.elementwise_grad(get_e0, 0)(params)
    #gprime = autograd.elementwise_grad(funcname, 0)(params, v0)
    sesq = gprime @ pcov @ gprime
    seconfe0 = np.sqrt(sesq)
    seprede0 = np.sqrt(sesq + alpha/V.shape[0])

    #delta method for B.
    def get_b(params):
        a = params[0]
        b = params[1]
        c = params[2]
        root = (-b + np.sqrt(b**2 - 3 * a * c)) / (3 * a)
        return root**5*2*(3*a*root + b)/9

    gprime = autograd.elementwise_grad(get_b, 0)(params)
    sesq = gprime @ pcov @ gprime
    seconfb = np.sqrt(sesq)

    print('Standard Error Confidences:')
    print('---------------------------')
    print(f'V0: {seconfv0:.5f} \nE0: {seconfe0:.5f} \nB: {seconfb*160.2:.3f}\n')
    print('95% Confidence Intervals:')
    print('-------------------------')
    print(f'V0: [{v0-tval*seconfv0:.4f}, {v0+tval*seconfv0:.4f}]') 
    print(f'E0: [{e0-tval*seconfe0:.4f}, {e0+tval*seconfe0:.4f}]') 
    print(f'B: [{(B-tval*seconfb)*160.2:.3f}, {(B+tval*seconfb)*160.2:.3f}]\n')


###########################
# all EOS models
###########################

def sj(params, V):
    p = np.poly1d(params)
    E = p(V**(-1/3))
    return E


def antonschmidt(params, V):
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
    Einf = params[0]
    B = params[1]
    n = params[2]
    V0 = params[3]
    
    E = B * V0 / (n + 1) * (V / V0)**(n + 1) * (np.log(V / V0) -
                                                (1 / (n + 1))) + Einf
    return E


def p3(params, V):
    'polynomial fit'
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]

    E = c0 + c1 * V + c2 * V**2 + c3 * V**3
    return E


def murnaghan(params, V):
    'From PRB 28,5480 (1983'

    E0 = params[0]
    B0 = params[1]
    BP = params[2]
    V0 = params[3]

    E = E0 + B0 * V / BP * (((V0 / V)**BP) / (BP - 1) + 1) - V0 * B0 / (BP - 1)
    return E


def birch(params, V):
    """
    From Intermetallic compounds: Principles and Practice, Vol. I: Principles
    Chapter 9 pages 195-210 by M. Mehl. B. Klein, D. Papaconstantopoulos
    paper downloaded from Web

    case where n=0
    """
    E0 = params[0]
    B0 = params[1]
    BP = params[2]
    V0 = params[3]

    E = (E0 +
         9 / 8 * B0 * V0 * ((V0 / V)**(2 / 3) - 1)**2 +
         9 / 16 * B0 * V0 * (BP - 4) * ((V0 / V)**(2 / 3) - 1)**3)
    return E

def poiriertarantola(params, V):
    'Poirier-Tarantola equation from PRB 70, 224107'

    E0 = params[0]
    B0 = params[1]
    BP = params[2]
    V0 = params[3]

    eta = (V / V0)**(1 / 3)
    squiggle = -3 * np.log(eta)

    E = E0 + B0 * V0 * squiggle**2 / 6 * (3 + squiggle * (BP - 2))
    return E

def vinet(params, V):
    'Vinet equation from PRB 70, 224107'

    E0 = params[0]
    B0 = params[1]
    BP = params[2]
    V0 = params[3]

    eta = (V / V0)**(1 / 3)

    E = (E0 + 2 * B0 * V0 / (BP - 1)**2 *
         (2 - (5 + 3 * BP * (eta - 1) - 3 * eta) *
          np.exp(-3 * (BP - 1) * (eta - 1) / 2)))
    return E
