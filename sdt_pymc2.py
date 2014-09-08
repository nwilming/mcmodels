import numpy
import pymc
from pymc import rbinomial,Binomial,Normal,Gamma
import pylab
import scipy.stats

#numpy.random.seed(15)

Nsubj = 4
Ntrls = 100

# the data
signal_resp = rbinomial(n=Ntrls, p=0.80, size=Nsubj)
noise_resp  = rbinomial(n=Ntrls, p=0.10, size=Nsubj)

# the model
Pmd = Normal('prior_md', mu=0.0, tau=0.001, value=0.0)
Pmc = Normal('prior_mc', mu=0.0, tau=0.001, value=0.0)
Ptaud = Gamma('prior_taud', alpha=0.001, beta=0.001, value=0.01)
Ptauc = Gamma('prior_tauc', alpha=0.001, beta=0.001, value=0.01)

dprm = Normal('dprime', mu=Pmd, tau=Ptaud, size=Nsubj, value=[0,0,0,0])
bias = Normal('bias',   mu=Pmc, tau=Ptauc, size=Nsubj, value=[0,0,0,0])

Phi = scipy.stats.norm.cdf

@pymc.deterministic
def hi(d=dprm, c=bias):
    return Phi(+0.5*d - c)

@pymc.deterministic
def fi(d=dprm, c=bias):
    return Phi(-0.5*d - c)

k_hi_signal = Binomial('signal_trials', n=100, p=hi, value=signal_resp, observed=True)
k_fi_noise  = Binomial('noise_trials',  n=100, p=fi, value=noise_resp,  observed=True)


if __name__ == '__main__':
    M = pymc.MCMC([Pmd,Pmc,Ptaud,Ptauc,dprm,bias,hi,fi])
    M.sample(iter=20000, burn=5000, thin=2)
#    pymc.Matplot.plot(Pmd)
#    pymc.Matplot.plot(Pmc)
#    pymc.Matplot.plot(Ptaud)
#    pymc.Matplot.plot(Ptauc)
    pymc.Matplot.plot(dprm)
    pymc.Matplot.plot(bias)
    pymc.Matplot.plot(hi)
    pymc.Matplot.plot(fi)
    pymc.gelman_rubin(M)
    pylab.show()

