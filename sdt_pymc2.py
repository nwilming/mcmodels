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
prior_md = Normal('prior_md', mu=0.0, tau=0.001, value=0.0)
prior_mc = Normal('prior_mc', mu=0.0, tau=0.001, value=0.0)
prior_taud = Gamma('prior_taud', alpha=0.001, beta=0.001, value=0.01)
prior_tauc = Gamma('prior_tauc', alpha=0.001, beta=0.001, value=0.01)

dprm = Normal('dprm', mu=Pmd, tau=Ptaud, size=Nsubj, value=[0,0,0,0])
bias = Normal('bias', mu=Pmc, tau=Ptauc, size=Nsubj, value=[0,0,0,0])

Phi = scipy.stats.norm.cdf

@pymc.deterministic
def hi(d=dprm, c=bias):
    return Phi(+0.5*d - c)

@pymc.deterministic
def fi(d=dprm, c=bias):
    return Phi(-0.5*d - c)

k_hi_signl = Binomial('k_hi_signl',n=Ntrls,p=hi,value=signl_resp, observed=True)
k_fi_noise = Binomial('k_fi_noise',n=Ntrls,p=fi,value=noise_resp, observed=True)


if __name__ == '__main__':
    M = pymc.MCMC([prior_md,prior_mc, prior_taud,prior_tauc, dprm,bias, hi,fi])
#    for i in [1,2]:
    M.sample(iter=10000, burn=5000, thin=2)
#    pymc.Matplot.plot(prior_md)
#    pymc.Matplot.plot(prior_mc)
#    pymc.Matplot.plot(prior_taud)
#    pymc.Matplot.plot(prior_tauc)
    pymc.Matplot.plot(dprm)
    pymc.Matplot.plot(bias)
    pymc.Matplot.plot(hi)
    pymc.Matplot.plot(fi)
#    pymc.gelman_rubin(M)
    pylab.show()

