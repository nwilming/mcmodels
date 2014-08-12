import numpy as np
from pymc import Deterministic, Model, T, Uniform, Exponential, Normal
import theano.tensor as Tns
import pymc as pm
from pylab import *

def best(name_a, data_a, name_b, data_b):
    pooled_data = np.concatenate([data_a,data_b])
    gmean = pooled_data.mean()
    L = pooled_data.std()/1000.0
    H = pooled_data.std()*1000.0
    with Model() as best_model:
        
        mean_prior_a = Normal('%s Mean'%name_a, mu=data_a.mean(), sd=H)
        mean_prior_b = Normal('%s Mean'%name_b, mu=data_b.mean(), sd=H)

        std_prior_a = Uniform('%s Std'%name_a, lower=L,upper=H)
        std_prior_b = Uniform('%s Std'%name_b, lower=L,upper=H)

        p = Uniform('I', lower=0.,upper=1.)
        n = 1 - 29* Tns.log(1-p)
        nu_prior = Deterministic('Nu-1', n)
        
        std_a = 1.0/(std_prior_a**2)
        std_b = 1.0/(std_prior_b**2)
        
        group_one = T('Group %s'%name_a, nu_prior, mu = mean_prior_a,  
                lam=std_a, observed=data_a)
        group_two = T('Group %s'%name_b, nu_prior, mu = mean_prior_b, 
                lam=std_b, observed=data_b)
    return best_model


def run_best():
    a = np.random.randn(10)
    b = np.random.randn(10)+2
    print 'A:', a.mean(), a.std()
    print 'B:', b.mean(), b.std()
    x_eval = np.linspace(-10,10,100)
    
    m = best('A', a, 'B', b)
    start = {'A Mean': b.mean(),
        'A Std': a.std(),
        'B Mean': a.mean(),
        'B Std': b.std(),
        'Nu-1': 100}
    with m:
        step = pm.Metropolis()
        #step = pm.Slice()
        trace = pm.sample(15000, step, start, tune=500, njobs=3, progressbar=True)
        pm.traceplot(trace)
    show()
    return m, trace

def oneway_banova(y,X):
    # X is a design matrix with a 1 one where factor is present
    with Model() as banova:
        sigma = Uniform('SD lowest', lower=0,upper=10)
        sd_prior = pm.Gamma('SD Prior', 1.01005, 0.1005)
        offset = Normal('offset', mu=0, tau=0.001)
        #alphas = Normal('alphas', mu=0.0, sd=sd_prior, shape=X.shape[0])
        #betas = alphas - alphas.mean()
        #betas = Deterministic('betas', betas)

        #data = Normal('data', mu= offset + Tns.dot(X.T, betas), 
        #        sd=sigma, observed=y)
    return banova

def run_banova():
    y = 10+hstack((np.random.randn(100),np.random.randn(100)+1, 
        np.random.randn(100)+2))
    y = y-y.mean()
    y = y/y.std()
    x = concatenate(([1.0]*100,[0.0]*200))
    X = vstack((x, np.roll(x,100), np.roll(x,200)))
    m = oneway_banova(y.astype(float),X.astype(float))
    start = {'offset': 0.0,
        'alphas': array([0,1,2.])}
    with m:
        #step = pm.Metropolis()
        step = pm.NUTS()
        trace = pm.sample(15000, step, start, tune=1500, njobs=1, progressbar=True)
        pm.traceplot(trace[::2])
    show()

if __name__ == '__main__':
    run_banova()
