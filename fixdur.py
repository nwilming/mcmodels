
import numpy as np
from pymc import TruncatedNormal, Binomial, Gamma, Deterministic, Model, T, Uniform, Exponential, Normal
import theano.tensor as Tns
import pymc as pm
from pylab import *
from scipy.stats import norm, binom, gaussian_kde
import seaborn as sns

def save(traces, filename):
    import h5py
    with h5py.File(filename) as f:
        for var in traces.varnames:
            data = traces.get_values(var)
            if type(data) == np.ndarray:
                f.create_dataset(var, data=data)
            else:
                a = f.create_group(var)
                for i, chain in enumerate(data):
                    a.create_dataset('chain%d'%i, data=chain)


def make_test_data(nobs = 10, split=45, intercept=250, slope1=1, slope2=-1):
    x = array(range(180)*10)
    fa, dur, obs = [],[],[]
    for n in range(nobs):
        fa.append(x)
        obs.append(0*x+n)
        dur.append(randn(len(x))*10 + piecewise_predictor(x, split, n*5+intercept, slope1, slope2))
    return hstack(dur), hstack(fa), hstack(obs)

def get_values(traces, name, thin, burn):
    vals = []
    if hasattr(traces[name], 'keys'):
        for chain in traces[name]:
            vals.append(chain[burn::thin])
        return hstack(vals)
    return traces[name][burn::thin]


def traceplot(traces, thin, burn):
    variables = ['Slope1', 'Slope2', 'Offset', 'Split']
    dims = [(-10,10), (-10,10),(0,500), (0,180)]
    for i,(var, dim) in enumerate(zip(variables, dims)):
        subplot(2,2,i+1)
        vals = get_values(traces, var, thin, burn)
        dim = (vals.min()-vals.std(), vals.max()+vals.std())
        x = linspace(*dim, num=1000)
        for v in vals.T:
            a = gaussian_kde(v)
            plot(x, a.evaluate(x), 'k', alpha=.5)
        try:
            
            vals = get_values(traces, 'Mean_'+var, thin, burn)
            a = gaussian_kde(vals)
            plot(x, a.evaluate(x), 'r', alpha=.75)
        except KeyError:
            pass
        sns.despine(trim=True)
        title(var)
            

def predict(traces, thin, burn, params = {}, variables = ['', 'Mean_']):
    res = []
    if params == {}:
        params['mean'] = 0.0
        params['std'] = 1.0
    for color, prefix in zip(['k', 'r'], variables):
        offsets = get_values(traces, prefix+'Offset', thin, burn)
        slope1 = get_values(traces, prefix+'Slope1', thin, burn)
        slope2 = get_values(traces, prefix+'Slope2', thin, burn)
        breakpoints = get_values(traces, prefix+'Split', thin, burn)

        for idx in range(len(offsets)):
                    x = arange(180)[:,np.newaxis]
                    y = piecewise_predictor(x, breakpoints[idx], offsets[idx],
                            slope1[idx], slope1[idx]-slope2[idx])
                    y = y*params['std'] + params['mean']
                    res.append(y)
                    plot(x,y,color, alpha=0.1)
    return res

def piecewise_predictor(x, split, intercept, slope1, slope2):
    breakdummy = x<split
    reg_full = np.array([np.ones(x.shape)*intercept,
        slope1*x,
        slope2*((x-split)*breakdummy)])
    return reg_full.sum(0)

def gamma_params(mode=10., sd=10.): 
    var = float(sd)**2
    mode = float(mode)
    rate = ( mode + sqrt( mode**2 + 4*var ) ) / ( 2 * var )
    shape = 1+mode*rate
    return shape, rate


    
def piecewise_durations(y,x,observer):
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print '\n Num Observers: %d \n'%num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal('Mean_Split', mu=90, 
                sd=10*50, lower=x.min(), upper=x.max())
        obs_offset = Normal('Mean_Offset', mu=0, sd=10)
        obs_slopes1 = Normal('Mean_Slope1', mu=0,sd=10)
        obs_slopes2 = Normal('Mean_Slope2', mu=0,sd=10)
        
        obs_sd_split = pm.Gamma('Obs_SD_Split', *gamma_params(mode=1,sd=5))

        obs_sd_intercept = pm.Gamma('Obs_SD_Offset', 
                *gamma_params(mode=1,sd=5))
        
        obs_sd_slopes1 = pm.Gamma('Obs_SD_slope1', *gamma_params(mode=1, sd=5.))
        obs_sd_slopes2 = pm.Gamma('Obs_SD_slope2', *gamma_params(mode=1, sd=5.))
        
        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=1, sd=10))

        split = TruncatedNormal('Split', mu=obs_splits,
                sd=obs_sd_split, lower=x.min(), upper=x.max(), shape=(num_observer,))

        intercept = Normal('Offset', mu=obs_offset, sd=obs_sd_intercept, 
                shape=(num_observer,))
        slopes1 = Normal('Slope1', mu=obs_slopes1, sd=obs_sd_slopes1, 
                shape=(num_observer,))
        slopes2 = Normal('Slope2', mu=obs_slopes2, sd=obs_sd_slopes2, 
                shape=(num_observer,))
        
        slopes1_param = slopes1-slopes2
        mu = piecewise_predictor(x, split[observer], 
                intercept[observer], slopes1_param[observer], slopes2[observer])
        data = Normal('Data', mu=mu, sd=data_sd, observed=y)
    return pl

def run_fixdur():
    import cPickle
    dur, fa, obs, params = cPickle.load(open('durtest.pickle'))
    dur_mean = dur.mean()
    dur_std = dur.std()
    dur = (dur-dur_mean)/dur_std
    m = piecewise_durations(dur, fa, obs-1)
    with m:
        start = pm.find_MAP() #cPickle.load(open('fixdur_map.pickle'))
        step = pm.Metropolis(vars=[m.named_vars['Mean_Offset'],
            m.named_vars['Mean_Slope1'], m.named_vars['Mean_Slope2'], 
            m.named_vars['Mean_Split'], m.named_vars['Data_SD'],
            m.named_vars['Obs_SD_Split'], m.named_vars['Obs_SD_slope1'],
            m.named_vars['Obs_SD_slope2'], m.named_vars['Obs_SD_Offset']], blocked=False)
        step2 = pm.Metropolis(vars=[m.named_vars['Slope1'], 
            m.named_vars['Slope2'], m.named_vars['Offset'], m.named_vars['Split']])
        trace = pm.sample(10000, [step, step2], start, tune=5000, njobs=1,
                progressbar=True)
    return trace

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    t = run_fixdur()
    save(t, filename)
    pm.traceplot(t)
    show()

