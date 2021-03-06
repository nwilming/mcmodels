
import numpy as np
from pymc import TruncatedNormal, Binomial, Gamma, Deterministic, Model, T, Uniform, Exponential, Normal
import theano.tensor as Tns
import pymc as pm
from pylab import *
from scipy.stats import norm, binom

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


def predict(traces, thin, burn):
    offsets = traces['Offsets']
    slope1 = traces['Slope1']
    slope2 = traces['Slope2']
    breakpoints = traces['Breakpoint']
    try:
        offsets.keys()
        for chain in range(len(offsets)):
            ch = 'chain%d'%chain
            for idx in range(burn, len(offsets[ch]), thin):
                x = arange(180)[:,np.newaxis]
                y = piecewise_predictor(x, breakpoints[ch][idx], offsets[ch][idx],
                        slope1[ch][idx], slope2[ch][idx])
                plot(x,y, alpha=0.1)
    except AttributeError:
        for idx in range(burn, len(offsets), thin):
            x = arange(180)[:,np.newaxis]
            y = piecewise_predictor(x, breakpoints[idx], offsets[idx],
                    slope1[idx], slope2[idx])
            plot(x,y, 'k', alpha=0.1)

def predict_mean(traces, thin, burn):
    offsets = traces['Mean_offset']
    slope1 = traces['Mean_slope1']
    slope2 = traces['Mean_slope2']
    breakpoints = traces['Mean_split']
    try:
        offsets.keys()
        for chain in range(len(offsets)):
            ch = 'chain%d'%chain
            for idx in range(burn, len(offsets[ch]), thin):
                x = arange(180)[:,np.newaxis]
                y = piecewise_predictor(x, breakpoints[ch][idx], offsets[ch][idx],
                        slope1[ch][idx], slope2[ch][idx])
                plot(x,y,'r', alpha=0.1)
    except AttributeError:
        for idx in range(burn, len(offsets), thin):
            x = arange(180)[:,np.newaxis]
            y = piecewise_predictor(x, breakpoints[idx], offsets[idx],
                    slope1[idx], slope2[idx])
            plot(x,y, 'r', alpha=0.1)


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
        step = pm.Metropolis(blocked=False)
        trace = pm.sample(10000, step, start, tune=1000, njobs=3, progressbar=False)
        pm.traceplot(trace)
    show()
    return m, trace

def oneway_banova(y,X):
    # X is a design matrix with a 1 one where factor is present
    with Model() as banova:
        sigma = Uniform('SD lowest', lower=0,upper=10)
        sd_prior = pm.Gamma('SD Prior', 1.01005, 0.1005)
        offset = Normal('offset', mu=0, tau=0.001)
        alphas = Normal('alphas', mu=0.0, sd=sd_prior, shape=X.shape[0])
        betas = alphas - alphas.mean()
        betas = Deterministic('betas', betas)

        data = Normal('data', mu= offset + Tns.dot(X.T, betas),
                sd=sigma, observed=y)
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
        step = pm.Metropolis()
        #step = pm.NUTS()
        trace = pm.sample(150000, step, start, tune=1500, njobs=1, progressbar=True)
        pm.traceplot(trace[::2])
    show()


def piecewise_predictor(x, split, intercept, slope1, slope2):
    breakdummy = x<split
    reg_full = np.array([np.ones(x.shape)*intercept,
        slope1*x,
        slope2*((x-split)*breakdummy)])
    return reg_full.sum(0)


def piecewise_linear(y,x):
    with Model() as pl:
        split = Uniform('Breakpoint', lower=0, upper=180)
        intercept = Normal('Offset', mu=y.mean(), sd=y.std()*1000)
        slopes = Normal('Slope1', mu=0, sd=100, shape=2)
        mu = piecewise_predictor(x, split, intercept, slopes[0], slopes[1])
        data = Normal('Data', mu=mu, sd=y.std(), observed=y)
    return pl

def piecewise_durations(y,x,observer):
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal('Mean_split', mu=45, sd=45, lower=0, 
                upper=180)
        obs_offset = Normal('Mean_offset', mu=y.mean(), sd=y.std()*10)
        obs_slopes1 = Normal('Mean_slope1', mu=0,sd=100)
        obs_slopes2 = Normal('Mean_slope2', mu=0,sd=100)
        
        split = TruncatedNormal('Breakpoint', mu=obs_splits,
                sd=5, lower=0, upper=180, shape=(num_observer,))
        intercept = Normal('Offsets', mu=obs_offset, sd=y.std()*10, 
                shape=(num_observer,))
        slopes1 = Normal('Slope1', mu=obs_slopes1, sd=10, 
                shape=(num_observer,))
        slopes2 = Normal('Slope2', mu=obs_slopes2, sd=10, shape=(num_observer,))

        mu = piecewise_predictor(x, split[observer],
                intercept[observer], slopes1[observer], slopes2[observer])
        data = Normal('Data', mu=mu, sd=y.std(), observed=y)
    return pl

def run_fixdur():
    import cPickle
    dur, fa, obs = cPickle.load(open('durtest.pickle'))
    m = piecewise_durations(dur, fa, obs-1)
    with m:
        start = pm.find_MAP() #cPickle.load(open('fixdur_map.pickle'))
        step = pm.Metropolis(vars=[m.named_vars['Mean_offset'],
            m.named_vars['Mean_slope1'], m.named_vars['Mean_slope2'], 
            m.named_vars['Mean_split']], blocked=False)
        step2 = pm.Metropolis(vars=[m.named_vars['Slope1'], 
            m.named_vars['Slope2'], m.named_vars['Offsets'], m.named_vars['Breakpoint']])
        trace = pm.sample(5000, [step, step2], start, tune=1000, njobs=1,
                progressbar=True)
    return trace

def run_pl():
    x = arange(180)
    x = concatenate((x,x,x,x))
    xx = piecewise_predictor(x, 90, 50, 1, 2)
    y = xx + 20*(np.random.randn(len(x)))
    y = y-y.mean()
    m = piecewise_linear(y,x)
    with m:
        step = pm.Metropolis()
        #step = pm.NUTS()
        trace = pm.sample(2000, step, {}, tune=50, njobs=1, progressbar=True)
    return trace

def phi(x):
    return (1+Tns.erf(x))*.5
#    return .5 + .5*(Tns.erf(x/sqrt(2))) dasselbe wie dr[ber

def sig_detect(signal_responses, noise_responses, num_observers, num_trials):
    with Model() as model:
        md = Normal('Pr. mean discrim.', 0., tau=0.001)
        mc = Normal('Pr. mean bias',     0., tau=0.001)
        taud = Gamma('taud', 0.001, 0.001)
        tauc = Gamma('tauc', 0.001, 0.001)

        discriminability = Normal('Discriminability', mu=md, tau=taud, shape=num_observers)
        bias = Normal('Bias', mu=mc, tau=tauc, shape=num_observers)

        hi = phi( 0.5*(discriminability-bias))
        fi = phi(-0.5*(discriminability-bias))

        counts_signal = Binomial('Signal trials', num_trials, hi, observed=signal_responses)
        counts_noise  = Binomial('Noise trials',  num_trials, fi, observed=noise_responses)
    return model


def run_sig():
    signal_responses = binom.rvs(100, 0.69, size=1)
    noise_responses  = binom.rvs(100, 0.30, size=1)
    m = sig_detect(signal_responses, noise_responses, 1, 100)
    with m:
        #step = pm.Metropolis(blocked=False)
        step = pm.HamiltonianMC()
        start = pm.find_MAP()
        #start = {'Pr. mean discrim.':0.0, 'Pr. mean bias':0.0,
        #         'taud':0.001, 'tauc':0.001}
        trace = pm.sample(5000, step, start, tune=500, njobs=2)
    return trace[1000:]

if __name__ == '__main__':
    t = run_fixdur()
    save(t, 'fixdur_trace.hdf5')
    pm.traceplot(t)
    show()

