'''
Bayesian Regression for fixation durations
'''
import numpy as np
from pymc import TruncatedNormal, Gamma
from pymc import Model, Normal
import theano.tensor as Tns
import pymc as pm
import pylab as plt
from scipy.stats import gaussian_kde
import seaborn as sns


def save(traces, filename):
    '''
    Save a multitrace object to HDF5
    '''
    import h5py
    with h5py.File(filename) as f:
        for var in traces.varnames:
            data = traces.get_values(var)
            if type(data) == np.ndarray:
                f.create_dataset(var, data=data)
            else:
                a = f.create_group(var)
                for i, chain in enumerate(data):
                    a.create_dataset('chain%d' % i, data=chain)


def get_values(traces, name, thin, burn):
    '''
    Return values for variable 'name' from
    hdf5 trace object.
    '''
    vals = []
    if hasattr(traces[name], 'keys'):
        for chain in traces[name]:
            vals.append(chain[burn::thin])
        return np.hstack(vals)
    return traces[name][burn::thin]


def run_test_case():
    '''
    Runs a test case with simulated data.

    TODO: Does not work for Gamma regresion
    '''
    obs, fa, dur = [], [], []
    for n in range(15):
        d, f, o = make_test_data(
            5, split=min(plt.rand()*50+120, 170),
            intercept=plt.rand()*50 + 225,
            slope1=1 + plt.randn()/0.75, slope2=plt.randn()/.75)
        obs.append(o+n)
        fa.append(f)
        dur.append(d)
        plt.plot(f, d, 'o', alpha=0.1)

    dur, fa, obs = (np.hstack(dur)[:, np.newaxis],
                    np.hstack(fa)[:, np.newaxis],
                    np.hstack(obs)[:, np.newaxis])

    dur_mean = dur.mean()
    dur_std = dur.std()
    dur = (dur-dur_mean)/dur_std

    m = piecewise_durations(dur, fa, obs)
    with m:
        start = pm.find_MAP()  # cPickle.load(open('fixdur_map.pickle'))
        step = pm.Metropolis(
            vars=[
                m.named_vars['Mean_Offset'],
                m.named_vars['Mean_Slope1'], m.named_vars['Mean_Slope2'],
                m.named_vars['Mean_Split'], m.named_vars['Data_SD'],
                m.named_vars['Obs_SD_Split'], m.named_vars['Obs_SD_slope1'],
                m.named_vars['Obs_SD_slope2'], m.named_vars['Obs_SD_Offset']
                ],
            blocked=False)
        step2 = pm.Metropolis(
            vars=[
                m.named_vars['Slope1'],
                m.named_vars['Slope2'],
                m.named_vars['Offset'],
                m.named_vars['Split']])
        trace = pm.sample(
            5000, [step, step2], start,
            tune=2500, njobs=1, progressbar=True)
    predict(trace, 5, 2500, {'mean': dur_mean, 'std': dur_std})
    plt.figure()
    traceplot(trace, 2, 2500)
    return trace


def make_test_data(N, split=45, intercept=250, slope1=1, slope2=-1):
    '''
    Make test data for one subject.
    '''
    x = np.array(range(1, 180, 2)*10)
    fa = x
    obs = 0*x
    dur = (plt.randn(len(x))*N + piecewise_predictor(x, split,
           intercept, slope1, slope2))
    return dur, fa, obs


def traceplot(traces, thin, burn):
    '''
    Plot parameter estimates for different levels of the model
    into the same plots. Black lines are individual observers
    and red lines are mean estimates.
    '''
    variables = ['Slope1', 'Slope2', 'Offset', 'Split']
    dims = [(-100, 100), (-100, 100), (0, 1000), (0, 180)]
    for i, (var, dim) in enumerate(zip(variables, dims)):
        plt.subplot(2, 2, i+1)
        vals = get_values(traces, var, thin, burn)
        dim = (vals.min()-vals.std(), vals.max()+vals.std())
        x = plt.linspace(*dim, num=1000)
        for v in vals.T:
            a = gaussian_kde(v)
            y = a.evaluate(x)
            y = y/y.max()
            plt.plot(x, y, 'k', alpha=.5)
        try:
            vals = get_values(traces, 'Mean_'+var, thin, burn)
            a = gaussian_kde(vals)
            y = a.evaluate(x)
            y = y/y.max()
            plt.plot(x, y, 'r', alpha=.75)
        except KeyError:
            pass
        plt.ylim([0, 1.1])
        plt.yticks([0])
        sns.despine(offset=5, trim=True)
        plt.title(var)


def predict(traces, thin, burn,
            params=None,
            variables=None,
            obs=None):
    '''
    Plot the saccadic momentum effect by sampling from the
    posterior.
    '''
    res = []
    if params is None:
        params['mean'] = 0.0
        params['std'] = 1.0
    if variables is None:
        variables = ['', 'Mean_']
    for color, prefix in zip(['k', 'r'], variables):
        offsets = get_values(traces, prefix+'Offset', thin, burn)
        slope1 = get_values(traces, prefix+'Slope1', thin, burn)
        slope2 = get_values(traces, prefix+'Slope2', thin, burn)
        breakpoints = get_values(traces, prefix+'Split', thin, burn)
        if len(offsets.shape) > 1 and obs is not None:
            offsets = offsets[:, obs]
            slope1 = slope1[:, obs]
            slope2 = slope2[:, obs]
            breakpoints = breakpoints[:, obs]
        m, s = params['mean'], params['std']
        m, s = 0.0, 1.0
        for idx in range(len(offsets)):
            x = np.arange(180)[:, np.newaxis]
            y = piecewise_predictor(
                x, breakpoints[idx], offsets[idx],
                slope1[idx], slope2[idx])
            res.append(y)
            plt.plot(x, s*y+m, color, alpha=0.1)
    return res


def piecewise_predictor(x, split, intercept, slope1, slope2):
    '''
    A piecewise linear predictor.
    '''
    slope1 = slope1 - slope2
    breakdummy = x < split
    reg_full = np.array([
        slope2*x,
        slope1*((x-split)*breakdummy)]).sum(0)
    reg_full = reg_full - reg_full.mean()
    return reg_full + intercept


def gamma_params(mode=10., sd=10.):
    '''
    Converst mode and sd to shape and rate of a gamma distribution.
    '''
    var = Tns.pow(sd, 2)
    rate = (mode + Tns.pow(Tns.pow(mode, 2) + 4*var, 0.5))/(2 * var)
    shape = 1+mode*rate
    return shape, rate


def piecewise_durations(y, x, observer):
    '''
    Hierarchical Normal model to predict fixation durations.
    '''
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print '\n Num Observers: %d \n' % num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal(
            'Mean_Split', mu=90,
            sd=1000, lower=10, upper=170)
        obs_offset = Normal('Mean_Offset', mu=0, sd=1.0)
        obs_slopes1 = Normal('Mean_Slope1', mu=0, sd=1.0)
        obs_slopes2 = Normal('Mean_Slope2', mu=0, sd=1.0)

        obs_sd_split = pm.Gamma(
            'Obs_SD_Split',
            *gamma_params(mode=1.0, sd=1.0))

        obs_sd_intercept = pm.Gamma(
            'Obs_SD_Offset',
            *gamma_params(mode=1, sd=1.0))

        obs_sd_slopes1 = pm.Gamma(
            'Obs_SD_slope1', *gamma_params(mode=1, sd=1.))
        obs_sd_slopes2 = pm.Gamma(
            'Obs_SD_slope2', *gamma_params(mode=1, sd=1.))

        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=1, sd=1.))

        split = TruncatedNormal(
            'Split', mu=obs_splits,
            sd=obs_sd_split, lower=10, upper=170,
            shape=(num_observer,))

        intercept = Normal(
            'Offset', mu=obs_offset,
            sd=obs_sd_intercept,
            shape=(num_observer,))
        slopes1 = Normal(
            'Slope1', mu=obs_slopes1,
            sd=obs_sd_slopes1,
            shape=(num_observer,))
        slopes2 = Normal(
            'Slope2', mu=obs_slopes2,
            sd=obs_sd_slopes2,
            shape=(num_observer,))

        mu = piecewise_predictor(
            x,
            split[observer],
            intercept[observer],
            slopes1[observer],
            slopes2[observer])
        data = Normal('Data', mu=mu, sd=data_sd, observed=y)
    return pl


def gamma_model(y, x, observer):
    '''
    Hierarchical Gamma model to predict fixation durations.
    '''
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print '\n Num Observers: %d \n' % num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal(
            'Mean_Split', mu=90,
            sd=1000, lower=10, upper=170)
        obs_offset = Normal('Mean_Offset', mu=0.0, sd=100.0)
        obs_slopes1 = Normal('Mean_Slope1', mu=0.0, sd=1.0)
        obs_slopes2 = Normal('Mean_Slope2', mu=0, sd=1.0)

        #obs_sd_split = pm.Gamma('Obs_SD_Split', *gamma_params(mode=.01,sd=1.01))

        #obs_sd_intercept = pm.Gamma('Obs_SD_Offset',
        #        *gamma_params(mode=.01,sd=1.01))

        #obs_sd_slopes1 = pm.Gamma('Obs_SD_slope1', *gamma_params(mode=.01, sd=1.01))
        #obs_sd_slopes2 = pm.Gamma('Obs_SD_slope2', *gamma_params(mode=.01, sd=1.01))

        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=.01, sd=1.01))

        split = TruncatedNormal(
            'Split', mu=obs_splits, sd=5,  # obs_sd_split,
            lower=10, upper=170, shape=(num_observer,))

        intercept = Normal(
            'Offset', mu=obs_offset, sd=15.,  # obs_sd_intercept,
            shape=(num_observer,))
        slopes1 = Normal(
            'Slope1', mu=obs_slopes1, sd=1.,  # obs_sd_slopes1,
            shape=(num_observer,))
        slopes2 = Normal(
            'Slope2', mu=obs_slopes2, sd=1,  # obs_sd_slopes2,
            shape=(num_observer,))

        mu = piecewise_predictor(
            x, split[observer], intercept[observer],
            slopes1[observer], slopes2[observer])
        shape, rate = gamma_params(mu, data_sd)
        data = Gamma('Data', shape, rate, observed=y)
    return pl


def run_fixdur():
    '''
    Run the model on sample data.
    '''
    import cPickle
    dur, fa, obs, _ = cPickle.load(open('durtest.pickle'))
    dur = dur[obs < 10]
    fa = fa[obs < 10]
    obs = obs[obs < 10]
    m = gamma_model(dur, fa, obs-1)
    with m:
        # start = pm.find_MAP() #cPickle.load(open('fixdur_map.pickle'))
        start = {}
        step = pm.Metropolis(vars=[
            m.named_vars['Mean_Offset'],
            m.named_vars['Mean_Slope1'],
            m.named_vars['Mean_Slope2'],
            m.named_vars['Mean_Split'],
            m.named_vars['Data_SD']],
            blocked=False)
        # m.named_vars['Obs_SD_Split'], m.named_vars['Obs_SD_slope1'],
        # m.named_vars['Obs_SD_slope2'], m.named_vars['Obs_SD_Offset']], blocked=False)
        step2 = pm.Metropolis(vars=[
            m.named_vars['Slope1'],
            m.named_vars['Slope2'],
            m.named_vars['Offset'],
            m.named_vars['Split']])
        trace = pm.sample(
            10000, [step, step2], start, 
            tune=5000, njobs=1,
            progressbar=True)
    return trace


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    # t = run_test_case() #run_fixdur()
    t = run_fixdur()
    save(t, filename)
    plt.figure()
    pm.traceplot(t)
    plt.show()
