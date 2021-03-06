'''
Bayesian Regression for fixation durations
'''
import numpy as np
#from pymc import TruncatedNormal
from pymc import Gamma
from pymc import Model, Normal
import theano.tensor as Tns
import pymc as pm
import pylab as plt
from scipy.stats import gaussian_kde, gamma
import seaborn as sns



def traceplot(traces, thin, burn):
    '''
    Plot parameter estimates for different levels of the model
    into the same plots. Black lines are individual observers
    and red lines are mean estimates.
    '''
    variables = ['Slope1', 'Slope2', 'Offset', 'Split']
    for i, var in enumerate(variables):
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
            obs=None,
            plot=True):
    '''
    Plot the saccadic momentum effect by sampling from the
    posterior.
    '''
    res = []
    if params is None:
        params = {}
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
            if plot:
                plt.plot(x, s*y+m, color, alpha=0.1)
    return res

def predict_cond(traces, thin, burn,
            variables=None,
            condition=0,
            no_add = False):
    '''
    Plot the saccadic momentum effect by sampling from the
    posterior.
    '''
    if variables is None:
        variables = ['', 'Mean_']
    prefix = 'Cond_'
    if not no_add:
        coffsets = get_values(traces, prefix+'Offset', thin, burn)[:, condition]
        cslope1 = get_values(traces, prefix+'Slope1', thin, burn)[:, condition]
        cslope2 = get_values(traces, prefix+'Slope2', thin, burn)[:, condition]
        cbreakpoints = get_values(traces, prefix+'Split', thin, burn)[:, condition]
    else:
        coffsets = 0
        cslope1 = 0
        cslope2 = 0
        cbreakpoints = 0
    res = []
    for prefix in variables:
        if not prefix == '':
            offsets = get_values(traces, prefix+'Offset', thin, burn) + coffsets
            slope1 = get_values(traces, prefix+'Slope1', thin, burn) + cslope1
            slope2 = get_values(traces, prefix+'Slope2', thin, burn) + cslope2
            breakpoints = get_values(traces, prefix+'Split', thin, burn) + cbreakpoints
            for idx in range(len(offsets)):
                x = np.arange(180)[:, np.newaxis]
                y = piecewise_predictor(
                    x, breakpoints[idx], offsets[idx],
                    slope1[idx], slope2[idx])
                res.append(y)
        else:
            for obs in range(get_values(traces, prefix+'Offset', thin, burn).shape[1]): 
                offsets = get_values(traces, prefix+'Offset', thin, burn)[:,obs] + coffsets
                slope1 = get_values(traces, prefix+'Slope1', thin, burn)[:,obs] + cslope1
                slope2 = get_values(traces, prefix+'Slope2', thin, burn)[:,obs] + cslope2
                breakpoints = get_values(traces, prefix+'Split', thin, burn)[:,obs] + cbreakpoints
                for idx in range(len(offsets)):
                    x = np.arange(180)[:, np.newaxis]
                    y = piecewise_predictor(
                        x, breakpoints[idx], offsets[idx],
                        slope1[idx], slope2[idx])
                    res.append(y)

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


def np_gamma_params(mode=10., sd=10.):
    '''
    Converst mode and sd to shape and rate of a gamma distribution.
    '''
    var = sd**2
    rate = (mode + (mode**2 + 4*var)**.5)/(2 * var)
    shape = 1+mode*rate
    return shape, rate


def normal_model(y, x, observer):
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



def appc_gamma_model(
        y,
        observer,
        ambiguity_regressor,
        context,
        observed=True):
    '''
    Hierarchical Gamma model to predict APPC data.
    '''
    num_obs_ctxt = len(np.unique(observer[context == 1]))
    num_obs_noctxt = len(np.unique(observer[context == 0]))
    obs = [num_obs_noctxt, num_obs_ctxt]
    with Model() as pl:
        # Population level:
        obs_ambiguity = Normal('Mean_Ambiguity', mu=0, sd=50.0)

        #obs_sd_offset = pm.Gamma(
        #    'Obs_SD_Offset', *gamma_params(mode=1.0, sd=100.0))
        #obs_sd_ambiguity = pm.Gamma(
        #    'Obs_SD_ambiguity', *gamma_params(mode=1, sd=100.))
        # Define variable for sd of data distribution:
        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y.std(), sd=y.std()))
        for ctxt, label in zip([1], ['context', 'nocontext']):
            obs_offset = Normal('Mean_Offset_'+label, mu=y.mean(),
                sd=y.std()*10.0)
            # Observer level:
            #offset = Normal(
            #    'Offset_'+label, mu=obs_offset, sd=obs_sd_offset,
            #    shape=(obs[ctxt],))

            ambiguity = Normal(
                'Ambiguity_'+label, mu=obs_ambiguity, sd=20, #obs_sd_ambiguity,
                shape=(obs[ctxt],))

            # Compute predicted mode for each fixation:
            data = y[context == ctxt]
            obs_c = observer[context == ctxt]
            ambig_reg_c = ambiguity_regressor[context == ctxt]
            mode = (obs_offset + 
                    ambiguity[obs_c]*ambig_reg_c)
            # Convert to shape rate parameterization
            shape, rate = gamma_params(mode, data_sd)
            print 'Data_'+label
            data_dist = Gamma('Data_'+label, shape, rate, observed=data)
    return pl



def gamma_model(y, x, observer, observed=True):
    '''
    Hierarchical Gamma model to predict fixation durations.
    '''
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print '\n Num Observers: %d \n' % num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal(
            'Mean_Split', mu=90,
            sd=500, lower=5, upper=175)
        obs_offset = Normal('Mean_Offset', mu=y.mean(), sd=y.std()*10.0)
        obs_slopes1 = Normal('Mean_Slope1', mu=0.0, sd=1.0)
        obs_slopes2 = Normal('Mean_Slope2', mu=0, sd=1.0)

        obs_sd_split = pm.Gamma(
            'Obs_SD_Split', *gamma_params(mode=1.0, sd=100.0))

        obs_sd_intercept = pm.Gamma(
            'Obs_SD_Offset', *gamma_params(mode=1, sd=100.))

        obs_sd_slopes1 = pm.Gamma(
            'Obs_SD_slope1', *gamma_params(mode=.01, sd=2.01))
        obs_sd_slopes2 = pm.Gamma(
            'Obs_SD_slope2', *gamma_params(mode=.01, sd=2.01))

        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y.std(), sd=y.std()))

        split = TruncatedNormal(
            'Split', mu=obs_splits, sd=obs_sd_split,
            lower=5, upper=175, shape=(num_observer,))

        intercept = Normal(
            'Offset', mu=obs_offset, sd=obs_sd_intercept,
            shape=(num_observer,))
        slopes1 = Normal(
            'Slope1', mu=obs_slopes1, sd=obs_sd_slopes1,
            shape=(num_observer,))
        slopes2 = Normal(
            'Slope2', mu=obs_slopes2, sd=obs_sd_slopes2,
            shape=(num_observer,))

        mu = piecewise_predictor(
            x, split[observer], intercept[observer],
            slopes1[observer], slopes2[observer])
        shape, rate = gamma_params(mu, data_sd)
        data = Gamma('Data', shape, rate, observed=y)
        data_pred = Gamma('Predictive', shape, rate, shape = y.shape)
    return pl

def gamma_model_2conditions(y, x, observer, condition, observed=True):
    '''
    Hierarchical Gamma model to predict fixation durations.
    '''
    # Different slopes for different observers.
    num_observer = len(np.unique(observer))
    print '\n Num Observers: %d \n' % num_observer
    with Model() as pl:
        obs_splits = TruncatedNormal(
            'Mean_Split', mu=90,
            sd=500, lower=5, upper=175)
        obs_offset = Normal('Mean_Offset', mu=y.mean(), sd=y.std()*10.0)
        obs_slopes1 = Normal('Mean_Slope1', mu=0.0, sd=1.0)
        obs_slopes2 = Normal('Mean_Slope2', mu=0, sd=1.0)

        obs_sd_split = pm.Gamma(
            'Obs_SD_Split', *gamma_params(mode=1.0, sd=100.0))

        obs_sd_intercept = pm.Gamma(
            'Obs_SD_Offset', *gamma_params(mode=1, sd=100.))

        obs_sd_slopes1 = pm.Gamma(
            'Obs_SD_slope1', *gamma_params(mode=.01, sd=2.01))
        obs_sd_slopes2 = pm.Gamma(
            'Obs_SD_slope2', *gamma_params(mode=.01, sd=2.01))

        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y.std(), sd=y.std()))

        split = TruncatedNormal(
            'Split', mu=obs_splits, sd=obs_sd_split,
            lower=5, upper=175, shape=(num_observer,))

        intercept = Normal(
            'Offset', mu=obs_offset, sd=obs_sd_intercept,
            shape=(num_observer,))
        slopes1 = Normal(
            'Slope1', mu=obs_slopes1, sd=obs_sd_slopes1,
            shape=(num_observer,))
        slopes2 = Normal(
            'Slope2', mu=obs_slopes2, sd=obs_sd_slopes2,
            shape=(num_observer,))
        # Now add the condition part
        # 
        split_cond = Normal(
            'Cond_Split', mu=0, sd=10, shape=(2,))
        intercept_cond = Normal(
            'Cond_Offset', mu=0, sd=20,
            shape=(2,))
        slopes1_cond = Normal(
            'Cond_Slope1', mu=0, sd=1,
            shape=(2,))
        slopes2_cond = Normal(
            'Cond_Slope2', mu=0, sd=1,
            shape=(2,))
        intercept_cond = intercept_cond - intercept_cond.mean()
        split_cond = split_cond - split_cond.mean()
        slopes1_cond = slopes1_cond - slopes1_cond.mean()
        slopes2_cond = slopes2_cond - slopes2_cond.mean()

        mu_sub = piecewise_predictor(
            x, 
            split[observer] + split_cond[condition], 
            intercept[observer] + intercept_cond[condition],
            slopes1[observer] + slopes1_cond[condition], 
            slopes2[observer] + slopes2_cond[condition]
            )

        shape, rate = gamma_params(mu_sub, data_sd)
        data = Gamma('Data', shape, rate, observed=y)
    return pl


def sample_model(model, steps, tune=None, njobs=1, observed=['Data']):
    if tune is None:
        tune = steps/2
    with model:
        start = pm.find_MAP()  # cPickle.load(open('fixdur_map.pickle'))
        non_blocked_step = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if ('Obs_SD' in k) or ('Mean_' in k) and not (k in set(observed))],
            blocked=False)
        blocked = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if not (('Obs_SD' in k) or ('Mean_' in k))
                  and not (k in set(observed))],
            blocked=True)
        trace = pm.sample(
            steps, [non_blocked_step, blocked], start,
            tune=tune, njobs=njobs, progressbar=True)
    return trace

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


def gamma_best(y1, observer1, 
               y2, observer2):
    '''
    Hierarchical Gamma model to predict fixation durations.
    '''
    # Different slopes for different observers.
    num_observer = len(np.unique(observer1))
    print '\n Num Observers: %d \n' % num_observer
    with Model() as pl:
        obs_offset1 = Normal('Mean_Offset_1', mu=y1.mean(), sd=y1.std()*10.0)
        obs_sd_intercept1 = pm.Gamma(
            'Obs_SD_Offset_1', *gamma_params(mode=1, sd=100.))

        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y1.std(), sd=y1.std()))

        intercept1 = Normal(
            'Offset_1', mu=obs_offset1, sd=obs_sd_intercept1,
            shape=(num_observer,))

        shape1, rate1 = gamma_params(intercept1[observer1], data_sd)
        data1 = Gamma('Data_1', shape1, rate1, observed=y1)

        obs_offset2 = Normal('Mean_Offset_2', mu=y2.mean(), sd=y2.std()*10.0)
        obs_sd_intercept2 = pm.Gamma(
            'Obs_SD_Offset_2', *gamma_params(mode=1, sd=100.))

        intercept2 = Normal(
            'Offset_2', mu=obs_offset2, sd=obs_sd_intercept2,
            shape=(num_observer,))

        shape2, rate2 = gamma_params(intercept2[observer2], data_sd)
        data2 = Gamma('Data_2', shape2, rate2, observed=y2)
   
    return pl



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


def normal_test_case():
    '''
    Runs a test case with simulated data from a normal distribution.
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

    m = normal_model(dur, fa, obs)
    trace = sample_model(m, 5000)
    predict(trace, 5, 2500, {'mean': dur_mean, 'std': dur_std})
    plt.figure()
    traceplot(trace, 2, 2500)
    return dur, fa, obs, (dur_mean, dur_std), trace


def gamma_test_case():
    '''
    Runs a test case with simulated data from a normal distribution.
    '''
    obs, fa, dur = [], [], []
    delta_angle = np.arange(180)
    for n in range(15):
        mode = piecewise_predictor(
            delta_angle,
            100 + plt.randn()*20,
            250 + plt.randn()*20,
            1 + plt.randn()/2.0,
            -1 + plt.randn()/2.0)
        a, b = np_gamma_params(mode, 10)
        for _ in range(10):
            d = gamma.rvs(a=a, scale=1.0/b)
            fa.append(delta_angle)
            dur.append(d)
            obs.append(d*0+n)
    dur, fa, obs = np.concatenate(dur), np.concatenate(fa), np.concatenate(obs)
    m = gamma_model(dur, fa, obs.astype(int))
    trace = sample_model(m, 5000)
    predict(trace, 5, 2500 )
    plt.figure()
    traceplot(trace, 2, 2500)
    return dur, fa, obs, trace


def run_fixdur():
    '''
    Run the model on sample data.
    '''
    import cPickle
    dur, fa, obs, _ = cPickle.load(open('durtest.pickle'))
    dur, fa, obs = dur[obs < 11], fa[obs < 11], obs[obs < 11]
    m = gamma_model(dur, fa, obs-1)
    return sample_model(m, 15000)

def make_viz(model, size = (4,4)):
    '''
    A helper function to make a Kruschke style diagram of the model.
    It generates an iconic plot for all of the prior distributions in 
    the model.
    '''
    import theano
    for i, (name, variable) in enumerate(model.named_vars.iteritems()):
        if type(variable) == pm.model.ObservedRV:
            continue
        try:
            s = variable.distribution.variance.eval()**.5
            m = variable.distribution.mean.eval()
        except AttributeError:
            s = variable.distribution.variance**.5
            m = variable.distribution.mean
        except theano.gof.MissingInputError:
            plt.subplot(size[0], size[1], i+1)
            plt.title(name)
            plt.xticks([])
            plt.yticks([])
            continue
        x = np.linspace(m-2*s, m+2*s, 1000)
        y = variable.distribution.logp(x).eval()
        plt.subplot(size[0], size[1], i+1)
        plt.plot(x, np.exp(y))
        #plt.xticks([])
        plt.yticks([])
        plt.title(name)
        sns.despine()


def run_best_avormberg():
    '''
    Run the model on sample data.
    '''
    import cPickle
    dur, fa, obs, cond = cPickle.load(open('avormberg_data.pickle'))

    #  fixx = dm[dm.condition==1]
    #  free = dm[dm.condition==2]

    m = gamma_model_2conditions(dur, fa, obs, cond)
    return sample_model(m, 15000)

def sample_model_appc(model, steps, tune=None, njobs=1, observed=['Data']):
    if tune is None:
        tune = steps/2
    with model:
        start = pm.find_MAP() 
        non_blocked_step = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if ('Obs_SD' in k) or ('Mean_' in k) and not (k in set(observed))],
            blocked=False)
        blocked = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if not (('Obs_SD' in k) or ('Mean_' in k))
                  and not (k in set(observed))],
            blocked=False)
        print blocked, non_blocked_step
        trace = pm.sample(
            steps, [non_blocked_step, blocked], start,
            tune=tune, njobs=njobs, progressbar=True)
    return trace

def run_appc_model():
    '''
    Run appc model and load data for it...
    '''
    from scipy.io import loadmat
    data = loadmat('bayes_datamat.mat')
    context = data['datamat']['context'][0, 0][0].astype(int)
    observer = data['datamat']['subject'][0, 0][0].astype(int)
    ambiguity_regressor = data['datamat']['ambiguity'][0, 0][0]
    duration = data['datamat']['fd'][0, 0][0]
    
    print duration[context].mean()
    print duration[~context].mean()

    m = appc_gamma_model(
        duration,
        observer.astype(int),
        ambiguity_regressor,
        context)
    return sample_model_appc(m, 50000, observed=['Data_context' ])


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    # t = run_test_case() #run_fixdur()
    t = run_appc_model()
    save(t, filename)
    plt.figure()
    pm.traceplot(t)
    plt.show()
