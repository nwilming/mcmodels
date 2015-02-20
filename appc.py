'''
Bayesian Regression for fixation durations
'''
import numpy as np
from pymc import Gamma
from pymc import Model, Normal, Deterministic
import theano.tensor as Tns
import pymc as pm
import pylab as plt
from scipy.stats import gaussian_kde, gamma
import seaborn as sns


class TruncatedNormal(pm.Continuous):

    """
    Normal log-likelihood.

    """

    def __init__(
            self, mu=0.0, tau=None, sd=None, lower=0, upper=1, *args, **kwargs):
        super(TruncatedNormal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau = pm.distributions.continuous.get_tau(tau=tau, sd=sd)
        self.variance = 1. / self.tau
        self.lower = lower
        self.upper = upper

    def logp(self, value):
        tau = self.tau
        mu = self.mu
        lower, upper = self.lower, self.upper
        return pm.distributions.continuous.bound((-tau * (value - mu) ** 2 + np.log(tau / np.pi / 2.)) / 2.,
                        tau > 0, value >= lower, value <= upper)


def traceplot(traces, thin, burn):
    '''
    Plot parameter estimates for different levels of the model
    into the same plots. Black lines are individual observers
    and red lines are mean estimates.
    '''
    variables = ['Slope1', 'Slope2', 'Offset', 'Split']
    for i, var in enumerate(variables):
        plt.subplot(2, 2, i + 1)
        vals = get_values(traces, var, thin, burn)
        dim = (vals.min() - vals.std(), vals.max() + vals.std())
        x = plt.linspace(*dim, num=1000)
        for v in vals.T:
            a = gaussian_kde(v)
            y = a.evaluate(x)
            y = y / y.max()
            plt.plot(x, y, 'k', alpha=.5)
        try:
            vals = get_values(traces, 'Mean_' + var, thin, burn)
            a = gaussian_kde(vals)
            y = a.evaluate(x)
            y = y / y.max()
            plt.plot(x, y, 'r', alpha=.75)
        except KeyError:
            pass
        plt.ylim([0, 1.1])
        plt.yticks([0])
        sns.despine(offset=5, trim=True)
        plt.title(var)


def gamma_params(mode=10., sd=10.):
    '''
    Converst mode and sd to shape and rate of a gamma distribution.
    '''
    var = Tns.pow(sd, 2)
    rate = (mode + Tns.pow(Tns.pow(mode, 2) + 4 * var, 0.5)) / (2 * var)
    shape = 1 + mode * rate
    return shape, rate


def np_gamma_params(mode=10., sd=10.):
    '''
    Converst mode and sd to shape and rate of a gamma distribution.
    '''
    var = sd ** 2
    rate = (mode + (mode ** 2 + 4 * var) ** .5) / (2 * var)
    shape = 1 + mode * rate
    return shape, rate


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
        obs_ambiguity = Normal(
            'DNP Mean_Ambiguity', mu=0, sd=500.0, shape=2)

        # Define variable for sd of data distribution:
        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y.std(), sd=y.std()))
        for ctxt, label in zip([1, 0], ['context', 'nocontext']):
            obs_offset = Normal('DNP Mean_Offset_' + label, mu=y.mean(),
                                sd=y.std() * 1.0)
            obs_sd_offset = Gamma('Mean_Offset_SD' + label, *gamma_params(mode=y.std(), sd=y.std()))

            # Observer level:
            offset = Normal(
                'DNP Offset_' + label, mu=obs_offset, sd=obs_sd_offset,
                shape=(obs[ctxt],))

            # Compute predicted mode for each fixation:
            data = y[context == ctxt]
            obs_c = observer[context == ctxt]
            ambig_reg_c = ambiguity_regressor[context == ctxt]

            b0 = obs_ambiguity.mean()
            obs_ambiguity_transformed = Deterministic("DNS Population Ambiguity" +label , obs_ambiguity-b0 )
            offset_transformed = Deterministic('DNS Subject Offsets ' + label, offset+b0)
            obs_offset_transformed = Deterministic('DNS Population Offsets ' + label, obs_offset+b0)
            oat = obs_ambiguity - b0
            # Dummy coding
            mode = (
                offset_transformed[obs_c] +
                obs_ambiguity_transformed[ambig_reg_c == 1])
            # Convert to shape rate parameterization
            shape, rate = gamma_params(mode, data_sd)
            data_dist = Gamma('Data_' + label, shape, rate, observed=data)
    return pl


def appc_subject_model(
        y,
        observer,
        ambiguity_regressor,
        context,
        observed=True):
    '''
    Hierarchical Gamma model to predict APPC data with no subject specific distributions.
    '''
    num_obs_ctxt = len(np.unique(observer[context == 1]))
    num_obs_noctxt = len(np.unique(observer[context == 0]))
    obs = [num_obs_noctxt, num_obs_ctxt]
    with Model() as pl:
        # Population level:
        obs_ambiguity = TruncatedNormal(
            'DNP Mean_Ambiguity', mu=0, sd=500.0, shape=2, lower=-100, upper=100)

        # Define variable for sd of data distribution:
        data_sd = pm.Gamma('Data_SD', *gamma_params(mode=y.std(), sd=y.std()))
        for ctxt, label in zip([1, 0], ['context', 'nocontext']):
            obs_offset = TruncatedNormal('DNP Mean_Offset_' + label, mu=y.mean(),
                                sd=y.std() * 1.0, lower=0, upper=np.inf)

            # Compute predicted mode for each fixation:
            data = y[context == ctxt]
            ambig_reg_c = ambiguity_regressor[context == ctxt]

            b0 = obs_ambiguity.mean()
            obs_ambiguity_transformed = Deterministic("DNS Population Ambiguity" +label , obs_ambiguity-b0 )
            obs_offset_transformed = Deterministic('DNS Population Offsets ' + label, obs_offset+b0)

            # Dummy coding
            mode = (
                obs_offset_transformed +
                obs_ambiguity_transformed[ambig_reg_c == 1])
            # Convert to shape rate parameterization
            shape, rate = gamma_params(mode, data_sd)
            data_dist = Gamma('Data_' + label, shape, rate, observed=data)
    return pl



def save(traces, filename):
    '''
    Save a multitrace object to HDF5
    '''
    import h5py
    with h5py.File(filename) as f:
        for var in traces.varnames:
            data = traces.get_values(var)
            if isinstance(data, np.ndarray):
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


def sample_model_appc(model, steps, tune=None, njobs=4, observed=['Data']):
    if tune is None:
        tune = steps / 2
    with model:
        start = pm.find_MAP()
        non_blocked_step = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if ('Obs_SD' in k) or ('Mean_' in k) and not (k in set(observed)) and not k.startswith('DNS')],
            blocked=False)
        blocked = pm.Metropolis(
            vars=[v for k, v in model.named_vars.iteritems()
                  if not (('Obs_SD' in k) or ('Mean_' in k))
                  and not (k in set(observed)) and not k.startswith('DNS')],
            blocked=True)
        trace = pm.sample(
            steps, [non_blocked_step, blocked], start,
            tune=tune, njobs=njobs, progressbar=True)
    return trace


from scipy.io import loadmat

def get_appc_model():
    data = loadmat('bayes_datamat.mat')['datamat']
    context = data['context'][0, 0][0].astype(int)
    observer = data['subject'][0, 0][0].astype(int)
    ambiguity_regressor = data['ambiguity'][0, 0][0].astype(int)
    duration = data['fd'][0, 0][0]
    idnan = np.isnan(context) | np.isnan(observer) | np.isnan(ambiguity_regressor) | np.isnan(duration)

    m = appc_gamma_model(
        duration[~idnan],
        observer.astype(int)[~idnan],
        ambiguity_regressor[~idnan],
        context[~idnan])
    return m

def run_appc_model():
    '''
    Run appc model and load data for it...
    '''
    return sample_model_appc(get_appc_model(), 500000, observed=['Data_context', 'Data_nocontext', ])

def run_appc_rt_model():
    '''
    Run appc model and load data for it...
    '''
    from scipy.io import loadmat
    
    data = loadmat('bayes_datamat_rt.mat')['datamatRT']
    context = data['context'][0, 0][0].astype(int)
    observer = data['subject'][0, 0][0].astype(int)
    ambiguity_regressor = data['ambiguity'][0, 0][0].astype(int)
    rt = data['rt'][0, 0][0]
    idnan = np.isnan(context) | np.isnan(observer) | np.isnan(ambiguity_regressor) | np.isnan(rt)

    m = appc_gamma_model(
        rt[~idnan],
        observer.astype(int)[~idnan],
        ambiguity_regressor[~idnan],
        context[~idnan])
    return sample_model_appc(
        m, 500000, observed=['Data_context', 'Data_nocontext', ])


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    # t = run_test_case() #run_fixdur()
    t = run_appc_rt_model()
    save(t, filename)
    pm.traceplot(t, vars=[v for v in t.varnames if not v.startswith('DNP')])
    plt.show()
