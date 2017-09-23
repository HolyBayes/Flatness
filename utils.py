from __future__ import print_function, division

import cPickle
import time
from collections import OrderedDict, defaultdict

import numpy
import pandas
from hep_ml.preprocessing import IronTransformer
from matplotlib import cm
from matplotlib import pyplot as plt
from rep.metaml.factory import train_estimator
from rep.metaml.utils import map_on_cluster
from rep.plotting import ErrorPlot
from rep.utils import get_efficiencies, weighted_quantile
from rep.estimators import SklearnClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import clone
from itertools import izip
import random

try:
    from sklearn.cross_validation import KFold
except:
    from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score

pdg_names_correspondence = {0: "Ghost", 11: "Electron", 13: "Muon", 211: "Pion", 321: "Kaon", 2212: "Proton"}
labels_names_correspondence = {0: "Ghost", 1: "Electron", 2: "Muon", 3: "Pion", 4: "Kaon", 5: "Proton"}

labels_names_correspondence = OrderedDict(sorted(labels_names_correspondence.items()))
pdg_names_correspondence = OrderedDict(sorted(pdg_names_correspondence.items()))

names_pdg_correspondence = OrderedDict(map(lambda (x, y): (y, x), pdg_names_correspondence.items()))
names_labels_correspondence = OrderedDict(map(lambda (x, y): (y, x), labels_names_correspondence.items()))


def shrink_floats(data):
    """
    Supplementary function that processes pandas.DataFrame
    and replaces float64 to float32 to spent less memory
    """
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[column] = data[column].astype('float32')


def add_constructed_features(data):
    """
    method that takes pandas.DataFrame and adds new features (mostly, linear combinations).
    :param pandas.DataFrame data: input of PID algorithm, information from sub-detectors.
    """
    features_DLL = ['CombDLLmu', 'CombDLLpi', 'CombDLLp', 'CombDLLe', 'CombDLLk']
    features_RICH_DLL = ['RichDLLpi', 'RichDLLe', 'RichDLLp', 'RichDLLmu', 'RichDLLk']
    features_acc = ['InAccSpd', 'InAccPrs', 'InAccBrem', 'InAccEcal', 'InAccHcal', 'InAccMuon']
    data_comb = convert_DLL_to_LL(data, features_DLL)
    data_rich = convert_DLL_to_LL(data, features_RICH_DLL)
    data_acceptance = compute_cum_sum(data, features_acc, prefix_name='acc_cum_sum_')[
        ['acc_cum_sum_3', 'acc_cum_sum_5']]
    data = pandas.concat([data, data_rich, data_comb, data_acceptance], axis=1)
    data['RichAboveSumPiKaElMuTHres'] = data.RichAbovePiThres + data.RichAboveKaThres + \
                                        data.RichAboveElThres + data.RichAboveMuThres
    data['RichAboveSumKaPrTHres'] = data.RichAboveKaThres + data.RichAbovePrThres
    data['RichUsedGas'] = data.RichUsedR1Gas + data.RichUsedR2Gas
    data['SpdCaloNeutralAcc'] = data.CaloNeutralSpd + data.InAccSpd  # for ghost
    data['SpdCaloChargedAcc'] = data.CaloChargedSpd + data.InAccSpd  # for muon
    data['SpdCaloChargedNeutral'] = data.CaloChargedSpd + data.CaloNeutralSpd  # for electron
    data['CaloSumSpdPrsE'] = data.CaloSpdE + data.CaloPrsE
    data['CaloSumPIDmu'] = data.EcalPIDmu + data.HcalPIDmu
    return data


def preprocess(X, fit=False, iron_scaler_path="iron.pkl"):
    """
    Preprocess the dataset by applying 'iron' - transform each feature to uniform distribution
    :param X: pandas.DataFrame to be processed
    :param fit: if True, preprocessing model will be first fit to the data
    :param iron_scaler_path: file used to store the model
    :return:
    """
    shrink_floats(X)
    X = add_constructed_features(X)
    if fit:
        iron_scaler = IronTransformer(symmetrize=True)
        iron_scaler.fit(X)
        with open(iron_scaler_path, 'w') as f:
            cPickle.dump(iron_scaler, f)
    else:
        with open(iron_scaler_path, 'r') as f:
            iron_scaler = cPickle.load(f)
    return iron_scaler.transform(X)


def compute_labels_and_weights(pdg_column):
    """
    Compute labels column (from zero to five) and weight (sum of weights for each class are the same - balanced data).
    
    :param array pdg_column: pdg value for each sample
    :return: labels, weights
    """
    labels = numpy.abs(pdg_column).astype(int)
    mask = numpy.zeros(len(labels), dtype=bool)
    for key, val in names_pdg_correspondence.items():
        if key == 'Ghost':
            continue
        mask |= labels == val
    labels[~mask] = 0  # all other particles are not tracks, so they are GHOST also

    for key, value in names_labels_correspondence.items():
        labels[labels == names_pdg_correspondence[key]] = value
    weights = numpy.ones(len(labels))
    for label in names_labels_correspondence.values():
        weights[labels == label] = 1. / sum(labels == label)
    weights /= numpy.mean(weights) + 1e-10
    return labels, weights


def compute_charges(pdg_column):
    """
    Compute charge for each track to check charges asymmetry for the algorithm.
    Charge can be -1, +1 and 0 (zero corresponds to GHOST tracks)
    
    :param array pdg_column: pdg value for each sample, it has the sign
    :return: charges
    """
    charges = numpy.zeros(len(pdg_column))
    charges[pdg_column == 11] = -1
    charges[pdg_column == 13] = -1
    charges[(pdg_column == 321) | (pdg_column == 211) | (pdg_column == 2212)] = 1
    charges[pdg_column == -11] = 1
    charges[pdg_column == -13] = 1
    charges[(pdg_column == -321) | (pdg_column == -211) | (pdg_column == -2212)] = -1
    return charges


def compute_cum_sum(data, features, prefix_name=""):
    """
    Compute cumulative sum for features from starting with the first feature.
    
    :param pandas.DataFrame data: data 
    :param list features: features
    :param str prefix_name: prefix for produced features names
    :return: pandas.DataFrame new features
    """
    cum_sum = numpy.zeros(len(data))
    cum_features = {}
    for n, feature in enumerate(features):
        cum_sum += data[feature].values
        cum_features[prefix_name + str(n)] = cum_sum.copy()
    return pandas.DataFrame(cum_features, index=None)


def convert_DLL_to_LL(data, features):
    """
    Compute Likelihood for each particle from the DLL=Likelihood_particle - Likelihood_pion.
    We assume here that probabilities for each track sum up to 1.

    :param pandas.DataFrame data: data with DLL features
    :param list features: DLL features
    :return: pandas.DataFrame with features names + '_LL' 
    """
    temp_data = data[features].values
    # this step needed to have stable computations
    temp_data -= temp_data.max(axis=1, keepdims=True)
    temp_data = numpy.exp(temp_data)
    temp_data /= numpy.sum(temp_data, axis=1, keepdims=True)
    return pandas.DataFrame(numpy.log(numpy.clip(temp_data, 1e-6, 10)), columns=map(lambda x: x + '_LL', features))


def plot_hist_features(data, labels, features, bins=30, ignored_sideband=0.01):
    """
    Plot histogram of features, ignores '-999' and ignores sidebands.
    
    :param pandas.DataFrame data: data with features
    :param array labels: labels (from 0 to 5)
    :param list features: plotted features
    """
    labels = numpy.array(labels)
    for n, feature in enumerate(features):
        plt.subplot(int(numpy.ceil(len(features) / 6)), min(6, len(features)), n + 1)
        temp_values = data[feature].values
        temp_labels = numpy.array(labels)[temp_values != -999]
        temp_values = temp_values[temp_values != -999]
        v_min, v_max = numpy.percentile(temp_values, [ignored_sideband * 100, (1. - ignored_sideband) * 100])
        for key, val in names_labels_correspondence.items():
            plt.hist(temp_values[temp_labels == val], label=key, alpha=0.2, normed=True, bins=bins,
                     range=(v_min, v_max))
        plt.legend(loc='best')
        plt.title(feature)


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array
    described in http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html#Rolling-window,--strided-tricks

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    # here we compute the same expression (using algebraic expressions for them).
    n_subindices = float(len(subindices))
    subindices = numpy.array([0] + sorted(subindices) + [total_events], dtype='int')
    # via sum of the first squares
    summand1 = total_events * (total_events + 1) * (total_events + 0.5) / 3. / (total_events ** 3)
    left_positions = subindices[:-1]
    right_positions = subindices[1:]

    values = numpy.arange(len(subindices) - 1)

    summand2 = values * (right_positions * (right_positions + 1) - left_positions * (left_positions + 1)) / 2
    summand2 = summand2.sum() * 1. / (n_subindices * total_events * total_events)

    summand3 = (right_positions - left_positions) * values ** 2
    summand3 = summand3.sum() * 1. / (n_subindices * n_subindices * total_events)

    return summand1 + summand3 - 2 * summand2


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions))

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def plot_roc_one_vs_rest(labels, predictions, weights=None, physics_notion=False,
                         predictions_comparison=None, separate_particles=False,
                         algorithms_name=('MVA', 'baseline'), precision=5):
    """
    Plot roc curves one versus rest.
    
    :param array labels: labels form 0 to 5
    :param array predictions: array of predictions with shape [n_samples, n_particle_types] 
    :param array weights: sample weights
    """
    if weights is None:
        weights = numpy.ones(len(labels))
    if separate_particles:
        plt.figure(figsize=(22, 22))
    else:
        plt.figure(figsize=(10, 8))
    for label, name in labels_names_correspondence.items():
        if separate_particles:
            plt.subplot(3, 2, label + 1)
        for preds, prefix in zip([predictions, predictions_comparison], algorithms_name):
            if preds is None:
                continue
            fpr, tpr, _ = roc_curve(labels == label, preds[:, label], sample_weight=weights)
            auc = roc_auc_score(labels == label, preds[:, label], sample_weight=weights)
            if physics_notion:
                plt.plot(tpr * 100, fpr * 100, label='{}, {}, AUC=%1.{}f'.format(prefix, name, precision) % auc,
                         linewidth=2)
                plt.yscale('log', nonposy='clip')
            else:
                plt.plot(tpr, 1 - fpr, label='{}, AUC=%1.{}f'.format(name, precision) % auc, linewidth=2)
        if physics_notion:
            plt.xlabel('Efficiency', fontsize=22)
            plt.ylabel('Overall MisID Efficiency', fontsize=22)
        else:
            plt.xlabel('Signal efficiency', fontsize=22)
            plt.ylabel('Background rejection', fontsize=22)
        plt.legend(loc='best', fontsize=18)


def plot_roc_one_vs_one(labels, predictions, weights=None, precision=5):
    """
    Plot roc curves one versus one.
    
    :param array labels: labels form 0 to 5
    :param array predictions: array of predictions with shape [n_samples, n_particle_types]
    :param array weights: sample weights
    :param int precision: number of digits to print
    """
    if weights is None:
        weights = numpy.ones(len(labels))
    plt.figure(figsize=(22, 24))
    for label, name in labels_names_correspondence.items():
        plt.subplot(3, 2, label + 1)
        for label_vs, name_vs in labels_names_correspondence.items():
            if label == label_vs:
                continue
            mask = (labels == label) | (labels == label_vs)
            fpr, tpr, _ = roc_curve(labels[mask] == label, predictions[mask, label], sample_weight=weights)
            auc = roc_auc_score(labels[mask] == label, predictions[mask, label], sample_weight=weights)
            plt.plot(tpr, 1 - fpr, label='{} vs {}, AUC=%1.{}f'.format(name, name_vs, precision) % auc, linewidth=2)
        plt.xlabel('Signal efficiency', fontsize=22)
        plt.ylabel('Background rejection', fontsize=22)
        plt.legend(loc='best', fontsize=18)


def compute_pairwise_roc_auc_matrix(labels, predictions, weights=None):
    """
    Calculate class vs class roc aucs matrix.
    
    :param array labels: labels form 0 to 5
    :param array predictions: array of predictions with shape [n_samples, n_particle_types] 
    :param array weights: sample weights
    """
    # Calculate roc_auc_matrices
    roc_auc_matrices = numpy.zeros(shape=[len(labels_names_correspondence)] * 2)
    if weights is None:
        weights = numpy.ones(len(labels))
    for label, name in labels_names_correspondence.items():
        for label_vs, name_vs in labels_names_correspondence.items():
            if label == label_vs:
                continue
            mask = (labels == label) | (labels == label_vs)
            roc_auc_matrices[label, label_vs] = roc_auc_score(labels[mask] == label, predictions[mask, label],
                                                              sample_weight=weights[mask])

    matrix = pandas.DataFrame(roc_auc_matrices,
                              columns=names_labels_correspondence.keys(),
                              index=names_labels_correspondence.keys())
    return matrix


def plot_pairwise_roc_auc_matrix(matrix, vmin=0.7, vmax=1., title='Particle vs particle ROC AUCs', fmt='%.4f'):
    """
    Plot pairwise ROC AUC matrix.
    
    :param pandas.DataFrame matrix: pairwise matrix
    :param float vmin: the minimum value to display color
    :param float vmax: the maximum value to display color
    :param str title: title of the figure
    :param str fmt: the precision to display values
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    heatmap = ax.pcolor(matrix.values, vmin=vmin, vmax=vmax, cmap=cm.coolwarm)

    plt.title(title + '\n\n', size=15)

    ax.set_yticks(numpy.arange(matrix.shape[0]) + 0.5, minor=False)
    ax.set_xticks(numpy.arange(matrix.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(matrix.columns, minor=False)
    ax.set_yticklabels(matrix.index, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.yticks(rotation=90, size=12)
    plt.xticks(size=12)

    def show_values(pc):
        pc.update_scalarmappable()
        ax = pc.get_axes()
        for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if numpy.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            if x == y:
                continue
            ax.text(x, y, fmt % value, ha="center", va="center", color=color)

    show_values(heatmap)
    plt.colorbar(heatmap)


def plot_flatness_by_particle(labels, predictions, spectator, spectator_name, predictions_comparison=None,
                              names_algorithms=['MVA', 'Baseline'], for_particle=None,
                              weights=None, bins_number=30, ignored_sideband=0.1,
                              thresholds=None, n_col=1):
    """
    Build a flatness-plot, which demonstrates the dependency between efficiency and some observable.

    :param labels: [n_samples], contains targets
    :param predictions: [n_samples, n_particle_types] with predictions of an algorithm
    :param spectator: [n_samples], values of spectator variable
    :param spectator_name: str, name shown on the plot
    :param predictions_comparison: [n_samples, n_particle types], optionally for comparison this may be provided
    :param names_algorithms: names for compared algorithms
    :param weights: [n_samples], optional
    :param bins_number: int,
    :param ignored_sideband: fraction of ignored sidebands
    :param thresholds: efficiencies, for which flatness is drawn
    :param n_col: number of columns in legend.
    """
    plt.figure(figsize=(22, 24))
    if predictions_comparison is not None:
        colors = ['blue', 'green']
        markers = ['o', 's', 'v', 'o', 's', 'v']
    else:
        colors = [None, None]
        markers = ['o'] * len(thresholds)
    
    for n, (particle_name, label) in enumerate(names_labels_correspondence.items()):
        plt.subplot(3, 2, n + 1)
        title = '{} algorithm'.format(particle_name)
        xlim_all = (1e10, -1e10)
        ylim_all = (20, -1e8) 
        legends = []
        for preds, algo_name, color in zip([predictions, predictions_comparison], names_algorithms, colors):
            if preds is None:
                continue
            particle_mask = labels == label
            particle_probs = preds[particle_mask, label]
            particle_weights = None if weights is None else weights[particle_mask]

            thresholds_values = [
                weighted_quantile(particle_probs, quantiles=1 - eff / 100., sample_weight=particle_weights)
                for eff in thresholds]
            
            if for_particle is not None:
                particle_mask = labels == names_labels_correspondence[for_particle]
                particle_probs = preds[particle_mask, label]
                particle_weights = None if weights is None else weights[particle_mask]
                title = '{} algorithm for {}'.format(particle_name, for_particle)
                
            eff = get_efficiencies(particle_probs, spectator[particle_mask],
                                   sample_weight=particle_weights,
                                   bins_number=bins_number,
                                   errors=True,
                                   ignored_sideband=ignored_sideband,
                                   thresholds=thresholds_values)
            for thr in thresholds_values:
                eff[thr] = (eff[thr][0], 100 * numpy.array(eff[thr][1]), 100 * numpy.array(eff[thr][2]), eff[thr][3])

            xlim, ylim = compute_limits_and_plot_errorbar(eff, markers, color=color)
            plt.xlabel('{} {}\n\n'.format(particle_name, spectator_name), fontsize=22)
            plt.ylabel('Efficiency', fontsize=22)
            plt.title('\n\n'.format(title), fontsize=22)
            plt.xticks(fontsize=12), plt.yticks(fontsize=12)
            legends.append(['{} Eff {}%'.format(algo_name, thr) for thr in thresholds])
            plt.grid(True)
            
            xlim_all = (min(xlim_all[0], xlim[0]), max(xlim_all[1], xlim[1]))
            ylim_all = (min(ylim_all[0], ylim[0]), max(ylim_all[1], ylim[1]))
        plt.legend(numpy.concatenate(legends), loc='best', fontsize=16, framealpha=0.5, ncol=n_col)
        plt.xlim(xlim_all[0], xlim_all[1])
        plt.ylim(ylim_all[0], ylim_all[1])

        
def compute_limits_and_plot_errorbar(errors, markers, color=None):
    xlim = (1e10, -1e10)
    ylim = (1e10, -1e10) 
    for (name, val), marker in zip(errors.items(), markers):
        x, y, y_err, x_err = val
        err_bar = plt.errorbar(x, y, yerr=y_err, xerr=x_err, label=name, fmt='o', marker=marker,
                               ms=7, color=color)
        err_bar[0].set_label('_nolegend_')
        delta = (max(x + x_err[-1]) - min(x - x_err[0])) * 0.02
        xlim = (min(xlim[0], min(x - x_err[0]) - delta), max(xlim[1], max(x + x_err[-1]) + delta))
        
        delta = (max(y) - min(y)) * 0.05
        ylim = (min(ylim[0], min(y) - delta), max(ylim[1], max(y) + delta))
        
    return xlim, ylim 

                    
def compute_cvm_by_particle(labels, predictions, spectators, folds=5):
    """
    Compute CVMs for each particle for a dict of spectator variables.
    Needed to track correlations between predictions and spectators.

    :param labels: array of shape [n_samples] with correct classes
    :param predictions: array of shape [n_samples, n_particle_types] with predict probabilities
    :param spectators: dict(name : values) where values of shape [n_samples]
        corresponding with values of spectator variables
    :param folds: how many folds to use to compute the error

    :return: pandas.DataFrame with computed values
    """
    cvm_values = defaultdict(list)
    for spectator_name, spectator in spectators.items():
        for n, (particle_name, particle_label) in enumerate(names_labels_correspondence.items()):
            mask_particle = labels == particle_label
            probs = predictions[mask_particle, particle_label]
            cvm_folds = []
            for _, mask in KFold(len(probs), n_folds=folds, shuffle=True, random_state=456):
                cvm_folds.append(compute_cvm(probs[mask], spectator[mask_particle][mask]))
            cvm_values[spectator_name].append(numpy.mean(cvm_folds))
            cvm_values[spectator_name + ' error'].append(numpy.std(cvm_folds))
    return pandas.DataFrame(cvm_values, index=names_labels_correspondence.keys())


def compute_eta(track_p, track_pt):
    """
    Calculate pseudo rapidity values using p, pt.
    Note, that in LHCb this transformation is correct, since eta can't be of different signs.
    
    :param track_p: array, shape = [n_samples], TrackP values.
    :param track_pt: array, shape = [n_samples], TrackPt values.
    :return: array, shape = [n_samples], Pseudo Rapidity values.
    """

    sinz = track_pt * 1. / track_p
    z = numpy.arcsin(sinz)
    eta = - numpy.log(numpy.tan(0.5 * z))

    return eta


def roc_auc_score_one_vs_all(labels, predictions, sample_weight=None, folds=5, precision=4):
    """
    Compute ROC AUC values for (one vs rest).

    :param array labels: labels (from 0 to 5)
    :param dict predictions: predictions for each track of shape [n_samples, n_particle_types]
    :param array sample_weight: weights
    :param folds: how many folds to use to compute the error
    :param precision: how many digits on the plot
    :return: pandas.DataFrame with ROC AUC values for each class
    """
    rocs = OrderedDict()
    if sample_weight is None:
        sample_weight = numpy.ones(len(labels))
    for key, label in names_labels_correspondence.items():
        rocs_mask = []
        for _, mask in KFold(len(labels), n_folds=folds, shuffle=True, random_state=456):
            rocs_mask.append(roc_auc_score(labels[mask] == label, predictions[mask, label],
                                           sample_weight=sample_weight[mask]))
        rocs[key] = ['%1.{precision}f $\pm$ %1.{precision}f'.format(precision=precision) % (
            numpy.mean(rocs_mask), numpy.std(rocs_mask))]
    result = pandas.DataFrame(rocs)
    return result


class PIDEstimator(BaseEstimator, RegressorMixin):
    """
    A helper to train a multiclassifier or set of one vs all classifiers.
    In the second case training of several models is done in parallel.
    """

    def __init__(self, base_estimator, multi_mode=True):
        self.base_estimator = base_estimator
        self.multi_mode = multi_mode
        if not self.multi_mode:
            if not isinstance(base_estimator, dict):
                self.models = OrderedDict()
            else:
                self.models = self.base_estimator

    def fit(self, X, y, parallel_profile=None, **params):
        if self.multi_mode:
            self.base_estimator.fit(X, y, **params)
        else:
            start_time = time.time()
            labels = []
            if len(self.models) == 0:
                keys = numpy.unique(y)
                for key in keys:
                    labels.append((y == key) * 1)
                    self.models[key] = clone(self.base_estimator)
            else:
                for key in self.models.keys():
                    labels.append((y == key) * 1)
            sample_weight = numpy.ones(len(X)) if 'sample_weight' not in params else params['sample_weight']
            result = map_on_cluster(parallel_profile, train_estimator, list(self.models.keys()),
                                    list(self.models.values()),
                                    [X] * len(self.models), labels, [sample_weight] * len(self.models))
            for status, data in result:
                if status == 'success':
                    name, estimator, spent_time = data
                    self.models[name] = estimator
                    print('model {:12} was trained in {:.2f} seconds'.format(name, spent_time))
                else:
                    print('Problem while training on the node, report:\n', data)

            print("Totally spent {:.2f} seconds on training".format(time.time() - start_time))
        return self

    def predict(self, X):
        """
        Predict probabilities for all particles.
        """
        if self.multi_mode:
            return self.base_estimator.predict_proba(X)
        else:
            result = numpy.zeros(shape=(len(X), len(self.models)))
            for id_model, model in enumerate(self.models.values()):
                result[:, id_model] = model.predict_proba(X)[:, 1]
            return result


def pickle_dt_pid_estimator(dt_pid_estimator, filename):
    """
    Pickles PIDEstimator based on decision train.
    Preliminary loss internal parameters are deleted, as those take too much disk space
    :param dt_pid_estimator: PIDEstimator, with DecisionTrain inside
    :param filename: name of file to store model at
    """
    assert isinstance(dt_pid_estimator, PIDEstimator)
    # deleting loss parameters, as those take too much space
    for name, model in dt_pid_estimator.models.iteritems():
        assert not isinstance(model, SklearnClassifier), 'no wrapper please'
        model.loss = clone(model.loss)

    with open(filename, 'w') as f:
        cPickle.dump(dt_pid_estimator, f, protocol=2)

def iterate_minibatches(*data, **options):
    batch_size = options.pop('batch_size', 1000)
    shuffle = options.pop('shuffle', True)
  
    shape = len(data)
    data = zip(*data)
    if shuffle: random.shuffle(data)
    data_shuffled = [numpy.array(map(lambda x: x[i], data)) for i in range(shape)]
    for i in range(0, data_shuffled[0].shape[0], batch_size):
        yield tuple([data_shuffled[j][i:i+batch_size] for j in range(shape)])