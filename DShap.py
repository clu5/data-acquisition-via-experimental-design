
#______________________________________PEP8____________________________________
#_______________________________________________________________________
from tqdm.notebook import tqdm
# from tqdm import tqdm
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
# import tensorflow as tf
import sys
from shap_utils import *
from Shapley import ModelWrapper
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score

class DShap(object):
    
    def __init__(self, train_dataset, test_dataset,
                 # x, y, x_test, y_test, 
                 num_test, sources=None, 
                 # sample_weight=None, 
                 directory=None, 
                 use_features=False,
                 metric='accuracy', 
                 seed=None, overwrite=False,
                 **kwargs):
        """
        Args:
            x: Data covariates
            y: Data labels
            x_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            overwrite: Delete existing data and start computations from 
                scratch
            **kwargs: Arguments of the model
        """
        if seed is not None:
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        x_train, y_train = train_dataset.tensors
        x_test, y_test = test_dataset.tensors
            
        # self.problem = problem
        # self.model_family = model_family
        self.metric = metric
        self.directory = Path(directory)
        if self.directory is not None:
            if overwrite and self.directory.exists():
                exit(f'Directory already exists. Please delete {self.directory}')
            elif not self.directory.exists():
                (self.directory / 'weights').mkdir(parents=True)
                (self.directory / 'plots').mkdir(parents=True)
                
        self.model = ModelWrapper(use_features=use_features, address=directory, **kwargs)
    
        # self.model = return_model(self.model_family, **kwargs)
        
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(x_train))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
            
        data_dir = self.directory / 'data.pkl'
        if data_dir.exists():
            '''Load the different sets of data if already exists.'''
            data_dic = pkl.load(open(data_dir, 'rb'))
            self.x_train = data_dic['x_train'] 
            self.y_train = data_dic['y_train']
            self.x_heldout = data_dic['x_heldout']
            self.y_heldout = data_dic['y_heldout']
            self.x_test = data_dic['x_test']
            self.y_test = data_dic['y_test']
            self.sources = data_dic['sources']
            # if 'sample_weight' in data_dic.keys():
                # self.sample_weight = data_dic['sample_weight']
            # else:
                # self.sample_weight = None
        else:
            self.x_heldout = x_test[:-num_test]  # todo: replace with data sets
            self.y_heldout = y_test[:-num_test]
            self.x_test = x_test[-num_test:]
            self.y_test = y_test[-num_test:]
            self.x_train, self.y_train, self.sources = x_train, y_train, sources
            # self.sample_weight = sample_weight
            data_dic = {
                'x_train': self.x_train, 
                'y_train': self.y_train, 
                'x_test': self.x_test,
                'y_test': self.y_test, 
                'x_heldout': self.x_heldout,
                'y_heldout':self.y_heldout, 
                'sources': self.sources,
            }
            # if sample_weight is not None:
                # data_dic['sample_weight'] = sample_weight
                # warnings.warn("Sample weight not implemented for G-Shapley")
                
            pkl.dump(data_dic, open(data_dir, 'wb'))        
            
        self.random_score = self.init_score(self.metric)
            
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
            
        n_sources = len(self.x_train) if self.sources is None else len(self.sources)
        n_points = len(self.x_train)
        self.tmc_number, self.g_number = self._which_parallel(self.directory)
        self._create_results_placeholder(
            self.directory, self.tmc_number, self.g_number, n_points, n_sources, 
            # self.model_family,
        )
            
        
    def _create_results_placeholder(self, directory, tmc_number, g_number,
                                    n_points, n_sources, 
                                    # model_family
                                   ):
        tmc_dir = os.path.join( 
            directory, 
            'mem_tmc_{}.pkl'.format(tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            directory, 
            'mem_g_{}.pkl'.format(g_number.zfill(4))
        )
        self.mem_tmc = np.zeros((0, n_points))
        self.mem_g = np.zeros((0, n_points))
        self.idxs_tmc = np.zeros((0, n_sources), int)
        self.idxs_g = np.zeros((0, n_sources), int)
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, open(tmc_dir, 'wb'))
        # if model_family not in ['logistic', 'NN']:
            # return
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, open(g_dir, 'wb'))
        
    def _which_parallel(self, directory):
        '''Prevent conflict with parallel runs.'''
        previous_results = os.listdir(directory)
        tmc_nmbrs = [int(name.split('.')[-2].split('_')[-1]) for name in previous_results if 'mem_tmc' in name]
        g_nmbrs = [int(name.split('.')[-2].split('_')[-1]) for name in previous_results if 'mem_g' in name]        
        tmc_number = str(np.max(tmc_nmbrs) + 1) if len(tmc_nmbrs) else '0' 
        g_number = str(np.max(g_nmbrs) + 1) if len(g_nmbrs) else '0' 
        return tmc_number, g_number
    
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y_test)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y_train)
            # if self.sample_weight is None:
            self.model.fit(self.x_train, rnd_y)
            # else:
                # self.model.fit(self.x_train, rnd_y, 
                               # sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, x=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            x: Covariates, valuation is performed on a data 
                different from test set.
            y: Labels, if valuation is performed on a data 
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if x is None:
            x = self.x_test
        if y is None:
            y = self.y_test
        # if inspect.isfunction(metric):
            # return metric(model, x, y)
        if metric == 'accuracy':
            return model.accuracy_score(x, y)
        elif metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(x))
        elif metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, x, y)
        elif metric == 'xe':
            return my_xe_score(model, x, y)
        raise ValueError('Invalid metric!')
        
    def run(self, save_every, err, tolerance=0.01, g_run=True, loo_run=True):
        """Calculates data sources(points) values.
        
        Args:
            save_every: save marginal contrivbutions every n iterations.
            err: stopping criteria.
            tolerance: Truncation tolerance. If None, it's computed.
            g_run: If True, computes G-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                self.save_results(overwrite=True)
                
        print('LOO values calculated!')
        tmc_run = True 
        # g_run = g_run and self.model_family in ['logistic', 'NN']
        while tmc_run or g_run:
            if g_run:
                print(self.mem_g)
                if error(self.mem_g) < err:
                    g_run = False
                else:
                    self._g_shap(save_every, sources=self.sources)
                    self.vals_g = np.mean(self.mem_g, 0)
            if tmc_run:
                print(self.mem_tmc)
                if error(self.mem_tmc) < err:
                    tmc_run = False
                else:
                    self._tmc_shap(
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.mem_tmc, 0)
            if self.directory is not None:
                self.save_results()
            
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        loo_dir = os.path.join(self.directory, 'loo.pkl')
        if not os.path.exists(loo_dir) or overwrite:
            pkl.dump({'loo': self.vals_loo}, open(loo_dir, 'wb'))
        tmc_dir = os.path.join(
            self.directory, 
            'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4))
        )
        g_dir = os.path.join(
            self.directory, 
            'mem_g_{}.pkl'.format(self.g_number.zfill(4))
        )  
        pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, 
                 open(tmc_dir, 'wb'))
        pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, 
                 open(g_dir, 'wb'))  
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tolerance         
        marginals, idxs = [], []
        for iteration in tqdm(range(iterations), total=iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.one_iteration(
                tolerance=tolerance, 
                sources=sources
            )
            self.mem_tmc = np.concatenate([
                self.mem_tmc, 
                np.reshape(marginals, (1,-1))
            ])
            self.idxs_tmc = np.concatenate([
                self.idxs_tmc, 
                np.reshape(idxs, (1,-1))
            ])
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        # for _ in range(1):
        # if self.sample_weight is None:
        self.model.fit(self.x_train, self.y_train)
        # else:
            # self.model.fit(self.x_train, self.y_train,
                          # sample_weight=self.sample_weight)
        for _ in range(100):
            bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
            scores.append(self.value(
                self.model, 
                metric=self.metric,
                x=self.x_test[bag_idxs], 
                y=self.y_test[bag_idxs]
            ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)
        
    def one_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        marginal_contribs = np.zeros(len(self.x_train))
        x_batch = torch.tensor(np.zeros((0,) + tuple(self.x_train.shape[1:])), dtype=torch.float)
        y_batch = torch.tensor(np.zeros(0, int), dtype=torch.long)
        # sample_weight_batch = np.zeros(0)
        truncation_counter = 0
        new_score = self.random_score
        for n, idx in tqdm(enumerate(idxs), total=len(idxs)):
            old_score = new_score
            # x_batch = np.concatenate([x_batch, self.x_train[sources[idx]]])
            # y_batch = np.concatenate([y_batch, self.y_train[sources[idx]]])
            x_batch = torch.cat([x_batch, torch.tensor(self.x_train[sources[idx]], dtype=torch.float)])
            y_batch = torch.cat([y_batch, torch.tensor(self.y_train[sources[idx]], dtype=torch.long)])
            # if self.sample_weight is None:
                # sample_weight_batch = None
            # else:
                # sample_weight_batch = np.concatenate([
                    # sample_weight_batch, 
                    # self.sample_weight[sources[idx]]
                # ])
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     if (self.is_regression 
            #         or len(set(y_batch.numpy())) == len(set(self.y_test.numpy()))): ##FIXIT
            self.restart_model()
            # if sample_weight_batch is None:
            self.model.fit(x_batch, y_batch)
                    # else:
                        # self.model.fit(
                        #     x_batch, 
                        #     y_batch,
                        #     sample_weight = sample_weight_batch
                        # )
            new_score = self.value(self.model, metric=self.metric)       
            marginal_contribs[sources[idx]] = (new_score - old_score)
            marginal_contribs[sources[idx]] /= len(sources[idx])
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def restart_model(self):
        self.model.model = self.model.model.cpu()
        for layer in self.model.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        torch.cuda.empty_cache()
        # try:
        #     self.model = clone(self.model)
        # except:
        #     self.model.fit(np.zeros((0,) + self.x_train.shape[1:]), self.y_train)
        
    def _one_step_lr(self):
        """Computes the best learning rate for G-Shapley algorithm."""
        if self.directory is None:
            address = None
        else:
            address = os.path.join(self.directory, 'weights')
        best_acc = 0.0
        for i in np.arange(1, 5, 0.5):
            model = ShapNN(
                self.problem, batch_size=1, max_epochs=1, 
                learning_rate=10**(-i), weight_decay=0., 
                validation_fraction=0, optimizer='sgd', 
                warm_start=False, address=address, 
                hidden_units=self.hidden_units)
            accs = []
            for _ in range(10):
                # model.fit(np.zeros((0, self.x_train.shape[-1])), self.y_train)
                model.fit(self.x_train, self.y_train)
                # accs.append(model.score(self.x_test, self.y_test))
                accs.append(model.accuracy_score(self.x_test, self.y_test))
            if np.mean(accs) - np.std(accs) > best_acc:
                best_acc  = np.mean(accs) - np.std(accs)
                learning_rate = 10**(-i)
        return learning_rate
    
    def _g_shap(self, iterations, err=None, learning_rate=None, sources=None):
        """Method for running G-Shapley algorithm.
        
        Args:
            iterations: Number of iterations of the algorithm.
            err: Stopping error criteria
            learning_rate: Learning rate used for the algorithm. If None
                calculates the best learning rate.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        address = None
        if self.directory is not None:
            address = os.path.join(self.directory, 'weights')
        if learning_rate is None:
            try:
                learning_rate = self.g_shap_lr
            except AttributeError:
                self.g_shap_lr = self._one_step_lr()
                learning_rate = self.g_shap_lr
        model = ShapNN(self.problem, batch_size=1, max_epochs=1,
                     learning_rate=learning_rate, weight_decay=0.,
                     validation_fraction=0, optimizer='sgd',
                     address=address, hidden_units=self.hidden_units)
        for iteration in range(iterations):
            # model.fit(np.zeros((0, self.x_train.shape[-1])), self.y)
            if 10 * (iteration+1) / iterations % 1 == 0:
                print('{} out of {} g-shapley iterations'.format(
                    iteration + 1, iterations))
            marginal_contribs = np.zeros(len(sources.keys()))
            model.fit(self.x_train, self.y_train, self.x_test, self.y_test, 
                      sources=sources, metric=self.metric, 
                      max_epochs=1, batch_size=1)
            val_result = model.history['metrics']
            marginal_contribs[1:] += val_result[0][1:]
            marginal_contribs[1:] -= val_result[0][:-1]
            individual_contribs = np.zeros(len(self.x_train))
            for i, index in enumerate(model.history['idxs'][0]):
                individual_contribs[sources[index]] += marginal_contribs[i]
                individual_contribs[sources[index]] /= len(sources[index])
            self.mem_g = np.concatenate(
                [self.mem_g, np.reshape(individual_contribs, (1,-1))])
            self.idxs_g = np.concatenate(
                [self.idxs_g, np.reshape(model.history['idxs'][0], (1,-1))])
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        print('starting loo score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        # if self.sample_weight is None:
            # print(self.x_train.shape, self.y_train.shape)
        self.model.fit(self.x_train, self.y_train)
        # else:
            # self.model.fit(self.x_train, self.y_train,
                          # sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.x_train))
        for i in tqdm(sources.keys()):
            x_batch = np.delete(self.x_train, sources[i], axis=0)
            y_batch = np.delete(self.y_train, sources[i], axis=0)
            # if self.sample_weight is not None:
                # sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            # if self.sample_weight is None:
            self.model.fit(x_batch, y_batch)
            # else:
                # self.model.fit(x_batch, y_batch, sample_weight=sw_batch)
                
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo
    
    def _merge_parallel_results(self, key, max_samples=None):
        """Helper method for 'merge_results' method."""
        numbers = [name.split('.')[-2].split('_')[-1]
                   for name in os.listdir(self.directory) 
                   if 'mem_{}'.format(key) in name]
        mem  = np.zeros((0, self.x_train.shape[0]))
        n_sources = len(self.x_train) if self.sources is None else len(self.sources)
        idxs = np.zeros((0, n_sources), int)
        vals = np.zeros(len(self.x_train))
        counter = 0.
        for number in numbers:
            if max_samples is not None:
                if counter > max_samples:
                    break
            samples_dir = os.path.join(
                self.directory, 
                'mem_{}_{}.pkl'.format(key, number)
            )
            print(samples_dir)
            dic = pkl.load(open(samples_dir, 'rb'))
            if not len(dic['mem_{}'.format(key)]):
                continue
            mem = np.concatenate([mem, dic['mem_{}'.format(key)]])
            idxs = np.concatenate([idxs, dic['idxs_{}'.format(key)]])
            counter += len(dic['mem_{}'.format(key)])
            vals *= (counter - len(dic['mem_{}'.format(key)])) / counter
            vals += len(dic['mem_{}'.format(key)]) / counter * np.mean(mem, 0)
            os.remove(samples_dir)
        merged_dir = os.path.join(
            self.directory, 
            'mem_{}_0000.pkl'.format(key)
        )
        pkl.dump({'mem_{}'.format(key): mem, 'idxs_{}'.format(key): idxs}, 
                 open(merged_dir, 'wb'))
        return mem, idxs, vals
            
    def merge_results(self, max_samples=None):
        """Merge all the results from different runs.
        
        Returns:
            combined marginals, sampled indexes and values calculated 
            using the two algorithms. (If applicable)
        """
        tmc_results = self._merge_parallel_results('tmc', max_samples)
        self.marginals_tmc, self.indexes_tmc, self.values_tmc = tmc_results
        # if self.model_family not in ['logistic', 'NN']:
            # return
        g_results = self._merge_parallel_results('g', max_samples)
        self.marginals_g, self.indexes_g, self.values_g = g_results
    
    def performance_plots(self, vals, name=None, 
                          num_plot_markers=20, sources=None):
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        plt.rcParams['figure.figsize'] = 8,8
        plt.rcParams['font.size'] = 25
        plt.xlabel('Fraction of train data removed (%)')
        plt.ylabel('Prediction accuracy (%)', fontsize=20)
        if not isinstance(vals, list) and not isinstance(vals, tuple):
            vals = [vals]
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        vals_sources = [np.array([np.sum(val[sources[i]]) 
                                  for i in range(len(sources.keys()))])
                  for val in vals]
        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys()) - 1
        plot_points = np.arange(
            0, 
            max(len(sources.keys()) - 10, num_plot_markers),
            max(len(sources.keys())//num_plot_markers, 1)
        )
        perfs = [self._portion_performance(
            np.argsort(vals_source)[::-1], plot_points, sources=sources)
                 for vals_source in vals_sources]
        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.argsort(vals_sources[0])[::-1]),
            plot_points, sources=sources) for _ in range(10)], 0)
        plt.plot(plot_points/len(self.x_train) * 100, perfs[0] * 100, 
                 '-', lw=5, ms=10, color='b')
        if len(vals)==3:
            plt.plot(plot_points/len(self.x_train) * 100, perfs[1] * 100, 
                     '--', lw=5, ms=10, color='orange')
            legends = ['tmc-shapley ', 'g-shapley ', 'loo', 'random']
        elif len(vals)==2:
            legends = ['tmc-shapley ', 'loo', 'random']
        else:
            legends = ['tmc-shapley ', 'random']
        plt.plot(plot_points/len(self.x_train) * 100, perfs[-1] * 100, 
                 '-.', lw=5, ms=10, color='g')
        plt.plot(plot_points/len(self.x_train) * 100, rnd * 100, 
                 ':', lw=5, ms=10, color='r')    
        plt.legend(legends)
        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(
                self.directory, 'plots', '{}.png'.format(name)),
                        bbox_inches = 'tight')
            plt.close()
            
    def _portion_performance(self, idxs, plot_points, sources=None):
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""
        if sources is None:
            sources = {i:np.array([i]) for i in range(len(self.x_train))}
        elif not isinstance(sources, dict):
            sources = {i:np.where(sources==i)[0] for i in set(sources)}
        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            keep_idxs = np.concatenate([sources[idx] for idx 
                                        in idxs[plot_points[i-1]:]], -1)
            x_batch, y_batch = self.x_train[keep_idxs], self.y_train[keep_idxs]
            # if self.sample_weight is not None:
                # sample_weight_batch = self.sample_weight[keep_idxs]
            # with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                # if (self.is_regression 
                    # or len(set(y_batch.numpy())) == len(set(self.y_test.numpy()))):
            self.restart_model()
                    # if self.sample_weight is None:
            self.model.fit(x_batch, y_batch)
                    # else:
                        # self.model.fit(x_batch, y_batch,
                                      # sample_weight=sample_weight_batch)
            scores.append(self.value(
                self.model,
                metric=self.metric,
                x=self.x_heldout,
                y=self.y_heldout,
            ))
                # else:
                    # scores.append(init_score)
        return np.array(scores)[::-1]
