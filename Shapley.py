import os
from pathlib import Path
import numpy as np
import torch
import torchvision
# import tensorflow as tf
# tf.get_logger().setLevel('INFO')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

class LR(torch.nn.Module):
    def __init__(self, input_dim=1000, output_dim=10):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 6, 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 10),
        ])
        
    def forward(self, x):
        # *_, x = (x := l(x) for x in self.layers)
        for l in self.layers:
            x = l(x)
        return x
    

class ModelWrapper(object):
    
    def __init__(self, use_features=False, learning_rate=0.001, 
                 weight_decay=0.0001, optimizer=torch.optim.SGD, batch_size=128,
                 max_epochs=100, early_stopping=0, validation_fraction=0.1,
                 address=None, test_batch_size=16, random_seed=666,
                ):
        
        self.model = LR() if use_features else CNN()
        print(self.model)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.address = Path(address)
        self.random_seed = random_seed
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def prediction_cost(self, x, y, batch_size=None):
        if batch_size is None:
            batch_size = self.test_batch_size
        # assert len(set(y.detach().cpu().numpy())) == self.num_classes, 'Number of classes does not match!'
        self.model.eval()
        self.model = self.model.cuda()
        idxs = np.arange(len(x))            
        batches = [idxs[k * batch_size: (k+1) * batch_size] for k in range(int(np.ceil(len(idxs)/batch_size)))]
        losses = torch.cat([self.loss_fn(self.model(x[batch].cuda()).detach().cpu(), y[batch]) for batch in batches], 0)
        # self.model = self.model.cpu()
        torch.cuda.empty_cache()
        return np.mean(losses.numpy()).item()
        
    def predict_proba(self, x, batch_size=None):
        if batch_size is None:
            batch_size = self.test_batch_size
        self.model.eval()
        self.model = self.model.cuda()
        idxs = np.arange(len(x))     
        batches = [idxs[k * batch_size: (k+1) * batch_size] for k in range(int(np.ceil(len(idxs)/batch_size)))]
        probs = torch.cat([torch.softmax(self.model(x[batch].cuda()).detach().cpu(), 0) for batch in batches], 0)
        # self.model = self.model.cpu()
        torch.cuda.empty_cache()
        return probs.numpy()
        
    def predict_log_proba(self, x, batch_size=None):
        probs = self.predict_proba(x)
        return np.log(np.clip(probs, 1e-12, None))   
        
    def predict(self, x, batch_size=None):
        return self.predict_proba(x).argmax(-1)
        
    def accuracy_score(self, x, y):
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        # assert len(set(y)) == self.num_classes, f'Number of classes does not match! {len(set(y))=} != {self.num_classes=}'
        self.model.eval()
        predictions = self.predict(x)
        # print(f'{predictions.shape=}', f'{type(predictions)}')
        # print(f'{y.shape=}')
        correct = (predictions == y).sum()
        total = predictions.shape[0]
        return (correct / total).item()
        
        # n = x.shape[0]
        # idxs = np.arange(n)
        # batches = [idxs[k * batch_size: (k+1) * batch_size] for k in range(int(np.ceil(n/batch_size)))]
        # print(f'{batch_size=}')
        # print(f'{idxs=}')
        # print(f'{batches[0].shape=}')
        # print(f'{self.model(x[batches[0]].cuda()).detach().cpu().argmax(1).shape=}')
        # print(f'{x.shape=}')
        # print(f'{y.shape=}')
        # print(f'{y[batches[0]].shape=}')
        # scores = torch.cat([(self.model(x[batch].cuda()).detach().cpu().argmax(1) == y[batch]).sum() / n for batch in batches], 0)
        # return np.mean(scores.numpy())
            
    def fit(self, x, y, x_val=None, y_val=None, sources=None, max_epochs=None,
            batch_size=None, save=False, load=False, metric='accuracy',
            # sample_weight=None, 
           ):
        assert len(x) and len(x) == len(y), f'{len(x)=}'
        
        self.num_classes = len(set(y.numpy() if isinstance(y, torch.Tensor) else y))
        self.metric = metric
        if max_epochs is None:
            max_epochs = self.max_epochs
        if batch_size is None:
            batch_size = self.batch_size

        torch.random.manual_seed(self.random_seed)
        
        if x_val is None and self.validation_fraction * len(x) > 2:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.validation_fraction)
        else:
            x_train, y_train = x, y

        assert len(x_train)==len(y_train), 'Input and labels not the same size'
        n = len(x_train)
        
        self.history = {'metrics':[], 'idxs':[]}
        stop_counter = 0
        best_performance = None
        optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        # print(next(self.model.parameters()).device)
        self.model = self.model.cuda()
        # print(next(self.model.parameters()).device)
        
        for epoch in range(max_epochs):
            vals = []
            if sources is None:
                # if sample_weight is None:
                idxs = np.random.permutation(n)
                # else:
                    # idxs = np.random.choice(n, n, p=sample_weight/np.sum(sample_weight))    
                    
                batches = [idxs[k*batch_size:(k+1) * batch_size] for k in range(int(np.ceil(n/batch_size)))]
                idxs = batches
            else:
                idxs = np.random.permutation(len(sources.keys()))
                batches = [sources[i] for i in idxs]
                
            for batch_counter, batch in enumerate(batches):
                # print(next(self.model.parameters()).device)
                pred = self.model(x_train[batch].cuda())
                loss = self.loss_fn(pred, y_train[batch].cuda())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                del pred
                torch.cuda.empty_cache()
                
                if x_val is not None:
                    if self.metric=='accuracy':
                        vals.append(self.accuracy_score(x_val, y_val))
                    elif self.metric=='f1':
                        vals.append(f1_score(y_val, self.predict(x_val)))
                    elif self.metric=='auc':
                        vals.append(roc_auc_score(y_val, self.predict_proba(x_val)[:,1]))
                    elif self.metric=='xe':
                        vals.append(-self.prediction_cost(x_val, y_val))
            vals_metrics, idxs = np.array(vals), np.array(idxs)

            self.history['idxs'].append(idxs)
            self.history['metrics'].append(vals_metrics)
            if self.early_stopping and x_val is not None:
                current_performance = np.mean(val_acc)
                if best_performance is None:
                    best_performance = current_performance
                if current_performance > best_performance:
                    best_performance = current_performance
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter > self.early_stopping:
                        break

        self.model = self.model.cpu()

        if save and self.address is not None:
            torch.save(self.model, self.address / 'model.pt')
            # self.saver.save(self.sess, self.address)

