import torch
import numpy as np
from lib.reverse_auto_diff import Tensor

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = torch.zeros_like(param.data)

class SGD(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            param.data -= self.learning_rate * param.grad


class RMSProp(Optimizer):
    def __init__(self, params, learning_rate=0.01, decay=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.decay = decay
        self.cache = [torch.zeros_like(p.data) for p in params]

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode RMSProp.
        La méthode RMSProp (Root Mean Square Propagation) est une méthode d'optimisation adaptative
        qui divise le taux d'apprentissage par une moyenne exponentielle glissante des carrés des gradients passés.
        """
        ## FIXME
        for i in range(len(self.params)):
          self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * (self.params[i].grad ** 2)
          self.params[i].data -= (self.learning_rate * self.params[i].grad) / (torch.sqrt(self.cache[i]) + 1e-8)
          self.params[i].grad = torch.zeros_like(self.params[i].grad)
        # FIXED

class Adagrad(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.cache = [torch.zeros_like(p.data) for p in params]

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode Adagrad.
        La méthode Adagrad (Adaptive Gradient) est une méthode d'optimisation adaptative
        qui divise le taux d'apprentissage par la racine carrée de la somme cumulée des carrés des gradients passés.
        """
        ## FIXME
        for p, c in zip(self.params, self.cache):
            c += p.grad ** 2
            p.data -= (self.learning_rate * p.grad) / (torch.sqrt(c) + 1e-8)
            p.grad = torch.zeros_like(p.grad)
        # FIXED


class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in params]
        self.v = [torch.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode Adam.
        La méthode Adam (Adaptive Moment Estimation) est une méthode d'optimisation adaptative
        qui utilise une moyenne exponentielle glissante des gradients passés et de leurs carrés
        pour estimer la première et la deuxième moments des gradients.
        """
        ## FIXME
        self.t += 1
        for p, mo, ve in zip(self.params, self.m, self.v):
          mo = self.beta1 * mo + (1 - self.beta1) * p.grad
          ve = self.beta2 * ve + (1 - self.beta2) * (p.grad ** 2)
          m_hat = mo / (1 - self.beta1 ** self.t)
          v_hat = ve / (1 - self.beta2 ** self.t)
          p.data -= (self.learning_rate * m_hat) / (torch.sqrt(v_hat) + self.eps)
          p.grad = torch.zeros_like(p.grad)
        # FIXED

class AdamW(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p.data) for p in params]
        self.v = [torch.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        """
        Mise à jour des paramètres en utilisant la méthode AdamW.
        La méthode AdamW (Adaptive Moment Estimation with Weight Decay) est une variante de la méthode Adam
        qui ajoute une régularisation L2 (ou weight decay) aux mises à jour des paramètres.
        """
        # FIXME
        self.t += 1
        for p, mo, ve in zip(self.params, self.m, self.v):
          mo = self.beta1 * mo + (1 - self.beta1) * p.grad
          ve = self.beta2 * ve + (1 - self.beta2) * (p.grad ** 2)
          m_hat = mo / (1 - self.beta1 ** self.t)
          v_hat = ve / (1 - self.beta2 ** self.t)
          p.data -= self.learning_rate * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
          p.grad = torch.zeros_like(p.grad)
        # FIXED

class Momentum(Optimizer):
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in params]

    def step(self):
        # FIXME
        for p, v in zip(self.params, self.velocity):
          v = self.momentum * v - self.learning_rate * p.grad
          p.data += v
          p.grad = torch.zeros_like(p.grad)
        # FIXED



class Adadelta(Optimizer):
    def __init__(self, params, rho=0.95, eps=1e-8):
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.cache = [torch.zeros_like(p.data) for p in params]
        self.delta = [torch.zeros_like(p.data) for p in params]

    def step(self):
        # FIXME
        for p, c, d in zip(self.params, self.cache, self.delta):
          c = self.rho * c + (1 - self.rho) * p.grad ** 2
          d = self.rho * d + (1 - self.rho) * p.data ** 2
          p.data -= (p.grad * torch.sqrt(d + self.eps)) / torch.sqrt(c + self.eps)
          p.grad = torch.zeros_like(p.grad)
        # FIXED

class LRScheduler:
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.iteration = 0  # Ou self.epoch = 0 si basé sur les époques

    def update_lr(self, metrics=None):
        # FIXME: update learning rate based on metrics
        self.iteration += 1
        self.lr = self.initial_lr * (0.9 ** self.iteration)
        # FIXED

    def step(self, metrics=None):
        # FIXME: update learning rate based on metrics
        self.update_lr(metrics)
        self.optimizer.learning_rate = self.lr
        # FIXED

class LRSchedulerOnPlateau(LRScheduler):
    def __init__(self, optimizer, initial_lr, patience=10, factor=0.1, min_lr=1e-6, mode='min', threshold=1e-4):
        super().__init__(optimizer, initial_lr)
        self.patience = patience  # Nombre d'époques sans amélioration avant de réduire le taux
        self.factor = factor  # Facteur de réduction du taux d'apprentissage
        self.min_lr = min_lr  # Valeur minimale du taux d'apprentissage
        self.mode = mode  # 'min' : réduire le taux quand la métrique cesse de diminuer, 'max' : inverse
        self.threshold = threshold  # Seuil pour déterminer la réduction de la métrique
        self.best_metric = float('inf') if mode == 'min' else float('-inf')  # Meilleure métrique observée
        self.num_bad_epochs = 0  # Nombre d'époques sans amélioration de la métrique

    def update_lr(self, metric):
        # FIXME: update learning rate based on metric
        if self.mode == 'min':
            if metric < self.best_metric - self.threshold:
                self.best_metric = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        else:
            if metric > self.best_metric + self.threshold:
                self.best_metric = metric
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            old_lr = self.lr
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0
            self.optimizer.learning_rate = self.lr
        # FIXED

