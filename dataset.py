import numpy.random as npr
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from torch.utils.data.dataset import Dataset
import torch


class AdSBHDataset(Dataset):
    def __init__(self, N=1000, threshold=0.01):
        self.size = 2*N
        self.threshold = threshold
        self.L, self.V, self.labels = self.gen_data()

    def __getitem__(self, index):
        return self.L[index], self.V[index], self.labels[index]

    def __len__(self):
        return len(self.L)

    def gen_data(self):
        N = self.size
        # Generate the curve
        L_list = []
        V_list = []
        for zs in np.linspace(0.1, 0.999, num=100):
            L = self.integrate_L(zs)
            V = self.integrate_V(zs)
            if V < 0:
                L_list.append(L)
                V_list.append(V)
            else:
                break
        curve = interp1d(L_list, V_list)
        # Randomly draw datapoints
        N_pos = 0
        N_neg = 0
        pos_data = np.empty((N, 2))
        neg_data = np.empty((N, 2))
        while N_neg < N:
            L_val = npr.uniform(min(L_list), max(L_list))
            V_val = npr.uniform(min(V_list), max(V_list))
            if abs(V_val-curve(L_val)) > self.threshold:
                neg_data[N_neg] = np.array([L_val, V_val])
                N_neg += 1
        while N_pos < N:
            L_val = npr.uniform(min(L_list), max(L_list))
            pos_data[N_pos] = np.array(
                [L_val, curve(L_val)
                 + npr.uniform(-self.threshold, self.threshold)])
            N_pos += 1

        # Shuffle the data
        perm = torch.randperm(2*N)
        pos = torch.tensor(pos_data, dtype=torch.double)
        neg = torch.tensor(neg_data, dtype=torch.double)
        xs = torch.cat([pos, neg])[perm]
        ys = torch.cat([torch.zeros(N, dtype=torch.double),
                        torch.ones(N, dtype=torch.double)])[perm]
        return xs[:, 0], xs[:, 1], ys

    def eval_f(self, z):
        return 1-z**4

    def eval_g(self, z):
        return 1/(1-z**4)

    def integrate_L(self, zs):
        def integrand(y):
            z = zs*(1-y)*(1+y)
            return np.sqrt(self.eval_g(z))*y/np.sqrt(
                self.eval_f(z)/((1-y)**4*(1+y)**4*self.eval_f(zs))-1)
        L = 4*zs*quad(integrand, 0.0, 1.0)[0]
        return L

    def integrate_V(self, zs):
        def integrand(y):
            z = zs*(1-y)*(1+y)
            return np.sqrt(self.eval_f(z)*self.eval_g(z))/((1-y)**2*(1+y)**2)*y*(
                1/np.sqrt(1-((1-y)**4*(1+y)**4*self.eval_f(zs))/self.eval_f(z))
                - 1)

        def disconnected(z):
            return np.sqrt(self.eval_f(z)*self.eval_g(z))/z**2

        V = 4/zs*quad(integrand, 0.0, 1.0)[0]
        V -= 2*quad(disconnected, zs, 1)[0]
        return V
