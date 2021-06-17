import torch
import torch.nn as nn
import numpy as np
from constants import dtype
# from spherical_jn import SphericalJN, SphericalJND


class AdSBHNet(nn.Module):
    def __init__(self, N=5):
        super(AdSBHNet, self).__init__()
        self.a = nn.Parameter(torch.normal(0.0, 0.5, size=(N,), dtype=dtype))
        self.b = nn.Parameter(torch.normal(0.0, 0.5, size=(N,), dtype=dtype))

    def forward(self, Ls):
        '''
        Initial version with torch.trapz
        instead of torchdiffeq.
        '''
        V = torch.zeros_like(Ls, dtype=dtype)
        zs_prev = 0.0009
        L_prev = self.integrate_L(zs_prev)
        '''
        This part could be made faster and more accurate if
        we first find L_max and then binary search for zs.
        This assumes that in (0, L_max) the function
        zs = zs(L) is monotonous.
        '''
        for zs in torch.linspace(0.001, 0.999, 100):
            if self.integrate_V(zs) > 0:
                break
            L = self.integrate_L(zs)
            for i in range(len(Ls)):
                if L_prev < Ls[i] < L:
                    slope = (zs-zs_prev)/(L-L_prev)
                    V[i] = self.integrate_V(slope*(Ls[i]-L_prev)+zs_prev)
            zs_prev = zs
            L_prev = L
        return V

    def get_Lmax(self):
        zs_UV, zs_IR = 0.001, 0.999
        V_IR = self.integrate_V(zs_IR)
        V_UV = self.integrate_V(zs_UV)
        assert V_IR > 0 and V_UV < 0
        while zs_IR-zs_UV > 1e-8:
            zs_mid = (zs_UV+zs_IR)/2
            V_mid = self.integrate_V(zs_mid)
            if V_mid > 0:
                zs_IR = zs_mid
                V_IR = V_mid
            else:
                zs_UV = zs_mid
                V_UV = V_mid
        return self.integrate_L((zs_UV+zs_IR)/2)

    def integrate_L(self, zs):
        zs = zs if isinstance(zs, torch.Tensor) else torch.tensor(zs, dtype=dtype)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
        z = zs*(1-y)*(1+y)
        f = self.eval_f(z)
        sqrtg = self.eval_g(z).sqrt()
        fs = self.eval_f(zs.unsqueeze(-1))
        integrand = sqrtg/torch.sqrt(f/(fs*(1-y)**4*(1+y)**4)-1)*y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat((((integrand[1]-integrand[0])/(y[1]-y[0])*(-y[0])+integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
        # Integrate
        L = 4*zs*torch.trapz(integrand, y)
        return L

    def integrate_V(self, zs):
        V_c = self.integrate_V_connected(zs)
        # Then we subtract the disconnected configuration
        V_d = self.integrate_V_disconnected(zs)
        return V_c - V_d

    def integrate_V_connected(self, zs):
        zs = zs if isinstance(zs, torch.Tensor) else torch.tensor(zs, dtype=dtype)
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dtype)
        z = zs*(1-y)*(1+y)
        f = self.eval_f(z)
        g = self.eval_g(z)
        fs = self.eval_f(zs.unsqueeze(-1))
        integrand = torch.sqrt(f*g)/((1-y)**2*(1+y)**2)*(1/torch.sqrt(1-(1-y)**4*(1+y)**4*fs/f)-1)*y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dtype), y))
        integrand = torch.cat((((integrand[1]-integrand[0])/(y[1]-y[0])*(-y[0])+integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dtype)))
        # Integrate
        V = 4*torch.trapz(integrand, y)/zs
        return V

    def integrate_V_disconnected(self, zs):
        start = zs.item() if isinstance(zs, torch.Tensor) else zs
        z = torch.linspace(start, 0.999, steps=1000, dtype=dtype)
        z[0] = zs
        f = self.eval_f(z)
        g = self.eval_g(z)
        integrand = torch.sqrt(f*g)/z**2
        # Add analytically known horizon limit
        # NOTE: This assumes that f(z)*g(z)->1 when z->1
        z = torch.cat((z, torch.tensor([1.0], dtype=dtype)))
        integrand = torch.cat((integrand, torch.tensor([1.0], dtype=dtype)))
        # Integrate
        V = 2*torch.trapz(integrand, z)
        return V

    def eval_a(self, z):
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.a):
            for j, cj in enumerate(self.a):
                out += ci*cj*z**(i+j+1)/(i+j+1)
        # Normalize
        return out

    def eval_f(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.from_numpy(z)
        return (1-z)*(1+z)*(1+z**2)/self.eval_a(z).exp()

    def eval_b(self, z):
        out = torch.zeros_like(z)
        for i, ci in enumerate(self.b):
            out += ci*z**(i+1)
        a1 = self.eval_a(torch.tensor(1.0, dtype=dtype))
        N = len(self.b)
        out -= (self.b.sum()+a1)*z**(N+1)
        return out

    def eval_g(self, z):
        z = z if isinstance(z, torch.Tensor) else torch.from_numpy(z)
        return self.eval_b(z).exp()/((1-z)*(1+z)*(1+z**2))
