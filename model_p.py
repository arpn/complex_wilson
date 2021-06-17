import torch
import torch.nn as nn
import numpy as np
from constants import dtype
# from spherical_jn import SphericalJN, SphericalJND


class AdSBHNet(nn.Module):
    def __init__(self, N=5):
        super(AdSBHNet, self).__init__()
        self.a = nn.Parameter(torch.normal(0.0, 1, size=(N,), dtype=dtype))
        # self.a = nn.Parameter(torch.zeros(N, dtype=dtype))
        # self.b = nn.Parameter(torch.zeros(N, dtype=dtype))

    def forward(self, Ls):
        '''
        Initial version with torch.trapz
        instead of torchdiffeq.
        '''
        V = torch.zeros_like(Ls, dtype=dtype)
        ps_prev = 10**4.1
        L_prev = self.integrate_L(ps_prev)
        '''
        This part could be made faster by if we
        first find L_max and then binary search for ps.
        This assumes that in (0, L_max) the function
        ps = ps(L) is monotonous.
        '''
        for ps in 10**torch.linspace(4, -1, 100):
            if self.integrate_V(ps) > 0:
                break
            L = self.integrate_L(ps)
            for i in range(len(Ls)):
                if L_prev < Ls[i] < L:
                    slope = (ps-ps_prev)/(L-L_prev)
                    V[i] = self.integrate_V(slope*(Ls[i]-L_prev)+ps_prev)
            ps_prev = ps
            L_prev = L
        return V

    def get_Lmax(self):
        ps_IR, ps_UV = 0.001, 10000
        V_IR = self.integrate_V(ps_IR)
        V_UV = self.integrate_V(ps_UV)
        assert V_IR > 0 and V_UV < 0
        while ps_UV-ps_IR > 1e-8:
            ps_mid = 10**((np.log10(ps_IR)+np.log10(ps_UV))/2)
            V_mid = self.integrate_V(ps_mid)
            if V_mid > 0:
                ps_IR = ps_mid
                V_IR = V_mid
            else:
                ps_UV = ps_mid
                V_UV = V_mid
        return self.integrate_L((ps_IR+ps_UV)/2)

    def integrate_L(self, ps):
        y = torch.linspace(0, 0.9999, steps=1000, dtype=dtype)
        p = ps/((1-y)*(1+y))
        z = self.eval_z(p)
        z_der = self.eval_z_derivative(p)
        sqrtg = self.eval_sqrtg(z)
        integrand = -sqrtg*z_der/((1-y)*(1+y)*torch.sqrt(2-y**2))
        # We append the analytically known value at y=1
        y = torch.cat([y, torch.tensor([1.0], dtype=dtype)])
        integrand = torch.cat([integrand, torch.tensor([0.0], dtype=dtype)])
        L = 4*ps*torch.trapz(integrand, y)
        return L

    def integrate_V(self, ps):
        V_c = self.integrate_V_connected(ps)
        # Then we subtract the disconnected configuration
        V_d = self.integrate_V_disconnected(ps)
        return V_c - V_d

    def integrate_V_connected(self, ps):
        _y = torch.linspace(0, 0.9999, steps=1000, dtype=dtype)
        p = ps/((1-_y)*(1+_y))
        z = self.eval_z(p)
        z_der = self.eval_z_derivative(p)
        sqrtg = self.eval_sqrtg(z)
        _integrand = -sqrtg/((1-_y)**3*(1+_y)**3)*(1/torch.sqrt(2-_y**2)-_y)*z_der
        # We append the analytically known value at y=1
        y = torch.cat([_y, torch.tensor([1.0], dtype=dtype)])
        integrand = torch.cat([_integrand, torch.tensor([0.0], dtype=dtype)])
        V = 4*ps**2*torch.trapz(integrand, y)
        return V

    def integrate_V_disconnected(self, ps):
        end = ps.item() if isinstance(ps, torch.Tensor) else ps
        _p = torch.linspace(np.log(0.001), np.log(end), steps=1000, dtype=dtype).exp()
        _p[-1] = ps
        z = self.eval_z(_p)
        z_der = self.eval_z_derivative(_p)
        sqrtg = self.eval_sqrtg(z)
        # These limits depend on the specific choice of a(z) and b(p).
        # TODO: Verify that these are ok.
        _integrand = _p*sqrtg*(-z_der)
        p = torch.cat([torch.tensor([0.0], dtype=dtype), _p])
        integrand = torch.cat([torch.tensor([0.0], dtype=dtype), _integrand])
        V = 2*torch.trapz(integrand, p)
        return V

    def eval_f(self, f):
        return 0

    def eval_g(self, z):
        return self.eval_sqrtg(z)**2

    def eval_sqrtg(self, z):
        a = torch.zeros_like(z, dtype=dtype)
        N = len(self.a)
        for i, coef in enumerate(self.a):
            a += coef*(z**(i+1)-z**(N+1))
        return torch.exp(a)/torch.sqrt((1-z)*(1+z)*(1+z**2))

    def eval_z(self, p):
        # spherical_jn = SphericalJN.apply
        # b = torch.zeros_like(p, dtype=dtype)
        # for i, coef in enumerate(self.b):
        #     it = torch.tensor(i)
        #     b += coef*spherical_jn(it+1, p)
        # self.z = torch.exp(b)*(1+p**2)**(-0.25)
        # return torch.exp(b)*(1+p**2)**(-0.25)
        return (1+p**2)**(-0.25)

    def eval_z_derivative(self, p):
        # spherical_jn = SphericalJN.apply
        # spherical_jnd = SphericalJND.apply
        # b = torch.zeros_like(p, dtype=dtype)
        # b_der = torch.zeros_like(p, dtype=dtype)
        # for i, coef in enumerate(self.b):
        #     it = torch.tensor(i)
        #     b += coef*spherical_jn(it+1, p)
        #     b_der += coef*spherical_jnd(it+1, p)
        # z_der = -0.5*p*(1+p**2)**(-1.25)*torch.exp(b) + \
        #     (1+p**2)**(-0.25)*torch.exp(b)*b_der
        # self.z_der = z_der
        # return z_der
        return -0.5*p*(1+p**2)**(-1.25)
