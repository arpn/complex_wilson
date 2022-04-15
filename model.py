import logging
import torch
import torch.nn as nn
import numpy as np
from numpy.random import random
from scipy.interpolate import interp1d
from constants import dreal, dcomplex


class AdSBHNet(nn.Module):
    def __init__(self, N=1, std=1.0):
        super(AdSBHNet, self).__init__()
        self.a = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal))
        self.b = nn.Parameter(torch.normal(0.0, std, size=(N,), dtype=dreal))
        '''
        `self.logcoef` is the log of the dimensionless parameter
        R^2/(2*pi*alpha') which multiplies the static potential V.
        '''
        self.logcoef = nn.Parameter(torch.tensor(0.0, dtype=dreal))
        '''
        The lattice data is supposed to be shifted such that it behaves
        correctly in the UV. `self.shift` holds that parameter.
        '''
        self.shift = nn.Parameter(torch.tensor(0.0, dtype=dreal))
        self.curve_L = []
        self.curvs_zs = []

    def check_positive_a(self):
        '''
        Checks that the function a(z) defined by self.a
        is positive in the range [0, 1]. We'll do this in
        an inelegant way because a(z) can be a high order
        polynomial where we don't know the minima analytically.
        '''
        with torch.no_grad():
            z_grid = torch.linspace(0, 0.999, steps=1000)
            _a = torch.cat((torch.tensor([1.0], dtype=dreal),
                            torch.zeros(3, dtype=dreal),
                            self.a))
            a_grid = torch.zeros_like(z_grid)
            for i, ci in enumerate(_a):
                a_grid += ci * z_grid**i
            assert torch.all(a_grid > 0), "a(z) is not positive."

    def check_positive_b(self):
        '''
        Same for b(z).
        '''
        with torch.no_grad():
            z_grid = torch.linspace(0, 0.999, steps=1000)
            _b = torch.cat((torch.tensor([1.0], dtype=dreal),
                            torch.zeros(3, dtype=dreal),
                            self.b))
            b_grid = torch.zeros_like(z_grid)
            for i, ci in enumerate(_b):
                b_grid += ci * z_grid**i
            assert torch.all(b_grid > 0), "b(z) is not positive."

    def forward(self, Ls):
        '''
        Initial version with torch.trapz
        instead of torchdiffeq.
        '''
        V = torch.zeros_like(Ls, dtype=dcomplex)

        self.check_positive_a()
        self.check_positive_b()
        self.find_curve(Ls.max().item())
        curve = interp1d(self.curve_L, self.curve_zs)

        for i, L in enumerate(Ls):
            init = complex(curve(L.item()))
            _zs = self.find_zs_newton(L, init)
            # TODO: Verify that flipping possibly negative imaginary parts is always safe
            zs = torch.complex(_zs.real, _zs.imag.abs())
            assert zs.real > 0, f'Real part of zs is negative: {zs} for L = {L}'
            V[i] = self.integrate_V(zs)
            assert not torch.isnan(V[i])
        return V

    def as_tensor(self, tensor, dtype):
        '''
        Converts `tensor` to a torch.tensor if it
        is not already a tensor.
        '''
        if isinstance(tensor, torch.Tensor) and tensor.dtype == dtype:
            return tensor
        else:
            return torch.as_tensor(tensor, dtype=dtype)

    def find_curve(self, L_high=1.0):
        zs_max, L_max = self.get_L_max()
        self.curve_L = [0.0, L_max.item()]
        self.curve_zs = [complex(0.0, 0.0), zs_max.item()]

        L_step_default = (L_high - L_max.item()) / 50
        L_step = L_step_default
        while True:
            L = self.curve_L[-1] + L_step
            if np.abs(self.curve_zs[-1].imag) < 1e-8:
                init = self.curve_zs[-1] + 0.1j
            else:
                init = self.curve_zs[-1]
            _zs = self.find_zs_newton(L, init)
            zs = complex(_zs.real.item(), _zs.imag.abs().item() if L > L_max else 0.0)
            if np.abs(zs - self.curve_zs[-1]) > 0.1:
                L_step /= 2
            else:
                self.curve_zs.append(zs)
                self.curve_L.append(L)
                L_step = L_step_default
            if self.curve_L[-1] > L_high:
                break

<<<<<<< HEAD
    def find_zs_newton(self, L, init, max_steps=25, retry=10):
        init = self.as_tensor(init, dcomplex)
=======
    def find_zs_newton(self, L, init, max_steps=50, retry=10):
        if not isinstance(init, torch.Tensor) or init.dtype != dcomplex:
            init = torch.as_tensor(init, dtype=dcomplex)
>>>>>>> no_squaring
        zs = [init]
        _L = self.integrate_L(zs[-1])
        for i in range(max_steps):
            dL = self.integrate_dL(zs[-1])
            zs.append(zs[-1] - (_L - L) / dL)
            assert zs[-1].abs() < 100 and not torch.isnan(zs[-1]), f'Something wrong in Newton:\n\tzs = {[_zs.item() for _zs in zs]}\n\tL = {L}\n\t_L = {_L}\n\tdL = {dL}'
            _L = self.integrate_L(zs[-1])
            diff = torch.abs(_L - L)
            if diff < 1e-8:
                return zs[-1]
        if retry > 0:
            rand_init = init + (0.2 * random() - 0.1) + 1.j * (0.2 * random() - 0.1)
            # Im(rand_init) should be positive
            rand_init = rand_init.real + 1j * rand_init.imag.abs()
            logging.warning(f'Newton\'s method failed to converge in {max_steps} iterations for L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}. Retrying with random init {rand_init:.5f}.')
            return self.find_zs_newton(L, rand_init, retry=retry - 1)
        # Save the last complex path for debugging
        self.newton_zs = zs
        assert retry > 0, f'Newton\'s method failed to converge in {max_steps} iterations for L = {L}\n\tzs = {zs[-1]}\n\tdiff = {diff}\n\tinit = {init}.'

    def get_L_max(self):
        '''
        Returns the point where L is maximal such that
        zs is still real. This is the last point on the
        real axis along the real L curve.
        '''
        zs_UV, zs_IR = 0.001, 0.999
        dL_IR = self.integrate_dL(zs_IR).real
        dL_UV = self.integrate_dL(zs_UV).real
        assert dL_IR < 0 and dL_UV > 0
        while zs_IR - zs_UV > 1e-8:
            zs_mid = (zs_UV + zs_IR) / 2
            dL_mid = self.integrate_dL(zs_mid).real
            if dL_mid < 0:
                zs_IR = zs_mid
            else:
                zs_UV = zs_mid
        zs_mid = (zs_UV + zs_IR) / 2
        L_max = self.integrate_L(zs_mid)
        assert L_max.imag.abs() < 1e-8
        return torch.tensor(zs_mid, dtype=dcomplex), L_max.real

    def integrate_L(self, zs):
        '''
        This computes the dimensionless combination T*L,
        where T = 1/(pi*z_h).
        '''
<<<<<<< HEAD
        zs = self.as_tensor(zs, dcomplex)
=======
        zs = torch.as_tensor(zs, dtype=dcomplex)
>>>>>>> no_squaring
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal)
        z = zs * (1 - y) * (1 + y)
        sqrtg = self.eval_g(z).sqrt()
        f_over_fs = self.eval_f(z) / self.eval_f(zs)
        integrand = sqrtg / \
            torch.sqrt(f_over_fs / ((1 - y)**4 * (1 + y)**4) - 1) * y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        L = 4 * zs * torch.trapz(integrand, y) / np.pi * torch.sqrt((1 + self.a.sum()) / (1 + self.b.sum()))
        assert not torch.isnan(L), f'integrate_L({zs}) = {L} for a = {self.a} b = {self.b}'
        return L

    def integrate_dL(self, zs):
        '''
        This computes the derivative of T*L w.r.t. z_*/z_h.
        '''
<<<<<<< HEAD
        zs = self.as_tensor(zs, dcomplex)
=======
        zs = torch.as_tensor(zs, dtype=dcomplex)
>>>>>>> no_squaring
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal)
        z = zs * (1 - y) * (1 + y)
        fs = self.eval_f(zs)
        g = self.eval_g(z)
        f_over_fs = self.eval_f(z) / fs
        df = self.eval_df(z)
        dfs = self.eval_df(zs)
        dg = self.eval_dg(z)
        integrand = (zs**4 / z**4 * f_over_fs * (zs * dfs / fs + 2 + z * dg / g) - zs**4 / z**3 * df / fs - 2 - z * dg / g)
        integrand *= 2 * torch.sqrt(1 - z / zs) * torch.sqrt(g) / (zs**4 / z**4 * f_over_fs - 1)**1.5
        # Extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add known value for y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        dL = torch.trapz(integrand, y) / np.pi * torch.sqrt((1 + self.a.sum()) / (1 + self.b.sum()))
        assert not torch.isnan(dL), f'integrate_dL({zs}) = {dL} for a = {self.a} b = {self.b}'
        return dL

    def integrate_V(self, zs):
        V_c = self.integrate_V_connected(zs)
        # Then we subtract the disconnected configuration
        V_d = self.integrate_V_disconnected(zs)
        return V_c - V_d

    def integrate_V_connected(self, zs):
        '''
        This computes the connected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
<<<<<<< HEAD
        zs = self.as_tensor(zs, dcomplex)
=======
        zs = torch.as_tensor(zs, dtype=dcomplex)
>>>>>>> no_squaring
        y = torch.linspace(0.001, 0.999, steps=1000, dtype=dreal)
        z = zs * (1 - y) * (1 + y)
        f = self.eval_f(z)
        fg = f * self.eval_g(z)
        f_over_fs = f / self.eval_f(zs)
        integrand = torch.sqrt(fg) / ((1 - y)**2 * (1 + y)**2) * \
            (1 / torch.sqrt(1 - (1 - y)**4 * (1 + y)**4 / f_over_fs) - 1) * y
        # We extrapolate to y=0
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat(
            (((integrand[1] - integrand[0]) / (y[1] - y[0]) * (-y[0]) + integrand[0]).unsqueeze(-1), integrand))
        # Add analytically known value at y=1
        y = torch.cat((y, torch.tensor([1.0], dtype=dreal)))
        integrand = torch.cat((integrand, torch.tensor([0.0], dtype=dcomplex)))
        # Integrate
        coef = self.logcoef.exp()
        V = coef * np.pi * 4 * torch.trapz(integrand, y) / zs * torch.sqrt((1 + self.b.sum()) / (1 + self.a.sum()))
        assert not torch.isnan(V), f'integrate_V_connected({zs}) = {V} for a = {self.a} b = {self.b}'
        return V

    def integrate_V_disconnected(self, zs):
        '''
        This computes the disconnected contribution of V/T,
        where T = 1/(pi*z_h).
        '''
        zs = torch.as_tensor(zs, dtype=dcomplex)
        # Coordinate is y = (1 - z) / (1 - zs)
        y = torch.linspace(0.001, 1, steps=1000, dtype=dreal)
        z = 1 - (1 - zs) * y
        fg = self.eval_f(z) * self.eval_g(z)
        integrand = torch.sqrt(fg) / z**2
        # NOTE: This assumes that f(z)*g(z)->1 when z->1
        y = torch.cat((torch.tensor([0.0], dtype=dreal), y))
        integrand = torch.cat((torch.tensor([1.0], dtype=dreal), integrand))
        coef = self.logcoef.exp()
        V = coef * np.pi * 2 * (1 - zs) * torch.trapz(integrand, y) * torch.sqrt((1 + self.b.sum()) / (1 + self.a.sum()))
        assert not torch.isnan(V), f'integrate_V_disconnected({zs}) = {V} for a = {self.a} b = {self.b}'
        return V

    def eval_f(self, z):
<<<<<<< HEAD
        z = z if isinstance(z, torch.Tensor) else torch.as_tensor(z)
        z = self.as_tensor(z, dcomplex)
=======
        z = torch.as_tensor(z, dtype=dcomplex)
>>>>>>> no_squaring
        out = torch.zeros_like(z)
        _a = torch.cat((torch.tensor([1.0], dtype=dreal),
                        torch.zeros(3, dtype=dreal),
                        self.a))
        for i, ci in enumerate(_a):
            if i == 4:
                out += -4 * ci * z**4 * torch.log(z)
            else:
                out += 4 * ci * (z**4 - z**i) / (i - 4)
        return out

    def eval_df(self, z):
        out = torch.zeros_like(z)
        _a = torch.cat((torch.tensor([1.0], dtype=dreal),
                        torch.zeros(3, dtype=dreal),
                        self.a))
        for i, ci in enumerate(_a):
            out += -4 * ci * z**i
        out += 4 * self.eval_f(z)
        out /= z
        # TODO: add z->0 limit exactly
        return out

    def eval_b(self, z):
        z = torch.as_tensor(z, dtype=dcomplex)
        out = torch.zeros_like(z)
        _b = torch.cat((torch.tensor([1.0], dtype=dreal),
                        torch.zeros(3, dtype=dreal),
                        self.b))
        for i, ci in enumerate(_b):
            out += ci * z**i
        return out

    def eval_db(self, z):
        z = torch.as_tensor(z, dtype=dcomplex)
        out = torch.zeros_like(z)
        _b = torch.cat((torch.tensor([1.0], dtype=dreal),
                        torch.zeros(3, dtype=dreal),
                        self.b))
        for i, ci in enumerate(_b[1:]):
            out += (i + 1) * ci * z**i
        return out

    def eval_g(self, z):
        return self.eval_b(z) / ((1 - z) * (1 + z) * (1 + z**2))

    def eval_dg(self, z):
        out = 4 * z**3 * self.eval_b(z) / ((1 - z) * (1 + z) * (1 + z**2))**2
        out += self.eval_db(z) / ((1 - z) * (1 + z) * (1 + z**2))
        return out
