"""Explicit Mellin identities shared by numerical reference tests."""

from functools import cache

import sympy as sp

from fftloggin.symbolic import from_mellin


@cache
def bessel_factory(order=0, spherical=False, after=False):
    s, mu, power = sp.symbols("s mu power")
    scale = sp.Symbol("scale", positive=True)
    z = s - order + power
    if spherical:
        mellin = (
            sp.sqrt(sp.pi)
            * 2 ** (z - 2)
            * sp.gamma((mu + z) / 2)
            / sp.gamma((mu + 3 - z) / 2)
        )
        upper = 2
    else:
        mellin = 2 ** (z - 1) * sp.gamma((mu + z) / 2) / sp.gamma((mu + 2 - z) / 2)
        upper = sp.Rational(3, 2)
    factor = sp.prod(s + (power if after else 0) - i for i in range(1, order + 1))
    return from_mellin(
        (-1) ** order * factor * scale**-z * mellin,
        s,
        parameters=(mu, power, scale),
        strip=(-mu + order - power, upper + order - power),
    )


def bessel_kernel(mu, *, power=0, scale=1, order=0, spherical=False, after=False):
    return bessel_factory(order, spherical, after)(mu, power, scale)
