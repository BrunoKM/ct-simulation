import context
import numpy as np
from ct_detect import *
from photons import photons
from material import Material
from source import Source


def test_photons_init():
    material = Material()
    source = Source()
    coeff = material.coeff('Bone')
    y = photons(source.photons[0], coeff, [0.0, 0.1, 200])
    assert isinstance(y, np.ndarray)
    return


def test_photons_with_ct_detect():
    material = Material()
    source = Source()
    coeff = material.coeff('Water')

    y = ct_detect(source.photons[0], coeff, np.arange(0, 10.1, 0.1), 1)

    assert isinstance(y, np.ndarray)
    return
