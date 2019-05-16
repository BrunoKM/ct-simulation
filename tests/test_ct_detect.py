import context
import numpy as np
from ct_detect import *
from photons import photons
from material import Material
from source import Source


def test_ct_detect():
    material = Material()
    source = Source()
    coeff_arr = np.vstack([material.coeff('Bone'), material.coeff('Water')])
    depths = np.array([[1, 2., 3], [2., 2., 2.]], dtype=np.float32)
    y = ct_detect(source.photons[0], coeff_arr, depths)
    assert isinstance(y, np.ndarray)
    assert np.all(y >= 1.)
    return
