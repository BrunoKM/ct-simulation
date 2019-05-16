import context
import matplotlib.pyplot as plt
from ct_phantom import *
from material import Material
import pytest


@pytest.fixture
def material():
    """Return a new names instance."""
    return Material()


@pytest.mark.parametrize("phantom_type, res", [
    (1, 256),
    (2, 256),
    (3, 256),
    (4, 256),
    (5, 1256),
    (6, 1256),
    (7, 1256),
])
def test_ct_detect(material, phantom_type, res):
    phantom = ct_phantom(material.name, res, phantom_type, 'Titanium')
