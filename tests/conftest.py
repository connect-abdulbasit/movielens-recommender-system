import numpy as np
import pytest

from src.data_loader import ERA_NAMES, GENRES

N_USERS, N_ITEMS = 40, 60

@pytest.fixture(scope="session")
def toy_data():
    rng = np.random.default_rng(1)
    matrix = np.where(rng.random((N_USERS, N_ITEMS)) < 0.4, rng.integers(1, 6, (N_USERS, N_ITEMS)), 0).astype(float)
    genres = (rng.random((N_ITEMS, len(GENRES))) < 0.15).astype(float)
    genres[np.arange(N_ITEMS), rng.integers(0, len(GENRES), N_ITEMS)] = 1
    eras = np.eye(len(ERA_NAMES))[rng.integers(0, len(ERA_NAMES), N_ITEMS)]
    return matrix, np.hstack([genres, eras]), GENRES + ERA_NAMES
