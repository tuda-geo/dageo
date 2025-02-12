import pytest
import numpy as np
from numpy.testing import assert_allclose

import dageo
from dageo import data_assimilation


def pseudopdf(data, bins=200, density=True, **kwargs):
    """Return the pdf from a simple bin count.

    If the data contains a lot of samples, this should be "smooth" enough - and
    much faster than estimating the pdf using, e.g.,
    `scipy.stats.gaussian_kde`.
    """
    x, y = np.histogram(data, bins=bins, density=density, **kwargs)
    return (y[:-1]+y[1:])/2, x


def forward(x, beta):
    """Simple model: y = x (1 + β x²) (linear if beta=0)."""
    return np.atleast_1d(x * (1 + beta * x**2))


def test_esmda_linear():
    # Use the simple linear ESMDA example from the gallery as simple test.
    xlocation = -1.0
    ne = int(1e7)
    obs_std = 1.0
    rng = dageo.utils.rng(1234)  # fixed seed for testing
    mprior = rng.normal(loc=1.0, scale=obs_std, size=(ne, 1))

    def lin_fwd(x):
        """Linear forward model."""
        return forward(x, beta=0.0)

    l_dobs = lin_fwd(xlocation)

    # Only return final model and data
    lm_post, ld_post = dageo.esmda(
        model_prior=mprior,
        forward=lin_fwd,
        data_obs=l_dobs,
        sigma=obs_std,
        alphas=4,
        random=3333,
    )
    assert lm_post.shape == (1e7, 1)
    assert ld_post.shape == (1e7, 1)

    x, p = pseudopdf(ld_post[:, 0])
    assert_allclose(0.563, np.max(p), atol=0.001)
    assert_allclose(0.012, x[np.argmax(p)], atol=0.001)

    # Also return steps
    lm_post2, ld_post2 = dageo.esmda(
        model_prior=mprior,
        forward=lin_fwd,
        data_obs=l_dobs,
        sigma=obs_std,
        alphas=4,
        return_steps=True,
        random=3333,
    )
    assert lm_post2.shape == (5, 1e7, 1)
    assert ld_post2.shape == (5, 1e7, 1)

    assert_allclose(lm_post2[-1, :, 0], lm_post[:, 0])
    assert_allclose(ld_post2[-1, :, 0], ld_post[:, 0])

    def cbp(x):
        x[:] /= 100  # Gets the model much narrowed around 0.

    # alpha-array, localization_matrix, callback_post, return only model
    lm_post3 = dageo.esmda(
        model_prior=mprior,
        forward=lin_fwd,
        data_obs=l_dobs,
        sigma=obs_std,
        alphas=[4, 4, 4, 4],
        localization_matrix=np.array([[0.5]]),
        callback_post=cbp,
        return_post_data=False,
        random=3333,
    )
    assert lm_post3.shape == (1e7, 1)

    ld_post3 = lin_fwd(lm_post3)
    x, p = pseudopdf(ld_post3[:, 0])
    assert_allclose(43420257, np.max(p), atol=1)
    assert_allclose(0.0, x[np.argmax(p)], atol=1e-8)

    # Warning
    with pytest.warns(UserWarning, match='provided: 1.25'):
        lm_post3 = dageo.esmda(
            model_prior=mprior,
            forward=lin_fwd,
            data_obs=l_dobs,
            sigma=obs_std,
            alphas=[4, 4, 4, 4, 4],
            localization_matrix=np.array([[0.5]]),
            callback_post=cbp,
            return_post_data=False,
            random=3333,
        )


def test_all_dir():
    assert set(data_assimilation.__all__) == set(dir(data_assimilation))
