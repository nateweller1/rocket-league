from jax import random, clear_caches

import numpyro
from numpyro.infer import SVI, Predictive, Trace_ELBO, autoguide


class NumpyroSVI:
    def __init__(self, model: callable, n_steps: int = 10000, n_samples: int = 2000):
        self.model = model
        self.n_steps = n_steps
        self.n_samples = n_samples
        self.rng_key = random.PRNGKey(716)
        self.posterior = None

    def fit(self, train_data: dict, progress_bar: bool = True, **kwargs):
        guide = autoguide.AutoNormal(self.model)

        optimizer = numpyro.optim.Adam(step_size=0.025)

        svi = SVI(
            model=self.model,
            guide=guide,
            optim=optimizer,
            loss=Trace_ELBO(),
        )

        svi_result = svi.run(
            rng_key=self.rng_key,
            num_steps=self.n_steps,
            progress_bar=progress_bar,
            **train_data,
            **kwargs
        )

        clear_caches()

        params = svi_result.params

        self.posterior = Predictive(guide, params=params, num_samples=self.n_samples)(self.rng_key, data=None)

    def predict(self, test_data: dict):
        return Predictive(self.model, self.posterior)(self.rng_key, **test_data)

