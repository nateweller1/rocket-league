from numpyro.infer import MCMC, NUTS, Predictive
from jax import random, clear_caches


class NumpyroMCMC:
    def __init__(self, model: callable, n_samples: int):
        self.model = model
        self.mcmc = MCMC(NUTS(model), num_warmup=int(n_samples / 2), num_samples=n_samples)
        self.posterior = None
        self.rng_key = random.PRNGKey(716)

    def fit(self, train_data: dict, **kwargs):
        rng_key, rng_key_ = random.split(self.rng_key)

        self.mcmc.run(rng_key_, **train_data, **kwargs)
        clear_caches()

        self.posterior = self.mcmc.get_samples()

    def predict(self, test_data: dict):
        return Predictive(self.model, self.posterior)(self.rng_key, **test_data)
