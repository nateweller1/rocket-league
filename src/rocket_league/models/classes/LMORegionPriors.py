import math
from typing import *

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from numpyro import handlers


def lmo_w_region_priors(
        n_seasons: Optional[int],  # number of unique seasons in dataset
        n_regions: Optional[int],  # number of unique regions in dataset

        season: np.array,  # season encodings

        blue_1: np.array,  # blue player 1 encoding
        blue_2: np.array,  # blue player 2 encoding
        blue_3: np.array,  # blue player 3 encoding
        orange_1: np.array,  # orange player 1 encoding
        orange_2: np.array,  # orange player 2 encoding
        orange_3: np.array,  # orange player 3 encoding

        player_regions: np.array,  # Region of each player

        y_blue: np.array,  # goals differential
        y_orange: np.array,  # goals differential

        blue_1_score: np.array,
        blue_2_score: np.array,
        blue_3_score: np.array,
        orange_1_score: np.array,
        orange_2_score: np.array,
        orange_3_score: np.array,

        tier_weights: np.array,

        # Hyperparameters
        factor_weight: float = .10,
        region_sd: float = .25,
        player_sd_ovr: float = .05,
        player_sd_att: float = .05,
        player_sd_def: float = .05,

        predict: bool = False,
        sample_target: bool = False,
):
    # Intercept for goals scored, initialize main outcome mus
    icpt = numpyro.sample("icpt", dist.Normal(0, 1))
    mu_blue = icpt  # Blue team goals scored
    mu_orange = icpt  # Orange team goals scored

    # Factor output score, individual player scores
    icpt_score = numpyro.sample("icpt_score", dist.Normal(0, 1))

    mu_b1_score = icpt_score
    mu_b2_score = icpt_score
    mu_b3_score = icpt_score

    mu_o1_score = icpt_score
    mu_o2_score = icpt_score
    mu_o3_score = icpt_score

    # Player ratings normally distributed around a region prior
    ovr_region_priors = numpyro.sample('ovr_region_priors', dist.Normal(0, region_sd / 2), sample_shape=(n_regions, ))
    att_region_priors = numpyro.sample('att_region_priors', dist.Normal(0, region_sd), sample_shape=(n_regions, ))
    def_region_priors = numpyro.sample('def_region_priors', dist.Normal(0, region_sd), sample_shape=(n_regions, ))

    # Initialize player rating based on their region
    ovr_ratings = [numpyro.sample("ovr_ratings_season_0", dist.Normal(ovr_region_priors[player_regions], player_sd_ovr))]
    att_ratings = [numpyro.sample("att_ratings_season_0", dist.Normal(att_region_priors[player_regions], player_sd_att))]
    def_ratings = [numpyro.sample("def_ratings_season_0", dist.Normal(def_region_priors[player_regions], player_sd_def))]

    # Generate player/season level ratings where ability in season n-1 is passed as prior for season n
    for season_i in range(1, n_seasons):
        prev_ratings_ovr = ovr_ratings[season_i - 1]
        prev_ratings_att = att_ratings[season_i - 1]
        prev_ratings_def = def_ratings[season_i - 1]

        ovr_ratings_season = numpyro.sample(
            f"ovr_ratings_season_{season_i}",
            dist.Normal(prev_ratings_ovr, player_sd_ovr)
        )

        att_ratings_season = numpyro.sample(
            f"att_ratings_season_{season_i}",
            dist.Normal(prev_ratings_att, player_sd_att)
        )

        def_ratings_season = numpyro.sample(
            f"def_ratings_season_{season_i}",
            dist.Normal(prev_ratings_def, player_sd_def)
        )

        ovr_ratings += [ovr_ratings_season]
        att_ratings += [att_ratings_season]
        def_ratings += [def_ratings_season]

    ovr_ratings = jnp.stack(tuple(ovr_ratings))
    att_ratings = jnp.stack(tuple(att_ratings))
    def_ratings = jnp.stack(tuple(def_ratings))

    # Index to find player ratings for given season
    blue_1_ovr = ovr_ratings[season, blue_1]
    blue_1_att = att_ratings[season, blue_1]
    blue_1_def = def_ratings[season, blue_1]

    blue_2_ovr = ovr_ratings[season, blue_2]
    blue_2_att = att_ratings[season, blue_2]
    blue_2_def = def_ratings[season, blue_2]

    blue_3_ovr = ovr_ratings[season, blue_3]
    blue_3_att = att_ratings[season, blue_3]
    blue_3_def = def_ratings[season, blue_3]

    orange_1_ovr = ovr_ratings[season, orange_1]
    orange_1_att = att_ratings[season, orange_1]
    orange_1_def = def_ratings[season, orange_1]

    orange_2_ovr = ovr_ratings[season, orange_2]
    orange_2_att = att_ratings[season, orange_2]
    orange_2_def = def_ratings[season, orange_2]

    orange_3_ovr = ovr_ratings[season, orange_3]
    orange_3_att = att_ratings[season, orange_3]
    orange_3_def = def_ratings[season, orange_3]

    # RAPM style models projecting goals for based on players in game
    mu_blue += (
            ((blue_1_ovr + blue_1_att) + (blue_2_ovr + blue_2_att) + (blue_3_ovr + blue_3_att)) -
            ((orange_1_ovr + orange_1_def) + (orange_2_ovr + orange_2_def) + (orange_3_ovr + orange_3_def))
    )

    mu_orange += (
            ((orange_1_ovr + orange_1_att) + (orange_2_ovr + orange_2_att) + (orange_3_ovr + orange_3_att)) -
            ((blue_1_ovr + blue_1_def) + (blue_2_ovr + blue_2_def) + (blue_3_ovr + blue_3_def))
    )

    # Loading for score factor, half normal centered at 1
    loading_score = numpyro.sample("loading_score", dist.HalfNormal((math.pi**.5) / 2**.5))

    # Using individual player scores to aid the models in dividing credit
    mu_b1_score += ((blue_1_ovr + blue_1_att) + ((orange_1_ovr + orange_1_def) + (orange_2_ovr + orange_2_def) + (orange_3_ovr + orange_3_def))) * loading_score
    mu_b2_score += ((blue_2_ovr + blue_2_att) + ((orange_1_ovr + orange_1_def) + (orange_2_ovr + orange_2_def) + (orange_3_ovr + orange_3_def))) * loading_score
    mu_b3_score += ((blue_3_ovr + blue_3_att) + ((orange_1_ovr + orange_1_def) + (orange_2_ovr + orange_2_def) + (orange_3_ovr + orange_3_def))) * loading_score

    mu_o1_score += ((orange_1_ovr + orange_1_att) + ((blue_1_ovr + blue_1_def) + (blue_2_ovr + blue_2_def) + (blue_3_ovr + blue_3_def))) * loading_score
    mu_o2_score += ((orange_2_ovr + orange_2_att) + ((blue_1_ovr + blue_1_def) + (blue_2_ovr + blue_2_def) + (blue_3_ovr + blue_3_def))) * loading_score
    mu_o3_score += ((orange_3_ovr + orange_3_att) + ((blue_1_ovr + blue_1_def) + (blue_2_ovr + blue_2_def) + (blue_3_ovr + blue_3_def))) * loading_score

    # Transform poisson outputs
    mu_blue = jnp.exp(mu_blue)
    mu_orange = jnp.exp(mu_orange)

    if predict and sample_target:
        numpyro.sample("pred_blue", dist.Poisson(mu_blue))
        numpyro.sample("pred_orange", dist.Poisson(mu_orange))

    elif predict:
        numpyro.deterministic("pred_blue", mu_blue)
        numpyro.deterministic("pred_orange", mu_orange)

    sigma_score = numpyro.sample("sigma_score", dist.Exponential(1))

    # Decrease weight of A-Tier events
    with handlers.scale(scale=tier_weights):
        numpyro.sample('y_blue', dist.Poisson(mu_blue), obs=y_blue)
        numpyro.sample('y_orange', dist.Poisson(mu_orange), obs=y_orange)

        with handlers.scale(scale=factor_weight):
            numpyro.sample('factor_bs1', dist.Normal(mu_b1_score, sigma_score), obs=blue_1_score)
            numpyro.sample('factor_bs2', dist.Normal(mu_b2_score, sigma_score), obs=blue_2_score)
            numpyro.sample('factor_bs3', dist.Normal(mu_b3_score, sigma_score), obs=blue_3_score)

            numpyro.sample('factor_os1', dist.Normal(mu_o1_score, sigma_score), obs=orange_1_score)
            numpyro.sample('factor_os2', dist.Normal(mu_o2_score, sigma_score), obs=orange_2_score)
            numpyro.sample('factor_os3', dist.Normal(mu_o3_score, sigma_score), obs=orange_3_score)
