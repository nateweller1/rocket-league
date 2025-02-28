import polars as pl
import numpy as np

from sklearn.preprocessing import LabelEncoder

from src.rocket_league.models.classes import LMORegionPriors
from src.rocket_league.models.utils.mcmc import NumpyroMCMC
from src.rocket_league.models.utils.svi import NumpyroSVI


class RLRatings:
    def __init__(self):
        self.model = LMORegionPriors.lmo_w_region_priors
        self.le = LabelEncoder()
        self.samples = None

    @staticmethod
    def _load_data() -> pl.DataFrame:
        return (
            pl.read_csv('src/rocket_league/data/games.csv')
            .with_columns(
                TierWeight=pl.when(pl.col("Tier") == "A")
                .then(.30)
                .otherwise(1))
        )

    @staticmethod
    def _add_seasons(games_df: pl.DataFrame) -> pl.DataFrame:
        return (
            games_df.with_columns(GameDate=pl.col('GameDate').str.to_datetime())
            .with_columns(Season=pl.col('GameDate').dt.year())
        )

    @staticmethod
    def _drop_bad_data(games_df: pl.DataFrame) -> pl.DataFrame:
        games_df = games_df.filter(~pl.col('Season').is_null())
        games_df = games_df.filter(~pl.col('OrangePlayer2PlayerId').is_null())
        games_df = games_df.filter(~pl.col('BluePlayer2PlayerId').is_null())
        return games_df

    @staticmethod
    def _normalize_score_factors(games_df: pl.DataFrame) -> pl.DataFrame:
        all_scores = (
                list(games_df['BluePlayer0Score'].cast(pl.Int64)) +
                list(games_df['BluePlayer1Score'].cast(pl.Int64)) +
                list(games_df['BluePlayer2Score'].cast(pl.Float64).cast(pl.Int64)) +
                list(games_df['BluePlayer0Score'].cast(pl.Int64)) +
                list(games_df['BluePlayer1Score'].cast(pl.Int64)) +
                list(games_df['BluePlayer2Score'].cast(pl.Float64).cast(pl.Int64))
        )

        avg_score = sum(all_scores) / len(all_scores)

        for team in ['Blue', 'Orange']:
            for num in [0, 1, 2]:
                games_df = (
                    games_df.with_columns(
                        pl.Series(((games_df[f'{team}Player{num}Score'].cast(pl.Float64).cast(pl.Int64)) - avg_score) /
                                  (np.array(all_scores).std())).alias(f'{team}Player{num}Score'))
                )

        return games_df

    def _build_player_encoder(self, games_df: pl.DataFrame) -> pl.DataFrame:
        players_df_list = []
        for team in ['Blue', 'Orange']:
            for num in [0, 1, 2]:
                players_df_list += [(
                    games_df.select([f'{team}Player{num}PlayerId', f'{team}Player{num}PlayerTag'])
                    .rename({f'{team}Player{num}PlayerId': 'PlayerId', f'{team}Player{num}PlayerTag': 'PlayerName'})
                    .unique()
                )]

        players_df = pl.concat(players_df_list).unique()
        self.le.fit(players_df['PlayerId'])
        players_df = players_df.with_columns(Encoding=self.le.transform(players_df['PlayerId']))

        return players_df

    def _encode_players(self, games_df: pl.DataFrame) -> pl.DataFrame:
        for team in ['Blue', 'Orange']:
            for num in [0, 1, 2]:
                games_df = games_df.with_columns(
                    pl.Series(self.le.transform(games_df[f'{team}Player{num}PlayerId'])).alias(f'{team}{num}Encoding')
                )

        return games_df

    def _encode_season(self, games_df: pl.DataFrame) -> pl.DataFrame:
        return games_df.with_columns(SeasonEncoding=self.le.fit_transform(games_df['Season']))

    def _find_player_regions(self, games_df: pl.DataFrame, all_players_df: pl.DataFrame) -> pl.DataFrame:
        players_long = []
        for team in ['Blue', 'Orange']:
            for num in [0, 1, 2]:
                players_long += [games_df[['Region', f'{team}{num}Encoding']].rename({f'{team}{num}Encoding': 'Encoding'})]

        players_df = pl.concat(players_long)

        player_regions_df = (
            players_df.filter(pl.col('Region') != 'INT')
            .group_by(['Encoding', 'Region'])
            .len()
            .sort('len', descending=True)
            .group_by(['Encoding'])
            .first()
            .sort('Encoding')
        )

        player_regions_df = player_regions_df.with_columns(
            RegionEncoding=pl.Series(self.le.fit_transform(player_regions_df['Region']))
        )

        regions = (
            all_players_df.join(player_regions_df,
                                on='Encoding',
                                how='left',
                                coalesce=True)
            .with_columns(RegionEncoding=pl.col('RegionEncoding') + 1)
            .fill_null(0)
            .sort('Encoding')
        )

        return regions

    def load_process_data(self) -> [pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        games_df = self._load_data()
        proc_df = self._add_seasons(games_df)
        proc_df = self._drop_bad_data(proc_df)
        proc_df = self._normalize_score_factors(proc_df)
        players_df = self._build_player_encoder(proc_df)
        proc_df = self._encode_players(proc_df)
        player_regions_df = self._find_player_regions(proc_df, players_df)
        proc_df = self._encode_season(proc_df)

        return proc_df, players_df, player_regions_df

    @staticmethod
    def build_train_data(matches_df: pl.DataFrame, regions_df: pl.DataFrame) -> dict:
        return dict(
            player_regions=np.array(regions_df.unique(subset='Encoding')['RegionEncoding']),
            n_regions=regions_df['RegionEncoding'].max() + 1,
            season=np.array(matches_df['SeasonEncoding']),
            n_seasons=matches_df['SeasonEncoding'].max() + 1,
            blue_1=np.array(matches_df['Blue0Encoding']),
            blue_2=np.array(matches_df['Blue1Encoding']),
            blue_3=np.array(matches_df['Blue2Encoding']),
            orange_1=np.array(matches_df['Orange0Encoding']),
            orange_2=np.array(matches_df['Orange1Encoding']),
            orange_3=np.array(matches_df['Orange2Encoding']),
            y_blue=np.array(matches_df['BlueGoals'].cast(pl.Int32)),
            y_orange=np.array(matches_df['OrangeGoals'].cast(pl.Int32)),
            blue_1_score=np.array(matches_df['BluePlayer0Score'].cast(pl.Float32)),
            blue_2_score=np.array(matches_df['BluePlayer1Score'].cast(pl.Float32)),
            blue_3_score=np.array(matches_df['BluePlayer2Score'].cast(pl.Float32)),
            orange_1_score=np.array(matches_df['OrangePlayer0Score'].cast(pl.Float32)),
            orange_2_score=np.array(matches_df['OrangePlayer1Score'].cast(pl.Float32)),
            orange_3_score=np.array(matches_df['OrangePlayer2Score'].cast(pl.Float32)),
            tier_weights=np.array(matches_df["TierWeight"])
        )

    def fit(self, train_dict: dict) -> None:
        svi = NumpyroSVI(model=LMORegionPriors.lmo_w_region_priors)
        svi.fit(train_dict)
        self.samples = svi.posterior

    def extract_player_effects(self, matches_df: pl.DataFrame, players_df: pl.DataFrame) -> pl.DataFrame:
        current_season = matches_df["SeasonEncoding"].max()
        ovr_ratings = np.array(self.samples[f'ovr_ratings_season_{current_season}'].mean(axis=0))
        att_ratings = np.array(self.samples[f'att_ratings_season_{current_season}'].mean(axis=0))
        def_ratings = np.array(self.samples[f'def_ratings_season_{current_season}'].mean(axis=0))

        ratings_df = pl.DataFrame().with_columns(
            PlayerAtt=np.exp(att_ratings + ovr_ratings),
            PlayerDef=np.exp(def_ratings + ovr_ratings),
            PlayerRating=np.exp(((att_ratings + ovr_ratings)) + ((def_ratings + ovr_ratings))),
            Encoding=np.array(range(0, len(ovr_ratings)))
        )

        return ratings_df.join(players_df, on="Encoding")


if __name__ == '__main__':
    mod = RLRatings()
    matches, players, regions = mod.load_process_data()
    train_data = mod.build_train_data(matches, regions)
    mod.fit(train_data)
    player_ratings = mod.extract_player_effects(matches, players)





