"""
This API appears to be deprecated. Will need to find an updated data source for future runs of the mdoel
"""

import pandas as pd
import requests

ENDPOINTS = {
    'events': 'https://zsr.octane.gg/events',
    'games': f"https://zsr.octane.gg/games"
}

COLS = {
    'player': {'player._id': 'PlayerId',
               'player.slug': 'Slug',
               'player.tag': 'PlayerTag',
               'player.country': 'Country',
               'stats.core.shots': 'Shots',
               'stats.core.goals': 'Goals',
               'stats.core.saves': 'Saves',
               'stats.core.assists': 'Assists',
               'stats.core.score': 'Score',
               'stats.demo.inflicted': 'Demos',
               'advanced.goalParticipation': 'GoalParticipation'},
    'games': {'_id': 'GameId',
              'number': 'GameNumber',
              'duration': 'GameDuration',
              'date': 'GameDate',
              'match.event._id': 'EventId',
              'match.event.name': 'EventName',
              'match.event.region': 'Region',
              'match.event.tier': 'Tier',
              'match.stage.name': 'Stage',
              'match.format.length': 'SeriesLength',
              'blue.team.team._id': 'BlueTeamId',
              'blue.team.team.name': 'BlueTeamName',
              'blue.players': 'BluePlayers',
              'blue.team.stats.core.shots': 'BlueShots',
              'blue.team.stats.core.goals': 'BlueGoals',
              'blue.team.stats.core.saves': 'BlueSaves',
              'blue.team.stats.core.assists': 'BlueAssists',
              'blue.team.stats.core.score': 'BlueScore',
              'orange.team.team._id': 'OrangeTeamId',
              'orange.team.team.name': 'OrangeTeamName',
              'orange.players': 'OrangePlayers',
              'orange.team.stats.core.shots': 'OrangeShots',
              'orange.team.stats.core.goals': 'OrangeGoals',
              'orange.team.stats.core.saves': 'OrangeSaves',
              'orange.team.stats.core.assists': 'OrangeAssists',
              'orange.team.stats.core.score': 'OrangeScore'
              }
}


def _get_events(tier: str):
    params = {'tier': tier, 'perPage': 1000}
    response = requests.get(ENDPOINTS['events'], params=params)
    events_json = response.json()
    events_df = pd.json_normalize(events_json['events'])

    return events_df['_id'].to_list()


def _get_games(event_id: str):
    params = {'event': event_id, 'perPage': 1000}
    response = requests.get(ENDPOINTS['games'], params=params)
    game_json = response.json()

    return pd.json_normalize(game_json['games'])


if __name__ == '__main__':

    all_events = []
    for tier in ['S', 'A']:
        events = _get_events(tier=tier)
        all_events += events

    all_games = []
    for event in events:
        print(event)
        games = _get_games(event)
        all_games.append(games)

    all_games_df = pd.concat(all_games).rename(columns=COLS['games'])[COLS['games'].values()]

    # Unpack player info
    for team in ['Blue', 'Orange']:
        players = pd.json_normalize(all_games_df[f'{team}Players'])
        for player_num in [0, 1, 2]:
            player = pd.json_normalize(players[player_num]).rename(columns=COLS['player'])[COLS['player'].values()]
            player.columns = f'{team}Player{player_num}' + player.columns
            all_games_df = pd.concat([all_games_df.reset_index(drop=True), player.reset_index(drop=True)], axis=1)


