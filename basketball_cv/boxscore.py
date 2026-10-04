"""Итоговая статистика матча (box score).

Треки ByteTrack объединяются в игроков по паре (команда, номер на майке).
Трек без подтверждённого номера остаётся отдельной строкой «трек #N».
События с статусом rejected в статистику не входят.
"""

from collections import defaultdict

PLAYER_FIELDS = [
    "player", "team_name", "number", "name", "tracks",
    "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "TOV", "BLK", "PASS",
    "visible_s", "possession_s", "distance_m", "avg_speed_kmh", "max_speed_kmh", "measured_s",
]
TEAM_FIELDS = [
    "team_name", "PTS", "FGM", "FGA", "FG%", "3PM", "3PA", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "TOV", "BLK", "possession_s", "possession_%",
]
TRACK_FIELDS = [
    "track_id", "team", "number", "player", "visible_s", "possession_s", "distance_m",
    "avg_speed_kmh", "max_speed_kmh", "measured_s", "first_seen_s", "last_seen_s", "rejected_motion_samples",
]


def _value(event, key):
    return event.get(key) if isinstance(event, dict) else getattr(event, key)


def _int_or_none(value):
    if value is None or str(value).strip() in ("", "None", "nan"):
        return None
    return int(float(value))


def identity_key(track_row):
    team, number = track_row.get("team"), track_row.get("number")
    if team in (0, 1) and number not in (None, ""):
        return f"{team}:{number}"
    return f"track:{track_row['track_id']}"


def build_boxscore(track_rows, events, team_names, rosters):
    """track_rows: строки по трекам (с полями team, number); events: Event или dict.

    Возвращает (игроки, команды, отображение трек → ключ игрока).
    """
    players = {}
    track_to_player = {}
    for row in track_rows:
        key = identity_key(row)
        track_to_player[int(row["track_id"])] = key
        team = row.get("team")
        number = row.get("number") or ""
        roster = rosters[team] if team in (0, 1) and team < len(rosters) else {}
        record = players.setdefault(key, dict(
            player=key, team=team, team_name=team_names[team] if team in (0, 1) else "",
            number=number, name=roster.get(str(number), "") if number else "",
            tracks=[], visible_s=0.0, possession_s=0.0, distance_m=None, measured_s=0.0, max_speed_kmh=None,
        ))
        record["tracks"].append(int(row["track_id"]))
        record["visible_s"] += float(row.get("visible_s") or 0)
        record["possession_s"] += float(row.get("possession_s") or 0)
        if row.get("distance_m") not in (None, ""):
            record["distance_m"] = (record["distance_m"] or 0.0) + float(row["distance_m"])
            record["measured_s"] += float(row.get("measured_s") or 0)
        if row.get("max_speed_kmh") not in (None, ""):
            record["max_speed_kmh"] = max(record["max_speed_kmh"] or 0.0, float(row["max_speed_kmh"]))
    counters = defaultdict(lambda: defaultdict(int))
    team_totals = [defaultdict(int), defaultdict(int)]
    for event in events:
        if _value(event, "status") == "rejected":
            continue
        kind = _value(event, "kind")
        track = _int_or_none(_value(event, "player_id"))
        team = _int_or_none(_value(event, "team"))
        points = _int_or_none(_value(event, "points"))
        stat = defaultdict(int)
        if kind == "shot":
            if points == 1:
                stat["FTA"] += 1
            else:
                stat["FGA"] += 1
                stat["3PA"] += int(points == 3)
        elif kind == "made":
            points = points if points is not None else 2
            stat["PTS"] += points
            if points == 1:
                stat["FTM"] += 1
            else:
                stat["FGM"] += 1
                stat["3PM"] += int(points == 3)
        elif kind in ("rebound_off", "rebound_def"):
            stat["OREB" if kind == "rebound_off" else "DREB"] += 1
            stat["REB"] += 1
        else:
            name = {"assist": "AST", "steal": "STL", "turnover": "TOV", "block": "BLK", "pass": "PASS"}.get(kind)
            if name is None:
                continue
            stat[name] += 1
        key = track_to_player.get(track)
        for name, value in stat.items():
            if key is not None:
                counters[key][name] += value
            if team in (0, 1):
                team_totals[team][name] += value
    rows = []
    for key, record in players.items():
        row = dict(record)
        for name in ("PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "TOV", "BLK", "PASS"):
            row[name] = counters[key][name]
        row["tracks"] = " ".join(map(str, sorted(record["tracks"])))
        measured = record["measured_s"]
        row["distance_m"] = round(record["distance_m"], 1) if record["distance_m"] is not None else None
        row["avg_speed_kmh"] = round(record["distance_m"] / measured * 3.6, 2) if record["distance_m"] and measured else None
        row["measured_s"] = round(measured, 1)
        row["visible_s"] = round(record["visible_s"], 1)
        row["possession_s"] = round(record["possession_s"], 1)
        rows.append(row)
    rows.sort(key=lambda r: (r["team"] if r["team"] in (0, 1) else 9, -r["PTS"], -(r["visible_s"] or 0)))
    possession = [sum(r["possession_s"] for r in rows if r["team"] == t) for t in (0, 1)]
    total_possession = sum(possession)
    teams = []
    for t in (0, 1):
        totals = team_totals[t]
        fga, fgm = totals["FGA"], totals["FGM"]
        teams.append(dict(
            team_name=team_names[t], **{k: totals[k] for k in ("PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA",
                                                                "OREB", "DREB", "REB", "AST", "STL", "TOV", "BLK")},
            **{"FG%": round(100 * fgm / fga, 1) if fga else None,
               "possession_s": round(possession[t], 1),
               "possession_%": round(100 * possession[t] / total_possession, 1) if total_possession else None},
        ))
    return rows, teams, track_to_player
