"""Экспорт: CSV/JSON, HTML-отчёт, ZIP и служебные функции оценки качества."""

from dataclasses import asdict
from html import escape
from pathlib import Path
import csv
import json
import math
import zipfile

from .boxscore import PLAYER_FIELDS, TEAM_FIELDS, build_boxscore
from .events import EVENT_NAMES, Event

EVENT_FIELDS = list(Event.__dataclass_fields__)
REVIEW_EVENT_COLUMNS = ["event_id", "time_s", "kind", "player_id", "team", "status", "outcome", "points", "reason"]
STATUSES = ("automatic", "needs_review", "confirmed", "rejected", "observed", "ocr_observed")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _table(rows, columns):
    header = "".join(f"<th>{escape(str(label))}</th>" for _, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if value is None or value == "":
                value = "—"
            if key == "kind":
                value = EVENT_NAMES.get(value, value)
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def make_report(folder, summary, players, teams, events, suffix=""):
    folder = Path(folder)
    names = summary.get("team_names", ["Команда 1", "Команда 2"])
    score = summary.get("score") or [0, 0]
    ocr = summary.get("scoreboard_ocr")
    ocr_line = f"Табло (OCR, для сверки): {ocr[0]} : {ocr[1]}" if ocr else "Табло не считывалось"
    player_table = _table(players, [
        ("team_name", "Команда"), ("number", "№"), ("name", "Игрок"), ("tracks", "Треки"),
        ("PTS", "Очки"), ("FGM", "ПОП"), ("FGA", "БР"), ("3PM", "3-ПОП"), ("3PA", "3-БР"),
        ("FTM", "ШТР-ПОП"), ("FTA", "ШТР"), ("OREB", "ПН"), ("DREB", "ПЗ"), ("AST", "АС"),
        ("STL", "ПХ"), ("TOV", "ПОТ"), ("BLK", "БЛ"), ("possession_s", "Владение, с"),
        ("distance_m", "Дистанция, м"), ("avg_speed_kmh", "Ср. скорость, км/ч"),
        ("max_speed_kmh", "Макс., км/ч"), ("visible_s", "В кадре, с"),
    ])
    team_table = _table(teams, [
        ("team_name", "Команда"), ("PTS", "Очки"), ("FGM", "ПОП"), ("FGA", "БР"), ("FG%", "% попаданий"),
        ("3PM", "3-ПОП"), ("3PA", "3-БР"), ("FTM", "ШТР-ПОП"), ("FTA", "ШТР"), ("OREB", "ПН"), ("DREB", "ПЗ"),
        ("AST", "АС"), ("STL", "ПХ"), ("TOV", "ПОТ"), ("BLK", "БЛ"), ("possession_%", "Владение, %"),
    ])
    event_table = _table(events, [
        ("time_s", "Время, с"), ("kind", "Событие"), ("player_id", "Трек"), ("team", "Команда"),
        ("points", "Очки"), ("outcome", "Исход"), ("status", "Статус"), ("reason", "Основание"),
    ])
    warnings = "".join(f"<li>{escape(str(w))}</li>" for w in summary.get("warnings", []))
    quality = summary.get("calibration_quality", {})
    title = "проверенный отчёт" if suffix else "автоматический отчёт"
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Basketball CV — {title}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f8;color:#151a21;font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px}}h1{{font-size:26px;margin:0 0 4px;font-weight:650}}
h2{{font-size:18px;margin:36px 0 10px;padding-left:10px;border-left:4px solid #d0561b}}
.score{{display:flex;align-items:center;gap:20px;margin:20px 0 2px;padding:14px 20px;background:#151a21;color:#fff;border-radius:10px;width:fit-content}}
.score .team{{font-size:18px}}.score .pts{{font:700 48px/1 'Arial Narrow','Roboto Condensed',system-ui,sans-serif;font-variant-numeric:tabular-nums;color:#ffb36b}}
.muted{{color:#5b6573}}.facts{{display:flex;flex-wrap:wrap;gap:6px 26px;font-size:14px}}
.scroll{{overflow-x:auto;background:#fff;border:1px solid #dde1e7;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;border-bottom:1px solid #e7eaee;padding:7px 9px;vertical-align:top;white-space:nowrap}}
td:last-child{{white-space:normal}}th{{background:#eef0f3;color:#39424e}}video,img{{width:100%;border-radius:8px;background:#1b2230}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
.note{{border-left:4px solid #d0561b;background:#fff;padding:10px 16px}}
</style><main>
<h1>Basketball CV — {title}</h1>
<div class="muted">{escape(summary.get("source_name", ""))} · обработано {summary.get("processed_seconds", 0):.1f} с видео</div>
<div class="score"><span class="team">{escape(names[0])}</span><span class="pts">{score[0]} : {score[1]}</span>
<span class="team">{escape(names[1])}</span></div>
<div class="muted">Счёт по распознанным попаданиям · {escape(ocr_line)}</div>
<div class="facts" style="margin-top:14px">
<span>Игроков: {summary.get("player_count", len(players))} (треков {summary.get("track_count", "—")}, с номером {summary.get("identified_tracks", "—")})</span>
<span>Событий: {len(events)}</span><span>Судей в кадре: до {summary.get("max_referees_seen", 0)}</span><span>Мяч виден: {100 * summary.get("ball_detection_fraction", 0):.0f}% кадров</span>
<span>Метры доступны: {100 * summary.get("calibrated_frame_fraction", 0):.0f}% кадров</span>
<span>Калибровка: {escape(str(summary.get("calibration", "")))}</span></div>
<h2>Видео</h2><video controls src="annotated.mp4"></video>
<h2>Команды</h2>{team_table}
<h2>Статистика игроков</h2>{player_table}
<p class="muted">ПОП/БР — попадания/броски с игры, ПН/ПЗ — подборы в нападении/защите, АС — передачи, ПХ — перехваты,
ПОТ — потери, БЛ — блок-шоты. Игрок без номера показан как отдельный трек.</p>
<div class="grid"><div><h2>Карта бросков</h2><img src="shot_chart.png" alt="Карта бросков">
<p class="muted">Круг — попадание, крест — промах; цвет — команда.</p></div>
<div><h2>Тепловая карта позиций</h2><img src="heatmap.png" alt="Тепловая карта"></div></div>
<h2>События</h2>{event_table}
<h2>Качество и ограничения</h2><p class="note">Результаты формируются автоматически. При необходимости
результаты сформированы автоматически; ручная разметка используется отдельно только для оценки качества.</p><ul>{warnings}</ul>
<p class="muted">Средняя ошибка гомографии: {quality.get("mean_reprojection_error_m", "—")} м ·
команды: {escape(str(summary.get("team_embedder", "")))}, названия — {escape(str(summary.get("team_mapping", "")))}</p>
</main></html>"""
    (folder / f"report{suffix}.html").write_text(html, encoding="utf-8")


def archive_result(folder):
    folder = Path(folder)
    target = folder / "results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix != ".zip" and path.name != "raw.mp4":
                archive.write(path, path.name)
    return str(target)


def _optional_int(value):
    if value is None or str(value).strip() in ("", "nan", "None"):
        return None
    return int(float(value))


def save_review(folder, identity_rows, event_rows, score_override=None):
    """Сохраняет исправления отдельно от автоматических файлов и пересчитывает box score.

    identity_rows: [track_id, команда (0/1/пусто), номер, имя];
    event_rows: строки с колонками REVIEW_EVENT_COLUMNS (можно добавлять новые).
    """
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    team_names = config.get("team_names", ["Команда 1", "Команда 2"])
    rosters = [dict(r) for r in config.get("rosters", [{}, {}])]
    tracks = {int(r["track_id"]): r for r in read_csv(folder / "tracks_summary.csv")}
    for row in tracks.values():
        row["team"] = _optional_int(row.get("team"))
    for row in identity_rows or []:
        if not row or row[0] in (None, ""):
            continue
        track = int(float(row[0]))
        if track not in tracks:
            raise ValueError(f"Неизвестный трек {track}.")
        team = _optional_int(row[1])
        if team not in (None, 0, 1):
            raise ValueError("Команда: 0, 1 или пусто.")
        number = str(row[2] or "").strip().removesuffix(".0")
        tracks[track]["team"], tracks[track]["number"] = team, number
        name = str(row[3] or "").strip()
        if name and team in (0, 1) and number:
            rosters[team][number] = name  # имя из проверки важнее имени из состава
    original = {int(e["event_id"]): e for e in json.loads((folder / "events.json").read_text(encoding="utf-8"))}
    events = []
    for position, row in enumerate(event_rows or [], 1):
        if not row or all(v is None or str(v).strip() == "" for v in row):
            continue
        event_id, t, kind, player, team, status, outcome, points, reason = row
        if kind not in EVENT_NAMES or status not in STATUSES:
            raise ValueError(f"Неизвестный тип события или статус: {kind}, {status}.")
        player, team, points = map(_optional_int, (player, team, points))
        if not math.isfinite(float(t)) or float(t) < 0 or team not in (None, 0, 1) or points not in (None, 0, 1, 2, 3):
            raise ValueError("Некорректные время, команда или очки.")
        if player is not None and player not in tracks:
            raise ValueError(f"У события указан неизвестный трек {player}.")
        base = original.get(_optional_int(event_id), {})
        events.append(asdict(Event(
            position, float(t), kind, player, other_player_id=base.get("other_player_id"), team=team,
            confidence=base.get("confidence", 0.0), status=status, outcome=str(outcome or ""), points=points,
            x_m=base.get("x_m"), y_m=base.get("y_m"), reason=str(reason or ""),
        )))
    players, teams, _ = build_boxscore(list(tracks.values()), events, team_names, rosters)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    summary["score"] = [teams[0]["PTS"], teams[1]["PTS"]]
    summary["score_source"] = "попадания после коррекции"
    if score_override is not None:
        if len(score_override) != 2 or any(not math.isfinite(v) or int(v) != v or v < 0 for v in score_override):
            raise ValueError("Исправленный счёт должен состоять из двух неотрицательных целых чисел.")
        summary["score"] = [int(v) for v in score_override]
        summary["score_source"] = "введён вручную при коррекции"
    summary["reviewed"] = True
    summary["player_count"] = len(players)
    write_csv(folder / "players_reviewed.csv", players, PLAYER_FIELDS)
    write_csv(folder / "teams_reviewed.csv", teams, TEAM_FIELDS)
    write_csv(folder / "events_reviewed.csv", events, EVENT_FIELDS)
    write_json(folder / "summary_reviewed.json", summary)
    make_report(folder, summary, players, teams, events, "_reviewed")
    return archive_result(folder)
