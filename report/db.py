"""SQLite store for Image Compare investigate reports.

A `run` is one comparison (one Image Compare execution); a `result` is one image in it
with its caption / prompt / settings / verdict. Settings are stored both as a JSON blob
(for display) and expanded into a key/value table `result_settings` (EAV) so the report
page can filter / sort / aggregate by an arbitrary setting field — new, never-seen fields
just become new rows, no schema change needed.

Only stdlib (`sqlite3`). Writes happen exclusively from the "Save run to report" button on
the served comparison page (via the /kinburg/report/save route); nodes never touch the DB.
"""
import os
import sqlite3
import json
import time

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_key TEXT UNIQUE NOT NULL,
  created_at TEXT, title TEXT, workflow TEXT, html_url TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  idx INTEGER, image_path TEXT, caption TEXT, prompt TEXT,
  settings_text TEXT, settings_json TEXT,
  status TEXT, rating INTEGER, comment TEXT, created_at TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS result_settings (
  result_id INTEGER NOT NULL, key TEXT, value TEXT,
  FOREIGN KEY(result_id) REFERENCES results(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS result_tags (
  result_id INTEGER NOT NULL, tag TEXT,
  FOREIGN KEY(result_id) REFERENCES results(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS ix_rs_key ON result_settings(key);
CREATE INDEX IF NOT EXISTS ix_rs_kv  ON result_settings(key, value);
CREATE INDEX IF NOT EXISTS ix_rt_tag ON result_tags(tag);
"""


def _connect(db_path):
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn


def _field_pairs(fields):
    """Normalize settings_fields into (key, value) pairs. Accepts a list of
    {key|label, value} dicts or a plain {key: value} dict."""
    if isinstance(fields, dict):
        return [(str(k), _stringify(v)) for k, v in fields.items()]
    out = []
    for f in (fields or []):
        if isinstance(f, dict):
            k = f.get("key", f.get("label", ""))
            out.append((str(k), _stringify(f.get("value", ""))))
    return out


def _stringify(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


def upsert_run(db_path, run, results):
    """Insert or replace a run (keyed by run_key/run_id) and all of its results."""
    run_key = str(run.get("run_id") or run.get("run_key") or "").strip()
    if not run_key:
        raise ValueError("run_id (run_key) is required")
    now = run.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S")

    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT id FROM runs WHERE run_key = ?", (run_key,)).fetchone()
        if row:
            run_id = row[0]
            cur.execute("DELETE FROM results WHERE run_id = ?", (run_id,))  # cascades to settings/tags
            cur.execute("UPDATE runs SET created_at=?, title=?, workflow=?, html_url=?, notes=? WHERE id=?",
                        (now, run.get("title", ""), run.get("workflow", ""),
                         run.get("html_url", ""), run.get("notes", ""), run_id))
        else:
            cur.execute("INSERT INTO runs(run_key, created_at, title, workflow, html_url, notes) VALUES (?,?,?,?,?,?)",
                        (run_key, now, run.get("title", ""), run.get("workflow", ""),
                         run.get("html_url", ""), run.get("notes", "")))
            run_id = cur.lastrowid

        for r in results:
            fields = r.get("settings_fields")
            cur.execute(
                "INSERT INTO results(run_id, idx, image_path, caption, prompt, settings_text, settings_json, status, rating, comment, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, r.get("idx"), r.get("image_path", ""), r.get("caption", ""), r.get("prompt", ""),
                 r.get("settings", ""), json.dumps(fields, ensure_ascii=False) if fields is not None else "",
                 r.get("status", "keep"), int(r.get("rating") or 0), r.get("comment", ""), now))
            rid = cur.lastrowid
            for k, v in _field_pairs(fields):
                cur.execute("INSERT INTO result_settings(result_id, key, value) VALUES (?,?,?)", (rid, k, v))
            for t in (r.get("tags") or []):
                cur.execute("INSERT INTO result_tags(result_id, tag) VALUES (?,?)", (rid, str(t)))

        conn.commit()
        return {"run_id": run_id, "run_key": run_key, "results": len(results)}
    finally:
        conn.close()


def update_result(db_path, result_id, status=None, rating=None, comment=None, tags=None):
    """Patch a single result's verdict fields (only the provided ones)."""
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        sets, params = [], []
        if status is not None:
            sets.append("status = ?"); params.append(str(status))
        if rating is not None:
            sets.append("rating = ?"); params.append(int(rating or 0))
        if comment is not None:
            sets.append("comment = ?"); params.append(str(comment))
        if sets:
            params.append(int(result_id))
            cur.execute(f"UPDATE results SET {', '.join(sets)} WHERE id = ?", params)
        if tags is not None:
            cur.execute("DELETE FROM result_tags WHERE result_id = ?", (int(result_id),))
            for t in tags:
                cur.execute("INSERT INTO result_tags(result_id, tag) VALUES (?,?)", (int(result_id), str(t)))
        conn.commit()
        return {"updated": int(result_id)}
    finally:
        conn.close()


def delete_run(db_path, run_key, remove_images=True):
    """Delete a run (cascades to its results/settings/tags) and, optionally, its image files."""
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT id FROM runs WHERE run_key = ?", (str(run_key),)).fetchone()
        if not row:
            return {"deleted": 0, "images_removed": 0}
        run_id = row[0]
        imgs = [p for (p,) in cur.execute("SELECT image_path FROM results WHERE run_id = ?", (run_id,)).fetchall()]
        cur.execute("DELETE FROM runs WHERE id = ?", (run_id,))  # cascades
        conn.commit()
    finally:
        conn.close()

    removed, dirs = 0, set()
    if remove_images:
        for p in imgs:
            try:
                if p and os.path.isfile(p):
                    os.remove(p); removed += 1; dirs.add(os.path.dirname(p))
            except Exception:
                pass
        for d in dirs:
            try:
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            except Exception:
                pass
    return {"deleted": 1, "images_removed": removed}


def fetch_results(db_path):
    """Return every result joined with its run, settings (as a dict) and tags, plus the
    distinct setting keys / tags / runs for the report page's filter controls."""
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT r.id, r.idx, r.image_path, r.caption, r.prompt, r.settings_text,"
            "       r.status, r.rating, r.comment, r.created_at,"
            "       ru.run_key, ru.title, ru.html_url, ru.created_at"
            " FROM results r JOIN runs ru ON ru.id = r.run_id"
            " ORDER BY ru.created_at DESC, r.idx ASC").fetchall()
        results = []
        for row in rows:
            rid = row[0]
            settings = {k: v for (k, v) in
                        cur.execute("SELECT key, value FROM result_settings WHERE result_id=?", (rid,)).fetchall()}
            tags = [t for (t,) in cur.execute("SELECT tag FROM result_tags WHERE result_id=?", (rid,)).fetchall()]
            results.append({
                "id": rid, "idx": row[1], "image_path": row[2], "caption": row[3], "prompt": row[4],
                "settings_text": row[5], "status": row[6], "rating": row[7], "comment": row[8],
                "created_at": row[9], "run_key": row[10], "run_title": row[11], "run_url": row[12],
                "run_created": row[13], "settings": settings, "tags": tags,
            })
        keys = [k for (k,) in cur.execute("SELECT DISTINCT key FROM result_settings ORDER BY key").fetchall()]
        tags_all = [t for (t,) in cur.execute("SELECT DISTINCT tag FROM result_tags ORDER BY tag").fetchall()]
        runs = [{"run_key": rk, "title": ti, "created_at": ca} for (rk, ti, ca) in
                cur.execute("SELECT run_key, title, created_at FROM runs ORDER BY created_at DESC").fetchall()]
        return {"results": results, "keys": keys, "tags": tags_all, "runs": runs}
    finally:
        conn.close()
