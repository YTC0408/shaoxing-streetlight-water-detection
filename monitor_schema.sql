-- monitor.py SQLite schema
-- Two tables: per-tick observations + fault events (UPSERT on start_ts).

CREATE TABLE IF NOT EXISTS observations (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  ts            TEXT    NOT NULL,                -- ISO 8601 local time
  light_id      TEXT    NOT NULL,
  cls           INTEGER,                          -- 0=light_damage, 1=light_on, NULL=miss/uncertain
  score         REAL,
  is_night      INTEGER NOT NULL,                -- 0/1
  truncated     INTEGER NOT NULL,                -- 0/1
  state         TEXT    NOT NULL,                -- normal/candidate_fault/candidate_daylight/fault/daylight_abnormal
  state_change  TEXT                             -- enter_fault/enter_daylight/clear/tick/NULL
);
CREATE INDEX IF NOT EXISTS obs_light_ts ON observations(light_id, ts);

CREATE TABLE IF NOT EXISTS faults (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  light_id    TEXT NOT NULL,
  start_ts    TEXT NOT NULL,
  end_ts      TEXT,                              -- NULL = ongoing
  fault_kind  TEXT NOT NULL                      -- 'night_damage' | 'day_light_on'
);
CREATE INDEX IF NOT EXISTS faults_open ON faults(light_id, start_ts);
CREATE INDEX IF NOT EXISTS faults_light ON faults(light_id, start_ts);
