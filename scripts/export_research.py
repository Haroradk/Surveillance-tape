"""Publish the signal-research results to MotherDuck for the dashboard's Research tab.

Runs the study against the LOCAL database (61 backfilled days) and writes a frozen,
dated snapshot to gold.signal_research on MotherDuck. Deliberately manual, not part of
the daily job: re-running on growing data would quietly move the discovery/holdout split.

  python scripts/export_research.py        # needs MOTHERDUCK_TOKEN in .env for the write
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import signal_research  # noqa: E402

local = duckdb.connect(config.DB_PATH, read_only=True)
local.execute("SET TimeZone = 'UTC'")
snapshot, meta = signal_research.tidy(local)
local.close()
print(meta, len(snapshot), "rows")

md = config.get_connection()
md.execute("CREATE SCHEMA IF NOT EXISTS gold")
md.execute("""CREATE TABLE IF NOT EXISTS gold.signal_research (
    version TIMESTAMP, question VARCHAR, rule VARCHAR, dataset VARCHAR, horizon_min INTEGER, n INTEGER,
    value DOUBLE, control DOUBLE, ci_lo DOUBLE, ci_hi DOUBLE, hit_rate DOUBLE,
    first_day DATE, last_day DATE, n_days INTEGER, split_day DATE, n_alerts INTEGER, cost_pct DOUBLE)""")
snapshot["version"] = __import__("pandas").Timestamp.utcnow().tz_localize(None)
for k, v in meta.items():
    snapshot[k] = v
md.execute("INSERT INTO gold.signal_research BY NAME SELECT * FROM snapshot")
print("published", md.execute("SELECT max(version), count(*) FROM gold.signal_research").fetchone())
