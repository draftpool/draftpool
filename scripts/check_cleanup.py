"""Check that cleanup removes exactly the damage and nothing else.

Run it against a *copy* of a data directory, never the live one:

    cp -R data /tmp/check && \
      DRAFTPOOL_DATA_DIR=/tmp/check uv run uvicorn app.main:app --port 8579 &
    uv run python scripts/check_cleanup.py /tmp/check http://127.0.0.1:8579

It snapshots every row and hashes every file, calls the real endpoints over
HTTP, and then proves the diff: every removed row was finished and had no
image, the removals match what the run reported, no surviving file changed,
nothing was added, and a second run finds nothing. Exit code 1 on any failure.
"""

import hashlib, json, sqlite3, sys, urllib.request
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
DATA = Path(sys.argv[1])
BASE = sys.argv[2]

def api(path, method="GET"):
    req = urllib.request.Request(BASE + path, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "detail": json.load(e)}

def snapshot():
    conn = sqlite3.connect(DATA / "draftpool.db"); conn.row_factory = sqlite3.Row
    rows = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM generations")}
    conn.close()
    files = {str(p.relative_to(DATA / "sessions")): hashlib.md5(p.read_bytes()).hexdigest()
             for p in (DATA / "sessions").glob("*/*") if p.is_file()}
    return rows, files

rows0, files0 = snapshot()
print(f"before: {len(rows0)} rows, {len(files0)} files")

survey = api("/api/cleanup")
if survey.get("blocked"):
    print("BLOCKED:", survey["blocked"]); sys.exit(1)
print(f"survey: {len(survey['missing'])} rows, "
      f"{len(survey['files'])} orphan files, blocked={survey['blocked']}")
for m in survey["missing"]:
    print(f"   row  {m['id']}  {m['status']:5}  {m['kind']:7}  {m['created_at']}  files={m['files']}")
for f in survey["files"]:
    print(f"   file {f}")

done = api("/api/cleanup", "POST")
print(f"removed: {len(done['rows'])} rows, {len(done['files'])} files")

rows1, files1 = snapshot()
gone_rows = set(rows0) - set(rows1)
gone_files = set(files0) - set(files1)
changed = [f for f in files0.keys() & files1.keys() if files0[f] != files1[f]]

ok = True
def check(label, cond, detail=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))
    ok = ok and cond

print("\nchecks")
check("rows removed == rows the survey listed",
      gone_rows == {m["id"] for m in survey["missing"]}, f"{sorted(gone_rows)}")
check("rows removed == rows the run reported", gone_rows == set(done["rows"]))
check("every removed row was finished (done/error)",
      all(rows0[r] in ("done", "error") for r in gone_rows),
      str({r: rows0[r] for r in gone_rows}))
check("no removed row had an image on disk",
      not [r for r in gone_rows if f"{r}.png" in {f.split('/')[-1] for f in files0}])
check("files removed == files the run reported", gone_files == set(done["files"]))
check("no surviving file was modified", not changed, str(changed))
check("no row was added", not set(rows1) - set(rows0))
check("no file was added", not set(files1) - set(files0))
check("everything else survived",
      set(rows1) == set(rows0) - gone_rows and set(files1) == set(files0) - gone_files)

print(f"\nafter: {len(rows1)} rows, {len(files1)} files")
print("\nsecond run is a no-op")
again = api("/api/cleanup")
check("nothing left to find", not again["missing"] and not again["files"])
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
