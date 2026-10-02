"""SQLite-backed session store: sessions, generations, lineage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import DATA_DIR, dims
from .schemas import Generation, SessionDetail, SessionSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    instruction TEXT NOT NULL,
    -- the composer as it stood: the enriched prompts, and which models they
    -- were written for. A session whose prompts were written but never
    -- generated from has this and no generations at all, which is the whole
    -- reason it is stored: the LLM round it took to get here is not free.
    prompts TEXT,
    -- what the user called the session. Separate from instruction, which is
    -- the composer's text and is rewritten by every Enrich and Generate; NULL
    -- means the picker shows the instruction, as it always did
    name TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    parent_id TEXT REFERENCES generations(id),
    -- the second picture a Combine took its subject from; parent_id is the
    -- scene. Treated as a parent link wherever drafts are removed.
    source_id TEXT REFERENCES generations(id),
    kind TEXT NOT NULL,
    model_key TEXT NOT NULL,
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL DEFAULT '',
    seed INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    aspect TEXT,
    steps INTEGER NOT NULL,
    cfg REAL NOT NULL,
    sampler TEXT,
    -- JSON list of {file, weight}: every LoRA this was rendered with; NULL on
    -- rows from before it was recorded (see Generation.loras)
    loras TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    strength REAL,
    describe_first INTEGER NOT NULL DEFAULT 0,
    region TEXT,
    duration_s REAL,
    variant_label TEXT,
    batch TEXT,
    published_at TEXT,
    published_post_id INTEGER,
    -- set when Remove drafts took this draft's image but had to keep the row,
    -- because something still names it as parent_id; see prune_drafts
    pruned_at TEXT,
    created_at TEXT NOT NULL
);
-- Indices follow the queries, which are few and fixed. Checked with EXPLAIN
-- QUERY PLAN; without these, four of them are full table scans, and three of
-- those run on every server start or page load.
--
-- session_id: loading a session and its stats. One column is enough for both,
-- and it also sorts the load for free -- an index entry carries the rowid as
-- its trailing key, so reading this index for one session yields the rows in
-- rowid order, which is the order get_session asks for. An index that also
-- carried created_at and id sorted the wrong thing (see get_session) and cost
-- two more columns on every write to do it.
CREATE INDEX IF NOT EXISTS idx_generations_session_order
    ON generations(session_id);
-- status: resuming unfinished work at startup, finding clips still rendering at
-- shutdown, and the running -> pending sweep. All three scanned the whole table
-- to find a handful of rows, and that cost grows with every image ever made.
CREATE INDEX IF NOT EXISTS idx_generations_status
    ON generations(status, created_at, id);
-- the session picker, on every page load
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
DROP INDEX IF EXISTS idx_generations_session;
DROP INDEX IF EXISTS idx_generations_session_created;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _middle(values: list[float]) -> float:
    """Median, low side on an even count. Used on samples of one and up."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


# The kinds render_timings measures: the ones whose cost is fixed by model,
# size and steps. See its docstring for the ones left out and why.
TIMED_KINDS = ("draft", "final", "faithful", "upscale")
# Renders per group the estimate is taken over, newest first
TIMING_WINDOW = 30
# Fewer than this and the group is not reported at all
TIMING_MIN_SAMPLES = 3


def timing_size(width: int, height: int) -> int:
    """The stage size a render is filed under: dims.stage_size, by the name the
    timing rows and static/index.html (stageSize) know it by."""
    return dims.stage_size(width, height)


class StorageMisconfigured(RuntimeError):
    """The image folder does not look like the one these images were written to."""


class Store:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = data_dir / "draftpool.db"
        # one-time migration from the project's old name: an existing install
        # must keep its sessions and generations, not wake up to an empty db
        old_path = data_dir / "imagegen.db"
        if old_path.is_file() and not self._db_path.is_file():
            old_path.rename(self._db_path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # migrations for databases created before these columns existed
            for ddl in (
                "ALTER TABLE generations ADD COLUMN strength REAL",
                "ALTER TABLE generations ADD COLUMN describe_first INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE generations ADD COLUMN region TEXT",
                "ALTER TABLE generations ADD COLUMN duration_s REAL",
                "ALTER TABLE generations ADD COLUMN num_frames INTEGER",
                "ALTER TABLE generations ADD COLUMN fps INTEGER",
                "ALTER TABLE generations ADD COLUMN video_ext TEXT",
                "ALTER TABLE generations ADD COLUMN aspect TEXT",
                "ALTER TABLE generations ADD COLUMN variant_label TEXT",
                "ALTER TABLE generations ADD COLUMN batch TEXT",
                "ALTER TABLE generations ADD COLUMN published_at TEXT",
                "ALTER TABLE generations ADD COLUMN published_post_id INTEGER",
                "ALTER TABLE generations ADD COLUMN pruned_at TEXT",
                "ALTER TABLE generations ADD COLUMN source_id TEXT",
                "ALTER TABLE generations ADD COLUMN loras TEXT",
                "ALTER TABLE sessions ADD COLUMN prompts TEXT",
                "ALTER TABLE sessions ADD COLUMN name TEXT",
            ):
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError:
                    pass

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- sessions ---------------------------------------------------------

    def create_session(self, instruction: str) -> SessionSummary:
        sid = _new_id()
        created = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, instruction, created_at) VALUES (?, ?, ?)",
                (sid, instruction, created),
            )
        (self.data_dir / "sessions" / sid).mkdir(parents=True, exist_ok=True)
        return SessionSummary(id=sid, instruction=instruction, created_at=created)

    def save_prompts(self, session_id: str, instruction: str, prompts: dict) -> bool:
        """Store the composer state, and the instruction it was written from.

        The instruction goes along because it is what names the session in the
        picker, and it is routinely edited after the row was created.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET instruction = ?, prompts = ? WHERE id = ?",
                (instruction, json.dumps(prompts), session_id),
            )
        return cur.rowcount > 0

    def rename_session(self, session_id: str, name: Optional[str]) -> bool:
        """Set the session's own name; blank clears it back to the instruction."""
        name = (name or "").strip() or None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET name = ? WHERE id = ?", (name, session_id)
            )
        return cur.rowcount > 0

    def list_sessions(self) -> list[SessionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [SessionSummary(**dict(r)) for r in rows]

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            # Insertion order, which is the order the browser showed while
            # the work was being done. created_at cannot do it: it is stored to
            # the second, so one press of Generate -- a dozen rows queued in
            # well under a second -- ties on it, and the id tiebreak is a random
            # uuid. Reopening the session dealt those cards out shuffled.
            gens = conn.execute(
                "SELECT * FROM generations WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        fields = dict(row)
        # stored as text, handed to the client as the object it was
        raw = fields.pop("prompts", None)
        try:
            prompts = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            prompts = None
        return SessionDetail(
            **fields,
            prompts=prompts,
            generations=[Generation(**dict(g)) for g in gens],
        )

    # -- generations ------------------------------------------------------

    def create_generation(
        self,
        *,
        session_id: str,
        kind: str,
        model_key: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        aspect: Optional[str] = None,
        cfg: float,
        sampler: Optional[str],
        loras: Optional[list] = None,
        parent_id: Optional[str] = None,
        source_id: Optional[str] = None,
        strength: Optional[float] = None,
        describe_first: bool = False,
        region: Optional[str] = None,
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        variant_label: Optional[str] = None,
        batch: Optional[str] = None,
    ) -> Generation:
        gen = Generation(
            id=_new_id(),
            session_id=session_id,
            parent_id=parent_id,
            source_id=source_id,
            kind=kind,  # type: ignore[arg-type]
            model_key=model_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            aspect=aspect,
            steps=steps,
            cfg=cfg,
            sampler=sampler,
            loras=loras,
            status="pending",
            strength=strength,
            describe_first=describe_first,
            region=region,
            num_frames=num_frames,
            fps=fps,
            variant_label=variant_label,
            batch=batch,
            created_at=_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO generations
                   (id, session_id, parent_id, source_id, kind, model_key, prompt,
                    negative_prompt, seed, width, height, aspect, steps, cfg, sampler,
                    loras, status, strength, describe_first, region, num_frames, fps,
                    variant_label, batch, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    gen.id, gen.session_id, gen.parent_id, gen.source_id, gen.kind, gen.model_key,
                    gen.prompt, gen.negative_prompt, gen.seed, gen.width, gen.height,
                    gen.aspect, gen.steps, gen.cfg, gen.sampler,
                    None if gen.loras is None
                    else json.dumps([l.model_dump() for l in gen.loras]),
                    gen.status, gen.strength,
                    int(gen.describe_first), gen.region, gen.num_frames, gen.fps,
                    gen.variant_label, gen.batch, gen.created_at,
                ),
            )
        return gen

    def update_prompts(self, gen_id: str, prompt: str, negative_prompt: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET prompt = ?, negative_prompt = ? WHERE id = ?",
                (prompt, negative_prompt, gen_id),
            )

    def mark_published(self, gen_id: str, post_id: int) -> str:
        """Record that this image now has a Civitai post, and when. Returns the time.

        Written after Civitai has answered with a post id, so the row can only
        say "published" about a post that exists. The id is kept as well as the
        timestamp because it is what links back to the post, and because
        re-publishing the same image simply overwrites both -- the card then
        points at the newer post, which is the one the user is looking at.
        """
        stamp = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET published_at = ?, published_post_id = ? WHERE id = ?",
                (stamp, int(post_id), gen_id),
            )
        return stamp

    def get_generation(self, gen_id: str) -> Optional[Generation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM generations WHERE id = ?", (gen_id,)
            ).fetchone()
        return Generation(**dict(row)) if row else None

    def unfinished_generations(self) -> list[str]:
        """Pending/running generation ids, oldest first: work lost by a restart."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET status = 'pending' WHERE status = 'running'"
            )
            rows = conn.execute(
                "SELECT id FROM generations WHERE status = 'pending' "
                "ORDER BY created_at, id"
            ).fetchall()
        return [r["id"] for r in rows]

    def session_stats(self, session_id: str) -> dict:
        """Counts and time for one session, by kind and by model.

        Grouped in SQL rather than counted in the browser so the numbers stay
        right for a session too big to hold on screen, and so a page that has
        only loaded part of the history cannot report a smaller total than the
        one on disk. Failed and queued rows are counted separately: an error is
        not an image, but pretending it never happened hides where time went.
        """
        with self._connect() as conn:
            rows = conn.execute(
                # A 2x fast row stores the parent's model_key as provenance --
                # what made the picture it sharpened -- but Real-ESRGAN is a
                # separate binary that queues nothing against Draw Things and
                # touches no checkpoint (see app/esrgan.py). Crediting that
                # checkpoint with the image, and with the seconds, is a lie the
                # per-model table cannot afford. The stored row keeps its
                # provenance; only the attribution is corrected on read.
                "SELECT kind, status, COUNT(*) n, "
                # aliased away from the real column: grouping by a bare
                # `model_key` here would be ambiguous between the two
                "       CASE WHEN kind = 'esrgan' THEN 'real-esrgan' "
                "            ELSE model_key END attributed, "
                "       COALESCE(SUM(duration_s), 0) secs, "
                # rows that predate duration tracking, or whose job was
                # recovered rather than timed: counted, but they make `seconds`
                # a lower bound, so the caller has to be able to say so
                "       SUM(duration_s IS NULL) untimed "
                # a pruned draft has left the page just as a deleted one has,
                # and the numbers should not count it for one and not the other
                "FROM generations WHERE session_id = ? AND pruned_at IS NULL "
                "GROUP BY kind, attributed, status",
                (session_id,),
            ).fetchall()
        kinds: dict[str, int] = {}
        models: dict[str, dict] = {}
        done = failed = pending = untimed = 0
        seconds = 0.0
        for r in rows:
            n, secs, status = r["n"], r["secs"] or 0.0, r["status"]
            if status == "error":
                failed += n
                seconds += secs
                continue
            if status != "done":
                pending += n
                continue
            done += n
            seconds += secs
            untimed += r["untimed"] or 0
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + n
            key = r["attributed"]
            m = models.setdefault(
                key, {"key": key, "total": 0, "seconds": 0.0, "kinds": {}}
            )
            m["total"] += n
            m["seconds"] += secs
            m["kinds"][r["kind"]] = m["kinds"].get(r["kind"], 0) + n
        return {
            "done": done,
            "failed": failed,
            "pending": pending,
            "seconds": round(seconds),
            "untimed": untimed,
            "kinds": kinds,
            "models": [
                {**m, "seconds": round(m["seconds"])}
                for m in sorted(models.values(), key=lambda m: (-m["total"], m["key"]))
            ],
        }

    def render_rate(self, model_key: str, long_edge: int) -> Optional[float]:
        """Seconds per unit of work this model has actually taken, or None.

        A unit is one step at full strength: img2img skips the early part of
        the schedule in proportion to strength, so a 0.5-strength pass at 8
        steps does the work of 4. Rows at `long_edge` are used when there are
        any, because cost per unit does not carry across sizes -- qwen-edit
        measured 20 s/unit at 1024 and 72 at its tiled 2048 upscale -- and the
        slowest size seen is the fallback when this one is new.

        Used only to size a request timeout, never shown as an estimate, so
        the middle value is taken rather than the mean or the max: a job held
        while Draw Things was restarted has the hold in its duration, and one
        such row would otherwise set the timeout for every job after it.

        Detail and 2x fast rows are excluded. Their duration covers a handful
        of crop renders rather than one image, so their seconds per unit
        describe a different shape of work.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT width, height, steps, strength, duration_s FROM generations "
                "WHERE model_key = ? AND status = 'done' AND duration_s IS NOT NULL "
                "AND steps > 0 AND kind NOT IN ('detail', 'esrgan', 'video') "
                "ORDER BY rowid DESC LIMIT 60",
                (model_key,),
            ).fetchall()
        by_size: dict[int, list[float]] = {}
        for r in rows:
            work = r["steps"] * (r["strength"] if r["strength"] is not None else 1.0)
            if work <= 0:
                continue
            by_size.setdefault(max(r["width"], r["height"]), []).append(
                r["duration_s"] / work
            )
        if not by_size:
            return None
        rates = by_size.get(long_edge) or max(by_size.values(), key=_middle)
        return _middle(rates)

    def render_timings(self) -> list[dict]:
        """Measured seconds per step, by model, kind and stage size.

        What the UI shows in place of a guess before a job is queued: the
        composer's total for a round of drafts, a model chip's tooltip, the
        wait named on Finalize. Separate from render_rate, which sizes a
        timeout and wants something different out of the same rows.

        Grouped by kind as well as model, because a kind is a shape of work and
        not just a label: a Faithful is img2img at the model's final strength,
        an Upscale runs an upscaler before its pass, and a draft is neither.
        Within one kind the strength is the model's configured one, so seconds
        per *step* is the whole normalisation -- dividing by strength as well
        would need the value the job actually ran at, and a Faithful's row
        stores None for it.

        Keyed by stage size (see timing_size) rather than long edge or aspect:
        a ratio holds the pixel budget, so a 3:4 draft costs what a square one
        does and would only split the sample if it were a group of its own.

        Only kinds whose cost is fixed by model, size and steps are kept.
        Edits, faces and combines depend on the rectangle and the number of
        faces; a Finalize+ carries an LLM round in its duration. The median of
        the most recent TIMING_WINDOW rows, and not the mean, so the one job
        that sat through a Draw Things restart does not become the estimate,
        and so a faster machine or a new Draw Things shows through within a
        few dozen renders. A group below TIMING_MIN_SAMPLES is left out: one
        render includes loading the checkpoint as often as not, and a number
        shown as measured should not be that.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_key, kind, width, height, steps, duration_s "
                "FROM generations WHERE status = 'done' AND duration_s > 0 "
                "AND steps > 0 AND describe_first = 0 "
                f"AND kind IN ({','.join('?' * len(TIMED_KINDS))}) "
                "ORDER BY rowid DESC",
                TIMED_KINDS,
            ).fetchall()
        groups: dict[tuple[str, str, int], list[float]] = {}
        for r in rows:
            key = (r["model_key"], r["kind"], timing_size(r["width"], r["height"]))
            got = groups.setdefault(key, [])
            if len(got) < TIMING_WINDOW:
                got.append(r["duration_s"] / r["steps"])
        return [
            {
                "model_key": model_key,
                "kind": kind,
                "size": size,
                "sec_per_step": round(_middle(rates), 3),
                "n": len(rates),
            }
            for (model_key, kind, size), rates in sorted(groups.items())
            if len(rates) >= TIMING_MIN_SAMPLES
        ]

    def running_video_generations(self) -> list[str]:
        """Ids of video jobs rendering right now, oldest first.

        Read at shutdown to warn about work this process cannot stop: Draw
        Things has no cancel endpoint, so a clip in flight keeps rendering
        after we are gone. See docs/decisions/VIDEO.md.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM generations WHERE status = 'running' "
                "AND kind = 'video' ORDER BY created_at, id"
            ).fetchall()
        return [r["id"] for r in rows]

    def set_status(
        self,
        gen_id: str,
        status: str,
        error: Optional[str] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET status = ?, error = ?, duration_s = ? WHERE id = ?",
                (status, error, duration_s, gen_id),
            )

    def delete_generation(self, gen_id: str) -> None:
        gen = self.get_generation(gen_id)
        if gen is None:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM generations WHERE id = ?", (gen_id,))
        self.image_path(gen).unlink(missing_ok=True)
        self.init_path(gen).unlink(missing_ok=True)
        # a video generation owns a clip and a poster frame too; both containers
        # are tried because video_ext records which one actually got written
        for ext in ("mp4", "webp"):
            (self.data_dir / "sessions" / gen.session_id / f"{gen.id}.{ext}").unlink(
                missing_ok=True
            )
        # A pruned draft is kept only as the parent of something. When the last
        # such thing goes, the stub has no card and no reason to stay.
        for up_id in dict.fromkeys(filter(None, (gen.parent_id, gen.source_id))):
            up = self.get_generation(up_id)
            if up is not None and up.pruned_at and not self._has_children(up.id):
                self.delete_generation(up.id)

    def _has_children(self, gen_id: str) -> bool:
        # a Combine's subject is as much its origin as its scene is
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM generations WHERE parent_id = ? OR source_id = ? LIMIT 1",
                (gen_id, gen_id),
            ).fetchone() is not None

    # -- removing drafts ----------------------------------------------------

    def survey_drafts(self, session_id: str) -> dict:
        """What Remove drafts would do, without doing it.

        A draft is worth nothing once it has been looked at, except as a link:
        a final, a faithful or a More-like-this draft names it as parent_id,
        and that id is what the "from" link, the lineage filter, the params
        chain and a Civitai post's settings all follow. So there are three
        outcomes, not one:

        `deleted`: the row and its image both go. Only a draft whose every
        descendant is going too -- a leaf, or a chain of drafts ending in one.

        `pruned`: the image goes and the row stays, marked pruned_at. It has no
        card any more, but the id still resolves to the prompt, seed and model
        it was made with, so the lineage above the images that remain is whole.

        `kept`: left alone. A draft with a queued or running child, because
        that job reads the draft's image when it starts. Nothing else is kept:
        drafts are not publishable, so a published_at on one is no reason to.

        Always one session: drafts are picked from a session at a time, and
        which of them are done with is a judgement about that session, never
        about the install. Only finished drafts are considered: a queued draft
        has not been seen yet. A row already pruned is a candidate again, and is deleted once
        whatever it was kept for has itself been deleted.
        """
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id, session_id, parent_id, source_id, kind, status, "
                "pruned_at FROM generations"
            )]
        # a Combine names two origins, its scene (parent_id) and its subject
        # (source_id), and either one being a draft holds that draft the same way
        children: dict[str, list[dict]] = {}
        for r in rows:
            for up in dict.fromkeys(filter(None, (r["parent_id"], r["source_id"]))):
                children.setdefault(up, []).append(r)
        kept, candidates = [], {}
        for r in rows:
            if r["kind"] != "draft" or r["status"] not in self.FINISHED:
                continue
            if r["session_id"] != session_id:
                continue
            if any(c["status"] not in self.FINISHED for c in children.get(r["id"], [])):
                kept.append({"id": r["id"], "reason": "a render from it is still queued"})
            else:
                candidates[r["id"]] = r
        # bottom up: a draft can go once everything made from it is going
        deleted: set[str] = set()
        grew = True
        while grew:
            grew = False
            for gen_id in candidates:
                if gen_id not in deleted and all(
                    c["id"] in deleted for c in children.get(gen_id, [])
                ):
                    deleted.add(gen_id)
                    grew = True
        pruned = [
            gen_id for gen_id, r in candidates.items()
            if gen_id not in deleted and not r["pruned_at"]
        ]
        files = []
        for gen_id in [*deleted, *pruned]:
            folder = self.data_dir / "sessions" / candidates[gen_id]["session_id"]
            files += [
                p for p in (folder / f"{gen_id}.png", folder / f"{gen_id}_init.png")
                if p.is_file()
            ]
        order = {r["id"]: i for i, r in enumerate(rows)}
        return {
            "deleted": sorted(deleted, key=order.get),
            "pruned": pruned,
            "kept": kept,
            "files": files,
            "bytes": sum(p.stat().st_size for p in files),
        }

    def prune_drafts(self, session_id: str) -> dict:
        """Carry out survey_drafts, and report what it removed.

        One transaction for the rows, so a draft is never deleted while a
        child it was kept for is left pointing at nothing. The report names
        files relative to the image folder, the way the storage sweep does.
        """
        found = self.survey_drafts(session_id)
        stamp = _now()
        with self._connect() as conn:
            conn.executemany(
                "UPDATE generations SET pruned_at = ? WHERE id = ?",
                [(stamp, gen_id) for gen_id in found["pruned"]],
            )
            conn.executemany(
                "DELETE FROM generations WHERE id = ?",
                [(gen_id,) for gen_id in found["deleted"]],
            )
        for path in found["files"]:
            path.unlink(missing_ok=True)
        return {**found, "files": [self.relative(p) for p in found["files"]]}

    # -- storage maintenance ----------------------------------------------

    # What a session directory is allowed to hold: <id>.png, <id>_init.png and
    # a clip in either container. Anything else in there was put there by
    # somebody, not by this app, and is never touched.
    OWNED_SUFFIXES = (".png", ".mp4", ".webp")

    # A row is a candidate only once nothing more can happen to it. Pending and
    # running rows have no image yet by design, and a held job stays pending
    # for as long as Draw Things is away. This is the whole guard: a job saves
    # its image before it sets the status, so a row that has finished and has
    # no image is always damage, never a render in progress.
    FINISHED = ("done", "error")

    def survey_storage(self, session_id: Optional[str] = None) -> dict:
        """The two ways the database and the disk can disagree.

        `missing` is a finished row with no image: a card that draws as a
        broken thumbnail, whether it says done (the image was deleted from
        under it) or error (the render never produced one). Each carries the
        files that would go with it, which is never the image it is missing
        and may be an init image or a clip it left behind. Nothing queued or
        running is ever a candidate; see FINISHED.

        `orphans` are files no row owns. Only files named the way this app
        names them are considered, so a note or an export dropped into a
        session folder is not a candidate for deletion.

        `session_id` narrows both halves to one session's folder. Ownership is
        still decided against every row in the database: a file whose row was
        moved to another session is not an orphan, and narrowing the sweep
        must not turn it into one.
        """
        rows = {}
        with self._connect() as conn:
            for r in conn.execute(
                "SELECT id, session_id, kind, model_key, status, created_at, "
                "pruned_at FROM generations"
            ):
                rows[r["id"]] = dict(r)
        root = self.data_dir / "sessions"
        missing = []
        # counted across the whole install rather than the swept session: it is
        # the evidence that data_dir is the directory these images were written
        # to, and one session legitimately having no image left says nothing
        # about that
        present = 0
        expected = 0
        for gen_id, r in rows.items():
            # a pruned draft has no image by design, and deleting its row would
            # cut the lineage it was kept to hold; it still owns its id, so the
            # orphan check below treats any file of its as owned
            if r["status"] not in self.FINISHED or r["pruned_at"]:
                continue
            # only a done row proves anything: it is the one status that means
            # an image was definitely written, so it is the one the check
            # counts. An install whose only finished rows are errors has
            # nothing to prove the folder with and nothing to lose either.
            if r["status"] == "done":
                expected += 1
            folder = root / r["session_id"]
            if (folder / f"{gen_id}.png").is_file():
                present += r["status"] == "done"
                continue
            if session_id is not None and r["session_id"] != session_id:
                continue
            r.pop("pruned_at")
            missing.append({
                **r,
                "files": [
                    p for p in (
                        folder / f"{gen_id}_init.png",
                        folder / f"{gen_id}.mp4",
                        folder / f"{gen_id}.webp",
                    ) if p.is_file()
                ],
            })
        folders = [root / session_id] if session_id is not None else (
            sorted(root.iterdir()) if root.is_dir() else []
        )
        orphans = []
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if not path.is_file() or path.suffix not in self.OWNED_SUFFIXES:
                    continue
                owner = (
                    path.stem[: -len("_init")]
                    if path.stem.endswith("_init") else path.stem
                )
                if owner not in rows:
                    orphans.append(path)
        missing.sort(key=lambda r: r["created_at"])
        return {
            "missing": missing,
            "orphans": orphans,
            "blocked": self._misconfigured(present, expected),
        }

    @staticmethod
    def _misconfigured(present: int, expected: int) -> Optional[str]:
        """Why this sweep must not run, if the images look like the wrong ones.

        Point DRAFTPOOL_DATA_DIR at an empty or unrelated directory and every
        completed row in the database looks like a deleted image, so a sweep
        would delete the lot. One completed card with its image still on disk
        is enough to prove the directory is the right one; none at all, where
        the database says there should be some, is a configuration to fix
        rather than damage to clean up.

        `expected` and `present` count done rows only, because done is the one
        status that means an image was written. An install whose finished rows
        are all errors has nothing to prove the folder with, and nothing that
        a sweep could destroy either.
        """
        if expected and not present:
            return (
                f"None of the {expected} completed images in the database were "
                "found on disk. That points at the wrong image folder rather "
                "than at deleted images, so nothing has been removed. Check "
                "DRAFTPOOL_DATA_DIR against the path shown above."
            )
        return None

    def cleanup_storage(self, session_id: Optional[str] = None) -> dict:
        """Delete both halves of what survey_storage found.

        Rows go through delete_generation, so a row whose image is gone still
        gives up its init image and any clip it left behind. What comes back
        is what was actually removed, named the way the survey named it.
        """
        found = self.survey_storage(session_id)
        if found["blocked"]:
            raise StorageMisconfigured(found["blocked"])
        for r in found["missing"]:
            self.delete_generation(r["id"])
        for path in found["orphans"]:
            path.unlink(missing_ok=True)
        return {
            "rows": [r["id"] for r in found["missing"]],
            "files": [
                self.relative(p)
                for r in found["missing"] for p in r["files"]
            ] + [self.relative(p) for p in found["orphans"]],
        }

    def relative(self, path: Path) -> str:
        """A file as `<session>/<name>`, the way the app refers to its own."""
        return f"{path.parent.name}/{path.name}"

    def storage_summary(self) -> dict:
        """Where the images live and how much of the disk they are using.

        Counts what is on disk rather than what the database says, because the
        point of showing it is to be able to see the two disagree.
        """
        root = self.data_dir / "sessions"
        files = [
            p for p in root.glob("*/*")
            if p.is_file() and p.suffix in self.OWNED_SUFFIXES
        ] if root.is_dir() else []
        return {
            "data_dir": str(self.data_dir),
            "images_dir": str(root),
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }

    # -- files ------------------------------------------------------------

    def image_path(self, gen: Generation) -> Path:
        return self.data_dir / "sessions" / gen.session_id / f"{gen.id}.png"

    def init_path(self, gen: Generation) -> Path:
        """Optional pre-composited init image for this generation's img2img."""
        return self.data_dir / "sessions" / gen.session_id / f"{gen.id}_init.png"

    def video_path(self, gen: Generation, ext: Optional[str] = None) -> Path:
        """The clip file for a video generation ('mp4' unless webp was used)."""
        return (
            self.data_dir / "sessions" / gen.session_id
            / f"{gen.id}.{ext or gen.video_ext or 'mp4'}"
        )

    def save_image(self, gen: Generation, png: bytes) -> Path:
        path = self.image_path(gen)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return path

    def set_video_ext(self, gen_id: str, ext: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET video_ext = ? WHERE id = ?", (ext, gen_id)
            )

    def set_dimensions(self, gen_id: str, width: int, height: int) -> None:
        """Correct the stored size to what Draw Things actually produced.

        It snaps requests to its own buckets (832x480 comes back 832x448), so
        the requested size would otherwise be recorded as fact.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE generations SET width = ?, height = ? WHERE id = ?",
                (width, height, gen_id),
            )
