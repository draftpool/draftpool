/**
 * Smoke test for the measured waits the page names before a job is queued.
 *
 *   node scripts/timings_smoke.mjs
 *
 * Self-contained, like session_smoke.mjs: its own server on a spare port and a
 * throwaway data directory, seeded with finished renders whose durations are
 * chosen, so every number on the page can be checked against arithmetic. Needs
 * `uv` and puppeteer. Nothing is rendered.
 *
 * What it covers that tests/test_timings.py cannot: that the composer's count
 * adds up the fan-out it names, says so when some models have not been timed
 * and says nothing when none have; that the model chips and a draft's Finalize
 * and Faithful tooltips carry the measured wait; that a final's Upscale button,
 * size picker and model picker name what each size and model has taken; and
 * that a finished job redraws all of them with the new numbers without
 * resetting what was picked.
 */
const require = (await import("node:module")).createRequire(import.meta.url);
const { spawn, execFileSync } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const PUPPETEER = [
  "/opt/homebrew/lib/node_modules/website-scraper-puppeteer/node_modules/puppeteer",
  "puppeteer",
];
let puppeteer = null;
for (const p of PUPPETEER) { try { puppeteer = require(p); break; } catch {} }
if (!puppeteer) { console.error("puppeteer not found; npm i -g puppeteer"); process.exit(2); }

const PORT = process.argv[2] || "8574";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(import.meta.dirname, "..");

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-timings-"));
const server = spawn("uv", ["run", "uvicorn", "app.main:app", "--port", PORT], {
  cwd: ROOT,
  env: { ...process.env, DRAFTPOOL_DATA_DIR: dataDir },
  stdio: "ignore",
});
const stop = () => {
  server.kill();
  fs.rmSync(dataDir, { recursive: true, force: true });
};
process.on("exit", stop);

const api = async p => {
  const r = await fetch(BASE + p);
  if (!r.ok) throw new Error(`${r.status} ${p}: ${await r.text()}`);
  return r.json();
};
for (let i = 0; ; i++) {
  try { await api("/api/models"); break; } catch (e) {
    if (i > 60) { console.error("server never came up:", e.message); process.exit(2); }
    await sleep(500);
  }
}

// Two models: one with history, one without. Which ones is up to the install;
// the timed one needs aspect support, img2img and more than one upscale size
// above its own finals, so every check has something to look at.
const usable = (await api("/api/models")).filter(m => m.available && m.shown && !m.edit_only);
const timed = usable.find(m => m.supports_aspect && m.supports_img2img && m.can_upscale
  && m.upscale_size_options.filter(z => z > m.final.size).length > 1);
const untimed = usable.find(m => m !== timed);
if (!timed || !untimed) { console.error("needs two usable models, one with aspect, img2img and two upscale sizes"); process.exit(2); }
// the size that gets timed, and the one it is not: the default stays untimed so
// the check that picking a size moves the button's wait means something
const upSizes = timed.upscale_size_options.filter(z => z > timed.final.size);
const upTimed = upSizes.find(z => z !== timed.upscale_default_size);

// Seconds per step, picked so a round of them lands on a round number
const DRAFT_RATE = 2.0, FINAL_RATE = 6.0, FAITHFUL_RATE = 4.0, UPSCALE_RATE = 10.0;
const faithfulSteps = m => {
  const { steps, strength } = m.final;
  return strength > 0 && strength < 1 ? Math.min(64, Math.round(steps / strength)) : steps;
};

// Three renders of each kind -- the least that is reported -- one of them slow
// by a restart, which the median must not pass on. Plus a draft in a session,
// for the card.
const seed = `
import json, sys, zlib, struct
from pathlib import Path
from app.sessions import Store
def png():
    raw = b"\\x00\\x80\\x80\\x80"
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
st = Store(Path(sys.argv[1]))
key, dsize, dsteps, fsize, fsteps, fasteps, usize, usteps = sys.argv[2], *map(int, sys.argv[3:10])
rates = json.loads(sys.argv[10])
s = st.create_session("timings smoke")
def mk(kind, size, steps, secs, parent=None):
    g = st.create_generation(session_id=s.id, kind=kind, model_key=key, prompt="p",
        negative_prompt="", seed=5, width=size, height=size, aspect="1:1", steps=steps,
        cfg=1.0, sampler=None, parent_id=parent)
    st.save_image(g, png()); st.set_status(g.id, "done", duration_s=secs); return g.id
drafts = [mk("draft", dsize, dsteps, rates["draft"] * dsteps * f) for f in (1, 1, 20)]
finals = [mk("final", fsize, fsteps, rates["final"] * fsteps * f, drafts[0]) for f in (1, 20, 1)]
[mk("faithful", fsize, fasteps, rates["faithful"] * fasteps * f, drafts[0]) for f in (20, 1, 1)]
[mk("upscale", usize, usteps, rates["upscale"] * usteps * f, finals[1]) for f in (1, 1, 20)]
print(json.dumps(dict(session=s.id, draft=drafts[1], final=finals[0])))
`;
const seedArgs = [
  timed.key, timed.draft.size, timed.draft.steps, timed.final.size, timed.final.steps,
  faithfulSteps(timed), upTimed, timed.upscale.steps,
  JSON.stringify({ draft: DRAFT_RATE, final: FINAL_RATE, faithful: FAITHFUL_RATE, upscale: UPSCALE_RATE }),
].map(String);
const ids = JSON.parse(execFileSync("uv", ["run", "python", "-c", seed, dataDir, ...seedArgs],
  { cwd: ROOT, encoding: "utf8" }));

// The page's own rounding, restated: an estimate that is right but printed
// differently is still a regression someone will see.
const fmtWait = sec => {
  if (sec < 90) return `${Math.max(1, Math.round(sec))}s`;
  const min = Math.round(sec / 60);
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
};
const oneDraft = DRAFT_RATE * timed.draft.steps;
const oneUpscale = UPSCALE_RATE * timed.upscale.steps;

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForSelector(`#sessionPicker option[value="${ids.session}"]`, { timeout: 30000 });

const select = keys => page.evaluate(keys => {
  [...chipEls.keys()].forEach(k => {
    if (selectedModels.has(k) !== keys.includes(k)) chipEls.get(k).click();
  });
}, keys);
const setSeeds = n => page.evaluate(n => {
  const el = document.querySelector("#seedsPer");
  el.value = n;
  el.dispatchEvent(new Event("input"));
}, n);
const count = () => page.evaluate(() => {
  const el = document.querySelector("#genCount");
  return { text: el.textContent, eta: el.querySelector("#genEta")?.textContent ?? null, title: el.title };
});

await page.evaluate(() => {
  const el = document.querySelector("#instruction");
  el.value = "a lighthouse at dusk";
  el.dispatchEvent(new Event("input"));
});

// -- the composer ------------------------------------------------------------

await select([timed.key]);
await setSeeds(2);
let c = await count();
check("one timed model: the count names the round's wait",
  c.eta === `≈${fmtWait(oneDraft * 2)}`, JSON.stringify(c));
check("...and the tooltip says where the number comes from",
  c.title.includes("this machine") && !c.title.includes("Not timed"), c.title);

await setSeeds(5);
c = await count();
check("more seeds, longer wait", c.eta === `≈${fmtWait(oneDraft * 5)}`, c.eta);

await select([timed.key, untimed.key]);
c = await count();
check("an untimed model makes the total a floor",
  c.eta === `≈${fmtWait(oneDraft * 5)}+`, c.eta);
check("...and is named in the tooltip", c.title.includes(`Not timed yet, so not in the total: ${untimed.label}`), c.title);

await select([untimed.key]);
c = await count();
check("no timed model: no wait is named at all", c.eta === null && c.text.length > 0, JSON.stringify(c));

// a ratio holds the pixel budget, so the square's measurements still apply
await select([timed.key]);
await setSeeds(1);
await page.evaluate(() => [...document.querySelectorAll("#aspectRow .chip")]
  .find(c => c.textContent.includes("3:4")).click());
c = await count();
check("a portrait draft is timed by the square renders of the same budget",
  c.eta === `≈${fmtWait(oneDraft)}`, c.eta);
await page.evaluate(() => [...document.querySelectorAll("#aspectRow .chip")]
  .find(c => c.textContent.includes("1:1")).click());

// the step count is the dial: the wait follows it
await page.evaluate((key, steps) => { MODELS.find(m => m.key === key).draft.steps = steps; renderGenCount(); },
  timed.key, timed.draft.steps * 2);
c = await count();
check("doubling the draft steps doubles the wait", c.eta === `≈${fmtWait(oneDraft * 2)}`, c.eta);
await page.evaluate((key, steps) => { MODELS.find(m => m.key === key).draft.steps = steps; renderGenCount(); },
  timed.key, timed.draft.steps);

// -- the model chips ---------------------------------------------------------

const chipTitle = key => page.evaluate(key => chipEls.get(key).title, key);
let t = await chipTitle(timed.key);
check("a timed chip names its draft and final waits",
  t.includes(`(≈${fmtWait(oneDraft)} here)`) &&
  t.includes(`(≈${fmtWait(FINAL_RATE * timed.final.steps)} here)`), t);
t = await chipTitle(untimed.key);
check("an untimed chip names none", !t.includes("here)"), t);

// -- a draft's card ----------------------------------------------------------

await page.select("#sessionPicker", ids.session);
await page.waitForSelector(`#rounds .card[data-gen-id="${ids.draft}"] [data-act="finalize"]`, { timeout: 10000 });
const tips = () => page.evaluate(id => Object.fromEntries(
  [...document.querySelectorAll(`#rounds .card[data-gen-id="${id}"] [data-act]`)]
    .filter(b => ["finalize", "finalize-desc", "faithful"].includes(b.dataset.act))
    .map(b => [b.dataset.act, b.title])), ids.draft);
let tp = await tips();
check("Finalize names what a final has taken",
  tp.finalize?.endsWith(`Takes ≈${fmtWait(FINAL_RATE * timed.final.steps)} here.`), tp.finalize);
check("Finalize+ says the LLM comes on top",
  tp["finalize-desc"]?.includes("plus the LLM reading the image"), tp["finalize-desc"]);
check("Faithful is timed by Faithfuls, at the steps it runs",
  tp.faithful?.endsWith(`Takes ≈${fmtWait(FAITHFUL_RATE * faithfulSteps(timed))} here.`), tp.faithful);

// -- a final's Upscale -------------------------------------------------------

const up = () => page.evaluate(id => {
  const card = document.querySelector(`#rounds .card[data-gen-id="${id}"]`);
  const opt = sel => [...card.querySelectorAll(`select.${sel} option`)]
    .map(o => ({ value: o.value, text: o.textContent }));
  return {
    title: card.querySelector('[data-act="upscale"]').title,
    size: card.querySelector("select.upsize").value,
    sizes: opt("upsize"),
    via: opt("upvia"),
  };
}, ids.final);
const pickSize = z => page.evaluate((id, z) => {
  const sel = document.querySelector(`#rounds .card[data-gen-id="${id}"] select.upsize`);
  sel.value = String(z);
  sel.dispatchEvent(new Event("change"));
}, ids.final, z);
const sizeLabel = z => (z % 1024 === 0 ? `${z / 1024}K` : `${z}px`);

let u = await up();
check("the default size is untimed, so the button names no wait",
  u.size === String(timed.upscale_default_size) && !u.title.includes("Takes"), JSON.stringify(u));
// The wait is the model picker's: a size carries only its cost over 2K, or
// the closed group would show the one measurement twice.
check("no size names a time, timed or not",
  u.sizes.length > 1 && u.sizes.every(o => !/≈|\d+s\b|min/.test(o.text)),
  JSON.stringify(u.sizes));
check("a size above 2K shows its multiplier",
  u.sizes.filter(o => +o.value > 2048).every(o => /· ~[\d.]+x$/.test(o.text)),
  JSON.stringify(u.sizes));

await pickSize(upTimed);
u = await up();
check("picking the timed size puts its wait on the button",
  u.title.endsWith(`Takes ≈${fmtWait(oneUpscale)} here at ${sizeLabel(upTimed)}.`), u.title);
const own = u.via.find(o => o.value === timed.key);
check("the image's own model says what it took at that size",
  own?.text.endsWith(`≈${fmtWait(oneUpscale)} here`), JSON.stringify(own));
const others = u.via.filter(o => o.value !== timed.key);
// The configured figure is printed by the same formatter as a measured one:
// 340 s configured beside 4 min measured reads as two scales, not one column.
const configured = o => usable.find(m => m.key === o.value)?.upscale_seconds;
check("the other models keep their configured 2K figure, labelled as one",
  others.length > 0 && others.every(o => !o.text.includes("here") &&
    (configured(o) ? o.text.endsWith(` · ~${fmtWait(configured(o))} at 2K`) : !o.text.includes("·"))),
  JSON.stringify(others.slice(0, 3)));

// -- a finished job moves every number ---------------------------------------

// three more drafts at five times the pace take the median with them
const more = `
import sys
from pathlib import Path
from app.sessions import Store
st = Store(Path(sys.argv[1]))
s = st.create_session("more")
for _ in range(4):
    g = st.create_generation(session_id=s.id, kind="draft", model_key=sys.argv[2], prompt="p",
        negative_prompt="", seed=5, width=int(sys.argv[3]), height=int(sys.argv[3]),
        steps=int(sys.argv[4]), cfg=1.0, sampler=None)
    st.set_status(g.id, "done", duration_s=float(sys.argv[5]))
    g = st.create_generation(session_id=s.id, kind="upscale", model_key=sys.argv[2], prompt="p",
        negative_prompt="", seed=5, width=int(sys.argv[6]), height=int(sys.argv[6]),
        steps=int(sys.argv[7]), cfg=1.0, sampler=None)
    st.set_status(g.id, "done", duration_s=float(sys.argv[8]))
`;
execFileSync("uv", ["run", "python", "-c", more, dataDir, timed.key,
  String(timed.draft.size), String(timed.draft.steps), String(oneDraft * 5),
  String(upTimed), String(timed.upscale.steps), String(oneUpscale * 5)], { cwd: ROOT });
// what the SSE handler does with a job that finished
await page.evaluate(id => updateCardStatus(cards.get(id), id, "done", null, 1), ids.draft);
await sleep(2500);
// opening the session reset the composer; put back one draft of the timed model
await select([timed.key]);
await setSeeds(1);
c = await count();
check("the composer picks up the new pace", c.eta === `≈${fmtWait(oneDraft * 5)}`, JSON.stringify(c));
t = await chipTitle(timed.key);
check("so does the chip", t.includes(`(≈${fmtWait(oneDraft * 5)} here)`), t);
tp = await tips();
check("and the card keeps one measured part, not two",
  (tp.finalize.match(/Takes/g) || []).length === 1, tp.finalize);
u = await up();
check("the Upscale button picks up the new pace, once",
  u.title.endsWith(`Takes ≈${fmtWait(oneUpscale * 5)} here at ${sizeLabel(upTimed)}.`)
  && (u.title.match(/Takes/g) || []).length === 1, u.title);
check("...without resetting the size that was picked", u.size === String(upTimed), u.size);

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
