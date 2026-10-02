/**
 * Smoke test for Remove drafts, in a real browser.
 *
 *   node scripts/drafts_smoke.mjs
 *
 * Self-contained like session_smoke.mjs: its own server on a spare port, a
 * throwaway data directory, rows seeded straight into the store. Needs `uv`
 * and puppeteer.
 *
 * The server tests prove which rows survive. This proves the page still makes
 * sense of what survived: the removed drafts have no cards, the two finals
 * made from one removed draft are still one lineage, and the "from" link and
 * the params chain say the draft was removed rather than failing on it.
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

const PORT = process.argv[2] || "8572";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(import.meta.dirname, "..");

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-drafts-"));
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
const model = (await api("/api/models")).find(m => m.available && m.shown && !m.edit_only);
if (!model) { console.error("no usable model installed; nothing to test"); process.exit(2); }

// leaf: a draft nothing came from.  kept -> finalA -> up, kept -> finalB:
// one draft finalised twice, one of the finals upscaled.
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
other = st.create_session("another session")
s = st.create_session("drafts smoke")
def mk(kind, parent=None, sid=None):
    g = st.create_generation(session_id=sid or s.id, kind=kind, model_key=sys.argv[2],
        prompt=kind, negative_prompt="", seed=5, width=512, height=512, steps=4,
        cfg=1.0, sampler=None, parent_id=parent)
    st.save_image(g, png()); st.set_status(g.id, "done"); return g.id
leaf = mk("draft"); kept = mk("draft")
a = mk("final", kept); b = mk("final", kept); up = mk("upscale", a)
elsewhere = mk("draft", sid=other.id)
print(json.dumps(dict(session=s.id, leaf=leaf, kept=kept, a=a, b=b, up=up,
                      other=other.id, elsewhere=elsewhere)))
`;
const ids = JSON.parse(execFileSync("uv", ["run", "python", "-c", seed, dataDir, model.key],
  { cwd: ROOT, encoding: "utf8" }));

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
page.on("console", m => process.env.SMOKE_DEBUG && console.log("console:", m.text()));
await page.goto(BASE + "/", { waitUntil: "networkidle2" });
// init() fills the picker only after its slower calls have answered
await page.waitForSelector(`#sessionPicker option[value="${ids.session}"]`, { timeout: 30000 });

// with nothing open there is nothing to scope to, and no install-wide fallback
await page.click("#stepsBtn");
check("Remove drafts is unavailable with no session open",
  await page.$eval("#draftsBtn", b => b.disabled));
await page.click("#stepsClose");
await page.select("#sessionPicker", ids.session);
await sleep(800);

const cardIds = () => page.evaluate(() =>
  [...document.querySelectorAll("#rounds .card")].map(c => c.dataset.genId));
const visible = () => page.evaluate(() =>
  [...document.querySelectorAll("#rounds .card")]
    .filter(c => c.style.display !== "none").map(c => c.dataset.genId));
const same = (a, b) => [...a].sort().join() === [...b].sort().join();

check("the session opens with all five cards", (await cardIds()).length === 5);
const hasPublish = id => page.evaluate(id => !!document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="publish"]`), id);
check("a draft card offers no Publish", !(await hasPublish(ids.leaf)));
check("a final card still does", await hasPublish(ids.a));

await page.click("#stepsBtn");
// the storage check's scope says everywhere; drafts must ignore it
await page.select("#cleanupScope", "all");
await page.click("#draftsBtn");
await sleep(500);
const listing = await page.$eval("#draftsList", el => el.textContent);
check("the survey lists the leaf draft for deletion",
  /to delete:\n\s+/.test(listing) && listing.split("lineage only")[0].includes(ids.leaf), listing);
check("and the finalised one as lineage only",
  (listing.split("lineage only")[1] || "").includes(ids.kept), listing);
check("nothing is removed by looking", (await cardIds()).length === 5);
check("the other session's draft is not listed", !listing.includes(ids.elsewhere));

await page.click("#draftsGo");
await sleep(800);
check("both draft cards are gone and the rest remain",
  same(await cardIds(), [ids.a, ids.b, ids.up]), JSON.stringify(await cardIds()));
const otherSession = await api(`/api/sessions/${ids.other}`);
check("the other session's draft survives",
  otherSession.generations.some(g => g.id === ids.elsewhere && !g.pruned_at));
check("the report says one was kept as lineage",
  (await page.$eval("#draftsOut", el => el.textContent)).includes("1 kept as lineage"));
await page.click("#stepsClose");

// the two finals are siblings only through the removed draft
await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="lineage"]`).click(), ids.b);
await sleep(300);
check("the lineage filter still joins both finals of the removed draft",
  same(await visible(), [ids.a, ids.b, ids.up]), JSON.stringify(await visible()));
await page.evaluate(() => document.querySelector("#fLineageChip")?.click());
await sleep(200);

await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] .fromLink`).click(), ids.a);
const toastText = await page.$eval("#toast", el => el.textContent);
check("the from link says the draft was removed", toastText.includes("removed"), toastText);

await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="params"]`).click(), ids.up);
await sleep(500);
const chain = await page.$eval("#viewer", el => el.innerText);
check("the params chain still reaches the removed draft",
  chain.includes(ids.kept.slice(0, 6)) && chain.includes("image removed"), chain.slice(0, 300));

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
