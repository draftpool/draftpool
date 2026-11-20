/**
 * Smoke test for the size filter, in a real browser.
 *
 *   node scripts/filter_smoke.mjs
 *
 * Self-contained like combine_smoke.mjs: its own server on a spare port, a
 * throwaway data directory, rows seeded straight into the store. Nothing is
 * rendered and nothing needs to be installed.
 *
 * What it proves: there is one set of kind chips, and it is the size filter
 * too -- drafts, finals, upscaled -- with no second group repeating those
 * names under other rules. "upscaled" is Upscale and 2x fast together; every
 * other kind is its own chip. Each chip's count is the number of cards it
 * shows, and the session's stats line says the same numbers in the same
 * words. Chips add up within the row (finals and upscaled together), clear
 * resets them, and the lineage chip keeps its selected look. chipKind and
 * kindWord, the two functions behind all of that, are checked directly.
 */
const { spawn, execFileSync } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const { launchBrowser } = await import("./smoke_browser.mjs");

const PORT = process.argv[2] || "8575";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(import.meta.dirname, "..");

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-filter-"));
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
const models = await api("/api/models");
const twoModels = models.filter(m => !m.edit_only).slice(0, 2).map(m => m.key);

// shaped like a real session: a pile of drafts over two models, then the
// finals, faithfuls, passes and enlargements made from a few of them
const seed = `
import json, sys, io
from pathlib import Path
from PIL import Image
from app.sessions import Store
st = Store(Path(sys.argv[1]))
s = st.create_session("filter smoke")
def mk(kind, w, h, parent=None, done=True, model=sys.argv[2], **kw):
    g = st.create_generation(session_id=s.id, kind=kind, model_key=model,
        prompt=kind, negative_prompt="", seed=1, width=w, height=h, steps=4,
        cfg=1.0, sampler=None, parent_id=parent, **kw)
    if done:
        b = io.BytesIO(); Image.new("RGB", (16, 16), (90, 90, 90)).save(b, format="PNG")
        st.save_image(g, b.getvalue()); st.set_status(g.id, "done")
    return g.id
ids = {}
drafts = [mk("draft", 640, 896, model=sys.argv[2 + i % 2]) for i in range(12)]
ids["draft"] = drafts[0]
ids["final"] = mk("final", 1024, 1024, drafts[0])
ids["faithful"] = mk("faithful", 1024, 1024, drafts[1])
ids["faithful2"] = mk("faithful", 1024, 1024, drafts[2], model=sys.argv[3])
ids["edit"] = mk("edit", 1024, 1024, ids["final"])
ids["detail"] = mk("detail", 1024, 1024, ids["faithful"])
ids["up"] = mk("upscale", 2048, 2048, ids["final"])
ids["up2"] = mk("upscale", 2048, 2048, ids["faithful2"], model=sys.argv[3])
ids["esrgan"] = mk("esrgan", 2048, 2048, ids["faithful"])
ids["clip"] = mk("video", 832, 480, ids["final"], num_frames=33, fps=16)
print(json.dumps(dict(session=s.id, total=12 + 9, **ids)))
`;
const ids = JSON.parse(execFileSync("uv", ["run", "python", "-c", seed, dataDir, ...twoModels],
  { cwd: ROOT, encoding: "utf8" }));

const browser = await launchBrowser();
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });

await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForSelector(`#sessionPicker option[value="${ids.session}"]`, { timeout: 30000 });
await page.select("#sessionPicker", ids.session);
await sleep(800);

// -- the two functions, directly
// a function that is missing is a failure to report, not a crash that hides
// every check after it
const unit = await page.evaluate(() => { try { return {
  kinds: ["draft", "final", "faithful", "edit", "detail", "upscale", "esrgan", "video"].map(chipKind),
  words: [kindWord("draft", 1), kindWord("draft", 3), kindWord("detail", 1), kindWord("detail", 2),
          kindWord("upscale", 1), kindWord("upscale", 4), kindWord("video", 2)],
  plurals: ["draft", "faithful", "detail", "upscale", "video"].map(kindPlural),
}; } catch (e) { return { kinds: String(e), words: [], plurals: [] }; } });
check("chipKind folds 2x fast into upscaled and leaves every other kind alone",
  JSON.stringify(unit.kinds) === JSON.stringify(
    ["draft", "final", "faithful", "edit", "detail", "upscale", "upscale", "video"]), JSON.stringify(unit.kinds));
check("kindWord counts in the chips' words, singular at one",
  JSON.stringify(unit.words) === JSON.stringify(
    ["1 draft", "3 drafts", "1 face fix", "2 face fixes", "1 upscaled", "4 upscaled", "2 clips"]),
  JSON.stringify(unit.words));
check("kindPlural: face fixes, not face fixs",
  JSON.stringify(unit.plurals) === JSON.stringify(["drafts", "faithfuls", "face fixes", "upscaled", "clips"]),
  JSON.stringify(unit.plurals));

// -- the row
const name = Object.fromEntries(Object.entries(ids).map(([k, v]) => [v, k]));
const row = await page.evaluate(() => ({
  chips: [...document.querySelectorAll("#fChips .chip")].map(c => ({
    kind: c.dataset.kind || null, model: c.dataset.model || null,
    label: c.firstChild.textContent.trim(), n: +c.querySelector("b").textContent })),
  hint: [...document.querySelectorAll("#fChips .hint")].map(h => h.textContent),
  stats: document.querySelector("#sessionStats summary")?.textContent || "",
  cards: document.querySelectorAll("#rounds .card").length,
}));
const kindChips = row.chips.filter(c => c.kind);
check("one kind chip per category, upscaled in place of upscales and 2xs",
  JSON.stringify(kindChips.map(c => `${c.label} ${c.n}`)) === JSON.stringify(
    ["drafts 12", "finals 1", "faithfuls 2", "edits 1", "face fixes 1", "upscaled 3", "clips 1"]),
  JSON.stringify(kindChips.map(c => `${c.label} ${c.n}`)));
const labels = row.chips.map(c => c.label);
check("no label appears twice in the row, and there is no second size group",
  new Set(labels).size === labels.length && !row.hint.length,
  JSON.stringify({ labels, hint: row.hint }));
check("the kind chips add up to every card",
  kindChips.reduce((t, c) => t + c.n, 0) === row.cards && row.cards === ids.total,
  `${kindChips.reduce((t, c) => t + c.n, 0)} vs ${row.cards}`);
const missing = ["12 drafts", "1 final", "2 faithfuls", "1 edit", "1 face fix", "3 upscaled", "1 clip"]
  .filter(w => !row.stats.includes(w));
check("the stats line says the chips' numbers in the chips' words",
  !missing.length && !/\b2x\b|upscales/.test(row.stats), JSON.stringify({ stats: row.stats, missing }));

// -- filtering: each chip shows exactly the cards it counts
const shown = async () => (await page.evaluate(() => [...document.querySelectorAll("#rounds .card")]
  .filter(c => c.style.display !== "none").map(c => c.dataset.genId))).map(id => name[id] || "draft*").sort();
const clickKind = k => page.evaluate(k => document.querySelector(`#fChips .chip[data-kind="${k}"]`).click(), k);
for (const c of kindChips) {
  await clickKind(c.kind);
  const got = await shown();
  check(`${c.label}: shows ${c.n}, as its chip says`, got.length === c.n, JSON.stringify(got));
  await clickKind(c.kind);
}
await clickKind("upscale");
let got = await shown();
check("upscaled shows both upscales and the 2x", JSON.stringify(got) === JSON.stringify(["esrgan", "up", "up2"]),
  JSON.stringify(got));
await clickKind("final");
got = await shown();
check("finals and upscaled together show both",
  JSON.stringify(got) === JSON.stringify(["esrgan", "final", "up", "up2"]), JSON.stringify(got));
const sel = await page.$$eval("#fChips .chip.selected", cs => cs.map(c => c.dataset.kind));
check("both chips show as selected", JSON.stringify(sel.sort()) === JSON.stringify(["final", "upscale"]),
  JSON.stringify(sel));
await page.click("#fClear");
await sleep(150);
got = await shown();
check("clear shows every card again", got.length === ids.total, String(got.length));

// -- the lineage chip is a selection too, and has to look like one
await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="lineage"]`).click(), ids.final);
await sleep(200);
check("the lineage chip keeps its selected look",
  await page.$eval("#fLineageChip", c => c.classList.contains("selected")));

// -- each group highlights in its own colour, at rest and selected
const look = () => page.evaluate(() => {
  const css = sel => { const el = document.querySelector(sel); if (!el) return null;
    const st = getComputedStyle(el); return { border: st.borderColor, color: st.color, bg: st.backgroundColor }; };
  return { kind: css('#fChips .chip[data-kind="final"]'), model: css("#fChips .chip[data-model]"),
           lineage: css("#fLineageChip") };
});
const rest = await look();
await page.click("#fLineageChip");          // drop the lineage, then select one of each
await sleep(150);
await clickKind("final");
await page.evaluate(() => document.querySelector("#fChips .chip[data-model]").click());
await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="lineage"]`).click(), ids.final);
await sleep(200);
const on = await look();
const distinct = (...xs) => new Set(xs).size === xs.length;
check("selected, kind, model and lineage chips are three different colours",
  on.kind && on.model && on.lineage && distinct(on.kind.border, on.model.border, on.lineage.border)
  && distinct(on.kind.color, on.model.color, on.lineage.color), JSON.stringify(on));
check("at rest the kind and model groups already differ",
  rest.kind.border !== rest.model.border, JSON.stringify(rest));
// small text: each selected chip's colour must stay readable on its own wash
// (WCAG AA for normal text, 4.5:1)
const ratio = (fg, bg) => {
  const lin = c => { const n = c.match(/[\d.]+/g).slice(0, 3).map(Number);
    const v = /^color\(srgb/.test(c) ? n : n.map(x => x / 255);
    const l = v.map(x => x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);
    return 0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]; };
  const [a, b] = [lin(fg), lin(bg)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
};
const contrast = Object.fromEntries(Object.entries(on).map(([k, v]) => [k, +ratio(v.color, v.bg).toFixed(2)]));
check("every group's selected text is readable on its wash (4.5:1)",
  Object.values(contrast).every(r => r >= 4.5), JSON.stringify(contrast));
check("selected is not the resting look", on.kind.bg !== rest.kind.bg && on.kind.border !== rest.kind.border,
  JSON.stringify({ rest: rest.kind, on: on.kind }));

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
