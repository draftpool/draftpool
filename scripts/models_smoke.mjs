/**
 * Smoke test for "select all" / "deselect all" under the model chips.
 *
 *   node scripts/models_smoke.mjs
 *
 * Self-contained: it starts its own server on a spare port against a throwaway
 * data directory and cleans both up. Needs `uv` and puppeteer (smoke_browser.mjs
 * says which copy, and which Chrome, it uses). Nothing renders.
 *
 * What it checks is the part a server test cannot see: that the chips' lit
 * state and the selection Generate reads move together, that a greyed chip is
 * never picked up by "all", and that the count beside Generate follows.
 */
const { spawn } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const { launchBrowser } = await import("./smoke_browser.mjs");

const PORT = process.argv[2] || "8572";
const BASE = `http://127.0.0.1:${PORT}`;

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-smoke-"));
const server = spawn("uv", ["run", "uvicorn", "app.main:app", "--port", PORT], {
  cwd: path.join(import.meta.dirname, ".."),
  env: { ...process.env, DRAFTPOOL_DATA_DIR: dataDir },
  stdio: "ignore",
});
const stop = () => {
  server.kill();
  fs.rmSync(dataDir, { recursive: true, force: true });
};
process.on("exit", stop);

let models;
for (let i = 0; ; i++) {
  try { models = await (await fetch(BASE + "/api/models")).json(); break; } catch (e) {
    if (i > 60) { console.error("server never came up:", e.message); process.exit(2); }
    await sleep(500);
  }
}
const pickable = models.filter(m => m.available && m.shown && !m.edit_only).map(m => m.key);
if (pickable.length < 2) {
  console.error("fewer than two installed models; nothing to select all of");
  process.exit(2);
}

const browser = await launchBrowser();
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => errors.push(String(e)));
await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.type("#instruction", "a red cube on a wooden table");

const state = () => page.evaluate(() => ({
  selected: [...selectedModels].sort(),
  lit: [...document.querySelectorAll("#modelChips .chip.selected")].length,
  greyLit: document.querySelectorAll("#modelChips .chip.unavailable.selected").length,
  count: document.querySelector("#genCount").textContent,
  shown: !document.querySelector("#modelBulk").hidden,
}));

let s = await state();
check("the pair is shown when there is more than one model", s.shown);
// it sits in the gutter under the last dialect label, left of the chips
// (after the last chip it wrapped onto a line of its own when that row was full)
const placed = await page.evaluate(() => {
  const bulk = document.querySelector("#modelBulk");
  const label = [...document.querySelectorAll("#modelChips .dialectLabel")].at(-1);
  const strip = [...document.querySelectorAll("#modelChips .dialectChips")].at(-1);
  const a = bulk.getBoundingClientRect(), l = label.getBoundingClientRect(),
        c = strip.getBoundingClientRect();
  return { under: a.top >= l.bottom - 1, left: a.right <= c.left, inBlock: a.bottom <= c.bottom + 1 };
});
check("the pair sits in the gutter under the last label",
  placed.under && placed.left, JSON.stringify(placed));
check("the page opens with one model selected", s.selected.length === 1, JSON.stringify(s.selected));
const oneCount = s.count;

await page.click("#selectAllModels");
s = await state();
check("select all picks every available model",
  JSON.stringify(s.selected) === JSON.stringify([...pickable].sort()),
  `${s.selected.length} of ${pickable.length}`);
check("and lights every one of their chips", s.lit === pickable.length, String(s.lit));
check("and no greyed chip", s.greyLit === 0);
check("the count beside Generate follows", s.count !== oneCount, `${oneCount} -> ${s.count}`);

await page.click("#selectNoModels");
s = await state();
check("deselect all leaves nothing selected", s.selected.length === 0 && s.lit === 0,
  JSON.stringify(s));

// a chip clicked after "none" works as it always did
await page.evaluate(k => chipEls.get(k).click(), pickable[0]);
s = await state();
check("a chip clicked afterwards selects just that one",
  s.selected.length === 1 && s.selected[0] === pickable[0] && s.lit === 1);

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
