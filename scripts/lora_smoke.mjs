/**
 * Smoke test for the LoRA ticks in the ⚙ dialog, in a real browser.
 *
 *   node scripts/lora_smoke.mjs
 *
 * Self-contained like session_smoke.mjs: its own server on a spare port and a
 * throwaway data directory, so the ticks it flips never reach the settings of
 * the install being worked in. Nothing renders.
 *
 * What only a browser can see: a tick changes more than the LoRA -- the step
 * boxes on its row, the ⚡ on the chip and, for a CFG 1 recipe, whether the
 * negative-prompt box is there at all -- and all of that is redrawn from the
 * server's reply rather than guessed on the page.
 */
const require = (await import("node:module")).createRequire(import.meta.url);
const { spawn } = await import("node:child_process");
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

const PORT = process.argv[2] || "8573";
const BASE = `http://127.0.0.1:${PORT}`;

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-lora-smoke-"));
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

// a generator whose ticked option takes the negative prompt away, installed here
const models = await api("/api/models");
const m = models.find(m => m.available && !m.edit_only && m.uses_negative_prompt === false
  && (m.lora_options || []).some(o => o.on && o.available));
if (!m) { console.error("needs a model with an installed LoRA option that is on"); process.exit(2); }
const opt = m.lora_options.find(o => o.on && o.available);
console.log(`using ${m.key} / ${opt.key}`);

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForFunction(() => typeof MODELS !== "undefined" && MODELS.length > 0);

// only this model in the composer, so the negative box answers for it alone
const state = () => page.evaluate(key => {
  const chip = [...document.querySelectorAll(".chip")].find(c => c.textContent.startsWith(
    MODELS.find(m => m.key === key).label));
  const btn = chip && chip.querySelector(".loraBtn");
  const row = [...document.querySelectorAll("#stepsRows tr")].find(r => r.dataset.key === key);
  return {
    chip: chip ? chip.textContent : null,
    chipOn: btn ? btn.classList.contains("on") : null,
    negHidden: document.querySelector("#negative").hidden,
    final: row ? row.querySelector('input[data-stage="final"]').value : null,
    ticked: row ? row.querySelector(".loraTick").checked : null,
  };
}, m.key);
await page.evaluate(key => {
  selectedModels = new Set([key]);
  renderModelChips();
  renderPlanLine();
}, m.key);

await page.click("#stepsBtn");
await page.waitForSelector(`#stepsRows tr[data-key="${m.key}"] .loraTick`);
let s = await state();
check("ticked: the chip's ⚡ is lit", s.chip && s.chip.endsWith("⚡") && s.chipOn === true, `${s.chip} on=${s.chipOn}`);
check("ticked: no negative box at CFG 1", s.negHidden === true);
check("ticked: the row shows the recipe's final steps", s.final === String(m.final.steps), s.final);

await page.click(`#stepsRows tr[data-key="${m.key}"] .loraTick`);
await page.waitForFunction(key => !MODELS.find(m => m.key === key).lora_options.some(o => o.on), {}, m.key);
await sleep(200);
s = await state();
const saved = JSON.parse(fs.readFileSync(path.join(dataDir, "settings.json"), "utf8"));
check("unticked: saved for this install", saved.loras?.[m.key]?.[opt.key] === false,
  JSON.stringify(saved.loras));
check("unticked: the tick stays off after the redraw", s.ticked === false);
check("unticked: the chip's ⚡ is dim", s.chip && s.chipOn === false, `${s.chip} on=${s.chipOn}`);
check("unticked: the negative box is back", s.negHidden === false);
check("unticked: the row shows the model's own final steps",
  s.final !== String(m.final.steps), s.final);

// back on from the chip itself, the dialog closed: the switch lives there too
await page.evaluate(() => document.querySelector("#stepsDlg").close());
await page.evaluate(key => {
  const chip = [...document.querySelectorAll(".chip")].find(c => c.textContent.startsWith(
    MODELS.find(m => m.key === key).label));
  chip.querySelector(".loraBtn").click();
}, m.key);
await page.waitForFunction(key => MODELS.find(m => m.key === key).lora_options.some(o => o.on), {}, m.key);
await sleep(200);
await page.click("#stepsBtn");
await sleep(200);
s = await state();
check("the chip's ⚡ turns it back on: recipe, lit, table ticked",
  s.final === String(m.final.steps) && s.chipOn === true && s.ticked === true,
  `${s.final} on=${s.chipOn} ticked=${s.ticked}`);
check("the model stayed selected through the switch",
  await page.evaluate(key => selectedModels.has(key), m.key));

check("no page errors", !errors.length, errors.join(" | "));
await browser.close();
console.log(failed ? `${failed} failed` : "all passed");
process.exit(failed ? 1 : 0);
