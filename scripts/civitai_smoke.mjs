/**
 * Smoke test for the Civitai field in the ⚙ dialog, in a real browser.
 *
 *   node scripts/civitai_smoke.mjs
 *
 * Self-contained like lora_smoke.mjs: its own server on a spare port and a
 * throwaway data directory, with the startup step off (DRAFTPOOL_NO_MIGRATE),
 * and the setting answered in the browser -- one model is made to lack its
 * version in /api/models, and the PUT never reaches the server -- so nothing
 * is written to this checkout's config.
 *
 * What only a browser can see: the field is on the row that lacks a version
 * and on no other, Enter and Set both send what was pasted, the server's
 * sentence is what the toast says, and the row and the chips are redrawn from
 * a fresh /api/models rather than patched on the page.
 */
const { spawn } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const { launchBrowser } = await import("./smoke_browser.mjs");

const PORT = process.argv[2] || "8574";
const BASE = `http://127.0.0.1:${PORT}`;

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-civitai-smoke-"));
const server = spawn("uv", ["run", "uvicorn", "app.main:app", "--port", PORT], {
  cwd: path.join(import.meta.dirname, ".."),
  env: { ...process.env, DRAFTPOOL_DATA_DIR: dataDir, DRAFTPOOL_NO_MIGRATE: "1" },
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

const target = (await api("/api/models")).find(m => !m.edit_only);
console.log(`using ${target.key}`);
const AIR = "urn:air:sdxl:checkpoint:civitai:1307857@3313575+3198988";
let given = null;   // the version the fake server has been told, once it has
const sent = [];

const browser = await launchBrowser();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
await page.setRequestInterception(true);
page.on("request", async req => {
  const url = new URL(req.url());
  if (url.pathname === "/api/models" && req.method() === "GET") {
    const rows = await api("/api/models");
    for (const m of rows) {
      if (m.key === target.key) {
        m.civitai_version_id = given;
        m.can_identify = given === null;
      } else {
        m.can_identify = false;
      }
    }
    return req.respond({ status: 200, contentType: "application/json", body: JSON.stringify(rows) });
  }
  if (url.pathname === `/api/models/${target.key}/civitai` && req.method() === "PUT") {
    sent.push(JSON.parse(req.postData()));
    given = 3313575;
    return req.respond({ status: 200, contentType: "application/json",
      body: JSON.stringify({ message: `${target.label}: credited; its settings are kept.`,
                             civitai_version_id: given }) });
  }
  req.continue();
});

await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForFunction(() => typeof MODELS !== "undefined" && MODELS.length > 0);
await page.click("#stepsBtn");
const rowSel = `#stepsRows tr[data-key="${target.key}"]`;
await page.waitForSelector(`${rowSel} .civitaiRef`);

// SHOT=/path/to.png keeps a picture of the dialog as it opens, for the layout
if (process.env.SHOT) {
  await page.setViewport({ width: 1300, height: 1100 });
  await (await page.$("#stepsDlg")).screenshot({ path: process.env.SHOT });
}
const fields = await page.$$eval("#stepsRows .civitaiRef", els => els.length);
check("the field is on the one row without a version, and only there", fields === 1, `${fields}`);

await page.type(`${rowSel} .civitaiRef`, AIR);
await page.keyboard.press("Enter");
await page.waitForFunction(sel => !document.querySelector(`${sel} .civitaiRef`), {}, rowSel);
await sleep(200);
check("Enter sends what was pasted, untouched", sent.length === 1 && sent[0].ref === AIR,
  JSON.stringify(sent));
const toastText = await page.evaluate(() => [...document.querySelectorAll(".toast, #toast")]
  .map(t => t.textContent).join(" | "));
check("the toast is the server's sentence", toastText.includes("credited; its settings are kept"),
  toastText);
const cell = await page.$eval(`${rowSel} td.civitai`, td => td.textContent.trim());
check("the row now shows the version, from a fresh model list", cell === "v3313575", cell);
check("the page's model list has it too",
  await page.evaluate(key => MODELS.find(m => m.key === key).civitai_version_id, target.key) === 3313575);

check("no page errors", !errors.length, errors.join(" | "));
await browser.close();
console.log(failed ? `${failed} failed` : "all passed");
process.exit(failed ? 1 : 0);
