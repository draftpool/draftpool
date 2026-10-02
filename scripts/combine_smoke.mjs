/**
 * Smoke test for Combine, in a real browser.
 *
 *   node scripts/combine_smoke.mjs
 *
 * Self-contained like drafts_smoke.mjs: its own server on a spare port, a
 * throwaway data directory, rows seeded straight into the store. Needs `uv`,
 * puppeteer and an instruction editor installed in Draw Things (the button is
 * only offered when one is). Nothing is rendered: the combine request is
 * answered inside the browser, so what is checked is what the page sends.
 *
 * What it proves: the button appears at exactly two picks, the dialog puts the
 * first pick as the subject and swaps on ⇄, the request names the right
 * pictures, a finished combine card links to both of its origins and shows
 * up under the subject's lineage as well as the scene's, and its Upscale starts
 * on the checkpoint that drew the scene and never offers the editor.
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

const PORT = process.argv[2] || "8573";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(import.meta.dirname, "..");

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-combine-"));
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
// the last checkpoint rather than the first, so "the picker starts on the model
// that drew it" cannot pass just because that model is first in any list
const model = models.filter(m => m.available && m.shown && !m.edit_only).at(-1);
const editor = models.find(m => m.available && m.shown && m.edit_only);
if (!model || !editor) { console.error("needs a checkpoint and an instruction editor installed"); process.exit(2); }

// scene and subject are two finals; done is a finished combine of the two
const seed = `
import json, sys, io
from pathlib import Path
from PIL import Image
from app.sessions import Store
def png(c):
    b = io.BytesIO(); Image.new("RGB", (64, 64), c).save(b, format="PNG"); return b.getvalue()
st = Store(Path(sys.argv[1]))
s = st.create_session("combine smoke")
def mk(kind, color, model, parent=None, source=None):
    g = st.create_generation(session_id=s.id, kind=kind, model_key=model,
        prompt=kind, negative_prompt="", seed=5, width=512, height=512, steps=4,
        cfg=1.0, sampler=None, parent_id=parent, source_id=source)
    st.save_image(g, png(color)); st.set_status(g.id, "done"); return g.id
scene = mk("final", (0, 0, 255), sys.argv[2])
subject = mk("final", (255, 0, 0), sys.argv[2])
done = mk("edit", (0, 255, 0), sys.argv[3], parent=scene, source=subject)
print(json.dumps(dict(session=s.id, scene=scene, subject=subject, done=done)))
`;
const ids = JSON.parse(execFileSync("uv", ["run", "python", "-c", seed, dataDir, model.key, editor.key],
  { cwd: ROOT, encoding: "utf8" }));

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
page.on("console", m => process.env.SMOKE_DEBUG && console.log("console:", m.text()));

// the combine request is answered here, so nothing reaches Draw Things
let sent = null, upscaleSent = null;
await page.setRequestInterception(true);
page.on("request", req => {
  if (/\/api\/generations\/[^/]+\/upscale$/.test(req.url()) && req.method() === "POST") {
    upscaleSent = JSON.parse(req.postData() || "{}");
    return req.respond({ status: 400, contentType: "application/json",
                         body: JSON.stringify({ detail: "not in the smoke test" }) });
  }
  if (req.url().endsWith("/api/combine") && req.method() === "POST") {
    sent = JSON.parse(req.postData());
    return req.respond({ status: 200, contentType: "application/json", body: JSON.stringify({
      id: "fffffffffff1", session_id: ids.session, parent_id: sent.scene_id,
      source_id: sent.subject_id, kind: "edit", model_key: editor.key, prompt: sent.prompt,
      negative_prompt: "", seed: 1, width: 896, height: 896, steps: sent.steps, cfg: 1,
      sampler: null, status: "pending", created_at: new Date().toISOString(),
    })});
  }
  req.continue();
});

await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForSelector(`#sessionPicker option[value="${ids.session}"]`, { timeout: 30000 });
await page.select("#sessionPicker", ids.session);
await sleep(800);

const shown = sel => page.$eval(sel, el => el.style.display !== "none" && !el.hidden);
const pick = id => page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] .pickBtn`).click(), id);

check("no Combine button with nothing picked", !(await shown("#fCombine")));
await pick(ids.subject);
check("none with one picked", !(await shown("#fCombine")));
await pick(ids.scene);
check("offered with exactly two picked", await shown("#fCombine"));
await pick(ids.done);
check("not with three", !(await shown("#fCombine")));
await pick(ids.done);

await page.click("#fCombine");
await sleep(300);
const roles = () => page.evaluate(() => ({
  subject: document.querySelector("#combSubject figcaption").textContent,
  scene: document.querySelector("#combScene figcaption").textContent,
}));
let r = await roles();
check("the first pick is the subject", r.subject.includes(ids.subject.slice(0, 6))
  && r.scene.includes(ids.scene.slice(0, 6)), JSON.stringify(r));

await page.click("#combGo");
await sleep(200);
check("an empty instruction is not sent", sent === null);

await page.click("#combSwap");
r = await roles();
check("⇄ swaps them", r.subject.includes(ids.scene.slice(0, 6))
  && r.scene.includes(ids.subject.slice(0, 6)), JSON.stringify(r));
await page.click("#combSwap");

await page.type("#combText", "put her on the steps, standing");
await page.click("#combGo");
await sleep(500);
check("the request names both pictures the right way round",
  sent && sent.subject_id === ids.subject && sent.scene_id === ids.scene
  && sent.prompt === "put her on the steps, standing" && sent.steps === editor.final.steps, JSON.stringify(sent));
check("the dialog closed", !(await page.$eval("#viewer", v => v.open)));
check("a card for the queued combine was added",
  await page.evaluate(() => !!document.querySelector('#rounds .card[data-gen-id="fffffffffff1"]')));
check("the picks were cleared", !(await shown("#fCombine"))
  && !(await page.evaluate(() => document.querySelector("#rounds .card.picked"))));

// the finished combine from the seed
const doneCard = `#rounds .card[data-gen-id="${ids.done}"]`;
const meta = await page.$eval(`${doneCard} .meta`, el => el.textContent);
check("the combine card is labelled combine and names both origins",
  meta.includes("combine") && meta.includes(ids.scene.slice(0, 6))
  && meta.includes(ids.subject.slice(0, 6)), meta.replace(/\s+/g, " "));

await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="lineage"]`).click(), ids.subject);
await sleep(300);
const visible = await page.evaluate(() =>
  [...document.querySelectorAll("#rounds .card")]
    .filter(c => c.style.display !== "none").map(c => c.dataset.genId));
check("the subject's lineage includes the combine made from it",
  visible.includes(ids.done) && visible.includes(ids.subject), JSON.stringify(visible));

await page.evaluate(id => document.querySelector(
  `#rounds .card[data-gen-id="${id}"] [data-act="params"]`).click(), ids.done);
await sleep(500);
const params = await page.$eval("#viewer", el => el.innerText);
check("the settings name the subject's picture",
  params.includes("Subject taken from") && params.includes(ids.subject.slice(0, 6)), params.slice(0, 300));

// an instruction edit is upscaled by the checkpoint that drew it, never by the
// editor: the picker starts on that checkpoint and does not offer the editor
await page.keyboard.press("Escape");
const offer = await page.$eval(doneCard, c => ({
  options: [...(c.querySelector("select.upvia")?.options || [])].map(o => o.value),
  disabled: c.querySelector('[data-act="upscale"]').disabled,
}));
check("the edit's upscale is offered", !offer.disabled, JSON.stringify(offer));
check("the editor is not among the upscale models", !offer.options.includes(editor.key),
  JSON.stringify(offer.options));
check("the picker starts on the model that drew it",
  !offer.options.length || offer.options[0] === model.key, JSON.stringify(offer.options));
await page.evaluate(sel => document.querySelector(`${sel} [data-act="upscale"]`).click(), doneCard);
await sleep(300);
check("the upscale request names that model or leaves it to the server",
  upscaleSent && (upscaleSent.via === undefined || upscaleSent.via === model.key),
  JSON.stringify(upscaleSent));

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
