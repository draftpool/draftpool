/**
 * Smoke test for publishing several pictures as one post, in a real browser.
 *
 *   node scripts/publish_many_smoke.mjs
 *
 * Self-contained like combine_smoke.mjs: its own server on a spare port, a
 * throwaway data directory, rows seeded straight into the store, and a dummy
 * CIVITAI_TOKEN (a real variable wins over .env), so even a request that got
 * past the interception below would be refused by Civitai rather than post.
 * The publish request is answered inside the browser; the dialog's defaults
 * come from the real server.
 *
 * What it proves: the filter row offers Publish N from two picks and refuses
 * it while a draft is picked, the dialog lists the pictures in pick order with
 * the first as the cover, "to front" and ✕ reorder and drop, the request names
 * exactly that order, every card gets its published link, and the picks clear.
 */
const { spawn, execFileSync } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const { launchBrowser } = await import("./smoke_browser.mjs");

const PORT = process.argv[2] || "8574";
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(import.meta.dirname, "..");

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-pubmany-"));
const server = spawn("uv", ["run", "uvicorn", "app.main:app", "--port", PORT], {
  cwd: ROOT,
  env: { ...process.env, DRAFTPOOL_DATA_DIR: dataDir, CIVITAI_TOKEN: "smoke-test-not-a-key" },
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
const model = models.find(m => m.available && m.shown && !m.edit_only) || models.find(m => !m.edit_only);

// three finals and a draft, all finished
const seed = `
import json, sys, io
from pathlib import Path
from PIL import Image
from app.sessions import Store
def png(c):
    b = io.BytesIO(); Image.new("RGB", (64, 64), c).save(b, format="PNG"); return b.getvalue()
st = Store(Path(sys.argv[1]))
s = st.create_session("a fox in the snow")
def mk(kind, color, seed):
    g = st.create_generation(session_id=s.id, kind=kind, model_key=sys.argv[2],
        prompt="1girl, fox ears, snow", negative_prompt="", seed=seed, width=512,
        height=512, steps=4, cfg=1.0, sampler=None)
    st.save_image(g, png(color)); st.set_status(g.id, "done"); return g.id
a = mk("final", (200, 0, 0), 1)
b = mk("final", (0, 200, 0), 2)
c = mk("final", (0, 0, 200), 3)
d = mk("draft", (90, 90, 90), 4)
print(json.dumps(dict(session=s.id, a=a, b=b, c=c, d=d)))
`;
const ids = JSON.parse(execFileSync("uv", ["run", "python", "-c", seed, dataDir, model.key],
  { cwd: ROOT, encoding: "utf8" }));

const browser = await launchBrowser();
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
page.on("console", m => process.env.SMOKE_DEBUG && console.log("console:", m.text()));

// the publish is answered here: nothing reaches Civitai
let sent = null;
await page.setRequestInterception(true);
page.on("request", req => {
  if (req.url().endsWith("/api/publish") && req.method() === "POST") {
    sent = JSON.parse(req.postData());
    const now = new Date().toISOString();
    return req.respond({ status: 200, contentType: "application/json", body: JSON.stringify(
      sent.ids.map((id, i) => ({
        id, session_id: ids.session, parent_id: null, kind: "final", model_key: model.key,
        prompt: "1girl, fox ears, snow", negative_prompt: "", seed: i, width: 512, height: 512,
        steps: 4, cfg: 1, sampler: null, status: "done", created_at: now,
        published_at: now, published_post_id: 777,
      })))});
  }
  if (/\/api\/generations\/[^/]+\/publish$/.test(req.url())) {
    sent = "single";   // the wrong route entirely
    return req.respond({ status: 400, contentType: "application/json", body: '{"detail":"no"}' });
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
const pubBtn = () => page.$eval("#fPublish", b => ({ text: b.textContent, disabled: b.disabled,
  why: b.parentElement.title }));

check("no Publish N with nothing picked", !(await shown("#fPublishWrap")));
await pick(ids.b);
check("none with one picked -- that is the card's own Publish", !(await shown("#fPublishWrap")));
await pick(ids.a);
let b = await pubBtn();
check("offered from two", await shown("#fPublishWrap") && b.text === "Publish 2…" && !b.disabled, JSON.stringify(b));
await pick(ids.d);
b = await pubBtn();
check("refused while a draft is picked, saying why", b.disabled && /draft/.test(b.why), JSON.stringify(b));
await pick(ids.d);
await pick(ids.c);
b = await pubBtn();
check("three finals: Publish 3", b.text === "Publish 3…" && !b.disabled, JSON.stringify(b));

await page.click("#fPublish");
await page.waitForSelector("#pubManyStrip figure", { timeout: 10000 });
const strip = () => page.$$eval("#pubManyStrip figure", fs => fs.map(f => f.dataset.id));
check("the dialog lists them in pick order", JSON.stringify(await strip()) === JSON.stringify([ids.b, ids.a, ids.c]),
  JSON.stringify(await strip()));
const head = await page.$eval("#pubManyHead", h => h.textContent);
check("the heading counts them", head.includes("3 images"), head);
const title = await page.$eval("#pubTitle", t => t.value);
check("the title is prefilled from the session", /fox in the snow/i.test(title), title);
const paramsCount = await page.$$eval("#pubManyParams pre", ps => ps.length);
check("each picture shows what goes with it", paramsCount === 3, String(paramsCount));
const imgs = await page.$$eval("#pubManyStrip img", is => is.every(i => i.src.length > 0));
check("every picture has a thumbnail", imgs);

// to front, then drop one
await page.evaluate(id => document.querySelector(`#pubManyStrip figure[data-id="${id}"] .pubFront`).click(), ids.c);
check("to front makes it the cover", (await strip())[0] === ids.c, JSON.stringify(await strip()));
await page.evaluate(id => document.querySelector(`#pubManyStrip figure[data-id="${id}"] .pubDrop`).click(), ids.a);
check("✕ leaves it out", JSON.stringify(await strip()) === JSON.stringify([ids.c, ids.b]), JSON.stringify(await strip()));
check("no ✕ is offered at two -- a post of several needs several",
  (await page.$$("#pubManyStrip .pubDrop")).length === 0);

await page.click("#pubDraft");
await page.waitForFunction(() => /Draft created/.test(document.querySelector("#viewer").innerText), { timeout: 5000 });
check("the request names the post's pictures in its order",
  sent && sent !== "single" && JSON.stringify(sent.ids) === JSON.stringify([ids.c, ids.b]), JSON.stringify(sent));
check("it is a simple post unless asked otherwise", sent && sent.simple === true);
check("the title went with it", sent && /fox in the snow/i.test(sent.title), sent && sent.title);
const done = await page.$eval("#viewer", v => v.innerText);
check("the dialog says how many went into the post", /post 777 · 2 images/.test(done), done.slice(0, 120));
const linked = await page.evaluate(ids => ids.map(id =>
  !!document.querySelector(`#rounds .card[data-gen-id="${id}"] .pubLink`)), [ids.c, ids.b, ids.a]);
check("both cards carry the published link, the dropped one does not",
  linked[0] && linked[1] && !linked[2], JSON.stringify(linked));
check("the picks were cleared", !(await shown("#fPublishWrap"))
  && !(await page.evaluate(() => document.querySelector("#rounds .card.picked"))));

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
