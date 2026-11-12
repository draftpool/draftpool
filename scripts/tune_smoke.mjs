/**
 * Smoke test for the Tune wizard in the ⚙ dialog, in a real browser.
 *
 *   node scripts/tune_smoke.mjs
 *
 * Self-contained like civitai_smoke.mjs: its own server on a spare port and a
 * throwaway data directory, with the startup step off (DRAFTPOOL_NO_MIGRATE).
 * Every /api/tune request is answered in the browser by a small fake of the
 * server's rounds, so nothing renders and nothing is written to this
 * checkout's config. tests/test_tune_api.py covers the real endpoints; this
 * covers what only a page can get wrong.
 *
 * What it checks: the button is on the tunable row only; a first click starts
 * a tuning and a later one continues it; the rendering line polls until the
 * round can be judged; a failed round offers its renders again; the step
 * line counts the rounds the server says this tuning has (the kind-of-model
 * round is only there for some); a picture
 * click sends its index as a number and "Can't tell" sends the word; a frame
 * in a count question is not itself an answer; the steps round can be
 * skipped; the review shows what changed and Save sends it and closes. That
 * no picture's setting is named before the review is the server's to keep
 * (tests/test_tune_api.py), so it is not faked here.
 */
const { spawn } = await import("node:child_process");
const fs = await import("node:fs");
const os = await import("node:os");
const path = await import("node:path");

const { launchBrowser } = await import("./smoke_browser.mjs");

const PORT = process.argv[2] || "8575";
const BASE = `http://127.0.0.1:${PORT}`;

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-tune-smoke-"));
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

// a 64px noise PNG for every frame: the page only has to show it
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAQ0lEQVR4nO3PQQ0AIBDAsIF/z4cK" +
  "PvTWBuvNbB9aD/AHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8xgBnNgFwVHlrYQAAAABJ" +
  "RU5ErkJggg==", "base64");

// ---- a fake of app/tune.py's rounds, as the endpoints report them ----------
const cells = (prefix, n, status = "done", w = 768) => Array.from({ length: n }, (_, i) => (
  { id: `${prefix}_${i}`, status, error: status === "error" ? "came back as one flat colour" : null,
    width: w, height: w }));
const base = { key: target.key, label: target.label, attempt: 1, notes: [], skipped: [],
  rounds: ["class", "sampler", "size", "steps"],
  before: { sampler: "DPM++ 2M Karras", cfg: 5 }, settings: {}, labels: {}, changes: {},
  running: false, estimate_s: null };
let state = null;        // null: not being tuned
let polls = 0;
const sent = [];

function sampler(phase) {
  return { ...base, round: "sampler", phase, pending: phase === "rendering" ? 5 : 0,
    estimate_s: phase === "rendering" ? 150 : null,
    questions: [1111, 2222].map(seed => ({ id: `sampler_${seed}`, round: "sampler", kind: "pick",
      crop: true, answer: null,
      cells: cells(`s${seed}`, 4, phase === "failed" ? "error" : "done") })) };
}
const klass = () => ({ ...base, round: "class", phase: "asking", pending: 0,
  questions: [1111, 2222].map(seed => ({ id: `class_${seed}`, round: "class", kind: "pick",
    crop: false, answer: null, cells: cells(`k${seed}`, 3, "done", 768) })) });
const size = () => ({ ...base, round: "size", phase: "asking", pending: 0,
  questions: ["a", "b", "c"].map(x => ({ id: `size_${x}`, round: "size", kind: "count",
    crop: false, answer: null, cells: cells(`z${x}`, 1, "done", 1536) })) });
const steps = () => ({ ...base, round: "steps", phase: "asking", pending: 0,
  questions: [1111, 2222].map(seed => ({ id: `steps_${seed}`, round: "steps", kind: "ab",
    crop: true, answer: null, cells: cells(`t${seed}`, 2, "done", 1536) })) });
const review = () => ({ ...base, round: null, phase: "review", pending: 0, questions: [],
  notes: ["Sampler and cfg: Euler a, cfg 3.5, picked on both seeds over DPM++ 2M Karras, cfg 5."],
  changes: { sampler: ["DPM++ 2M Karras", "Euler a"], cfg: [5, 3.5] },
  labels: { s1111_0: "Euler a, cfg 3.5" } });

const browser = await launchBrowser();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", e => { errors.push(String(e)); console.log("PAGEERROR", String(e)); });
await page.setRequestInterception(true);
const json = (req, body, status = 200) =>
  req.respond({ status, contentType: "application/json", body: JSON.stringify(body) });
page.on("request", async req => {
  const url = new URL(req.url());
  const p = url.pathname, m = req.method();
  if (p === "/api/models" && m === "GET") {
    const rows = await api("/api/models");
    for (const r of rows) {
      r.can_tune = r.key === target.key;
      r.tuning = r.key === target.key && state !== null;
    }
    return json(req, rows);
  }
  const prefix = `/api/tune/${target.key}`;
  if (!p.startsWith("/api/tune/")) return req.continue();
  if (p.startsWith(prefix + "/cells/")) {
    return req.respond({ status: 200, contentType: "image/png", body: PNG });
  }
  sent.push({ m, p: p.slice(prefix.length), body: req.postData() ? JSON.parse(req.postData()) : null });
  if (p === prefix && m === "GET") {
    if (state === null) return json(req, { detail: "this model is not being tuned" }, 404);
    // rendering -> failed -> (retry) rendering -> asking, a poll at a time
    if (state.phase === "rendering" && ++polls >= 2) state = state.retried ? klass() : sampler("failed");
    return json(req, state);
  }
  if (p === prefix && m === "POST") { state = sampler("rendering"); return json(req, state); }
  if (p === prefix + "/retry") { state = { ...sampler("rendering"), retried: true }; polls = 0; return json(req, state); }
  if (p === prefix + "/answer") {
    const { question, answer } = JSON.parse(req.postData());
    const q = state.questions.find(q => q.id === question);
    q.answer = answer;
    if (state.questions.every(q => q.answer !== null)) {
      state = state.round === "class" ? sampler("asking") : state.round === "sampler" ? size()
        : state.round === "size" ? steps() : review();
    }
    return json(req, state);
  }
  if (p === prefix + "/skip") { state = review(); return json(req, state); }
  if (p === prefix + "/apply") {
    state = null;
    return json(req, { message: `${target.label}: sampler DPM++ 2M Karras → Euler a, cfg 5 → 3.5.` });
  }
  if (p === prefix && m === "DELETE") { state = null; return json(req, { discarded: target.key }); }
  return json(req, { detail: "the fake does not know " + p }, 500);
});

await page.goto(BASE + "/", { waitUntil: "networkidle2" });
await page.waitForFunction(() => typeof MODELS !== "undefined" && MODELS.length > 0);
await page.click("#stepsBtn");
const rowSel = `#stepsRows tr[data-key="${target.key}"]`;
await page.waitForSelector(`${rowSel} .tuneBtn`);
const buttons = await page.$$eval("#stepsRows .tuneBtn", els => els.length);
check("the Tune button is on the tunable row only", buttons === 1, `${buttons}`);

// found and clicked in one step: the table is redrawn when the model list
// arrives, and a button found before that is gone by the time it is clicked
await page.$eval(`${rowSel} .tuneBtn`, b => b.click());
await page.waitForFunction(() => document.querySelector("#tuneDlg").open);
const firstLine = await page.$eval("#tuneBody", el => el.textContent);
check("a first click starts a tuning (GET 404, then POST)",
  sent[0]?.m === "GET" && sent[1]?.m === "POST" && sent[1].p === "", JSON.stringify(sent.slice(0, 2)));
check("the rendering line says how far and how long", /Rendering 4 of 8/.test(firstLine) && /about 3 min left/.test(firstLine),
  firstLine.trim().slice(0, 90));

await page.waitForSelector("#tuneRetry", { timeout: 15000 });
check("a round with nothing to judge offers its renders again",
  (await page.$eval("#tuneBody", el => el.textContent)).includes("came back as one flat colour"));
await page.click("#tuneRetry");
await page.waitForSelector("#tuneBody .tuneOpt img", { timeout: 15000 });
check("after the retry it polls until the round can be judged",
  sent.some(s => s.p === "/retry") && await page.$$eval("#tuneBody .tuneOpt img.whole", els => els.length) === 6);
check("the step line counts the rounds this tuning has",
  (await page.$eval("#tuneStep", el => el.textContent)).includes("Round 1 of 4 · Kind of model"));

// leave and come back: the row says it is unfinished, the click continues it
await page.click("#tuneClose");
await page.waitForFunction(sel => document.querySelector(`${sel} .tuneBtn`)?.textContent.includes("Continue"),
  { timeout: 5000 }, rowSel);
check("a closed tuning is offered to continue", true);
const before = sent.length;
// found and clicked in one step: the table is redrawn when the model list
// arrives, and a button found before that is gone by the time it is clicked
await page.$eval(`${rowSel} .tuneBtn`, b => b.click());
await page.waitForSelector("#tuneBody .tuneOpt img");
check("continuing does not start a new one", !sent.slice(before).some(s => s.m === "POST"),
  JSON.stringify(sent.slice(before)));

// the kind-of-model round: whole frames, picked like any other
check("the kind round asks which looks right",
  (await page.$eval("#tuneBody", el => el.textContent)).includes("which looks right"));
await page.click('#tuneBody [data-q="class_1111"] .tuneOpt[data-answer="1"]');
await page.waitForFunction(() => TUNE.questions[0].answer === 1);
await page.click('#tuneBody [data-q="class_2222"] .tuneOpt[data-answer="1"]');
await page.waitForFunction(() => TUNE.round === "sampler");
check("the step line moves on", (await page.$eval("#tuneStep", el => el.textContent)).includes("Round 2 of 4"));
sent.splice(0, sent.length, ...sent.filter(s => s.p !== "/answer"));

// the sampler round: a picture on the first seed, "Can't tell" on the second
await page.click('#tuneBody [data-q="sampler_1111"] .tuneOpt[data-answer="2"]');
await page.waitForFunction(() => TUNE.questions[0].answer === 2);
await page.click('#tuneBody [data-q="sampler_2222"] button[data-answer="cant_tell"]');
await page.waitForFunction(() => TUNE.round === "size");
const answers = sent.filter(s => s.p === "/answer").map(s => s.body);
check("a picture sends its index as a number, the button its word",
  answers[0]?.answer === 2 && answers[1]?.answer === "cant_tell", JSON.stringify(answers));

// the size round: frames whole, and only the buttons answer
check("a count shows whole frames", await page.$$eval("#tuneBody img.whole", els => els.length) === 3);
await page.click('#tuneBody [data-q="size_a"] .tuneOpt');
await sleep(150);
check("clicking the frame in a count question answers nothing",
  sent.filter(s => s.p === "/answer").length === 2);
for (const q of ["size_a", "size_b", "size_c"]) {
  await page.click(`#tuneBody [data-q="${q}"] button[data-answer="one"]`);
  await sleep(80);
}
await page.waitForFunction(() => TUNE.round === "steps");
check("round 3 can be skipped", !!(await page.$("#tuneSkip")));
await page.click("#tuneSkip");
await page.waitForSelector("#tuneApply");
const reviewText = await page.$eval("#tuneBody", el => el.textContent);
check("the review lists what changed", reviewText.includes("Euler a") && reviewText.includes("3.5"),
  reviewText.trim().slice(0, 120));

await page.click("#tuneApply");
await page.waitForFunction(() => !document.querySelector("#tuneDlg").open);
const toastText = await page.evaluate(() => [...document.querySelectorAll(".toast, #toast")]
  .map(t => t.textContent).join(" | "));
check("Save sends the picks, closes, and says what changed",
  sent.some(s => s.p === "/apply") && toastText.includes("cfg 5 → 3.5"), toastText);
await page.waitForFunction(sel => document.querySelector(`${sel} .tuneBtn`)?.textContent.includes("Tune"),
  { timeout: 5000 }, rowSel);
check("the row goes back to Tune once saved", true);

check("no page errors", !errors.length, errors.join(" | "));
await browser.close();
console.log(failed ? `${failed} failed` : "all passed");
process.exit(failed ? 1 : 0);
