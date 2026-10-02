/**
 * Smoke test for what a session switch does to the composer, in a real browser.
 *
 *   node scripts/session_smoke.mjs
 *
 * Self-contained: it starts its own server on a spare port against a throwaway
 * data directory, seeds the sessions it needs and cleans both up. Needs `uv`
 * and puppeteer (the copy that comes with a global website-scraper-puppeteer,
 * or a plain `puppeteer` resolvable from here).
 *
 * It exists because the composer is the one part of the page that a session
 * switch has to *empty* as well as fill, and nothing on the server can see
 * that. The bug it was written for: switching to a session that saved no
 * enriched blocks -- which is every session generated straight from the box at
 * the top -- left the previous session's blocks standing, so pressing Generate
 * queued the other session's prompts against this session's models.
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

const PORT = process.argv[2] || "8571";
const BASE = `http://127.0.0.1:${PORT}`;

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};
const sleep = ms => new Promise(r => setTimeout(r, ms));

// its own pool, so a run never lands test sessions in the picker of the one
// being worked in
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

const api = async (p, init) => {
  const r = await fetch(BASE + p, {
    ...init,
    headers: init?.body ? { "content-type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${p}: ${await r.text()}`);
  return r.json();
};

for (let i = 0; ; i++) {
  try { await api("/api/models"); break; } catch (e) {
    if (i > 60) { console.error("server never came up:", e.message); process.exit(2); }
    await sleep(500);
  }
}

// Which models exist is a property of the machine, not of this test: it needs
// one that reads a negative prompt and one that does not, and says so rather
// than failing obscurely if the install has neither.
const models = await api("/api/models");
const usable = models.filter(m => m.available && m.shown && !m.edit_only);
const withNeg = usable.find(m => m.uses_negative_prompt);
const plain = usable.find(m => !m.uses_negative_prompt) || withNeg;
if (!withNeg) {
  console.error("no installed model reads a negative prompt; nothing to test");
  process.exit(2);
}

const newSession = async (instruction, prompts) => {
  const { id } = await api("/api/sessions", { method: "POST", body: { instruction } });
  if (prompts) await api(`/api/sessions/${id}/prompts`, { method: "PUT", body: { instruction, prompts } });
  return id;
};

// A: Enrich was run, so the session owns a block of written prompts.
const A_TEXT = "a girl in a red coat";
const enrichedState = {
  models: [withNeg.key],
  negative: "blurry, watermark",
  groups: [{
    style: withNeg.prompt_style === "tags" ? "tags" : "prose",
    usesNeg: true,
    keys: [withNeg.key],
    variants: [
      { prompt: "ENRICHED ONE", negative_prompt: "blurry, watermark", label: "V1" },
      { prompt: "ENRICHED TWO", negative_prompt: "blurry, watermark", label: "V2" },
    ],
  }],
};
const enriched = await newSession(A_TEXT, enrichedState);
// B: generated straight from the box at the top, which saves a selection and a
// negative and no blocks at all. This is the session the bug was visible in.
const B_TEXT = "a red cube on a wooden table";
const plainSession = await newSession(B_TEXT, {
  models: [plain.key], negative: "text, logo", groups: [],
});
// C: older than the composer being saved at all -- nothing but an instruction.
const C_TEXT = "castle in the sky";
const bare = await newSession(C_TEXT, null);

const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });
const errors = [];
page.on("pageerror", e => errors.push(String(e)));
await page.goto(BASE + "/", { waitUntil: "networkidle2" });

const composer = () => page.evaluate(() => ({
  instruction: document.querySelector("#instruction").value,
  written: [...document.querySelectorAll("#variants .vgroup .vprompt")].map(t => t.value),
  negative: document.querySelector("#negative").value,
  // what Generate would queue right now, which is the half of this that bites
  queued: (composerRequests().requests || [])
    .flatMap(r => r.variants.map(v => v.prompt)),
}));
const open = async id => { await page.select("#sessionPicker", id); await sleep(600); };

await open(enriched);
let c = await composer();
check("an enriched session reopens with its prompts",
  c.instruction === A_TEXT && c.written.join("|") === "ENRICHED ONE|ENRICHED TWO",
  JSON.stringify(c.written));
check("an enriched session reopens with its negative",
  c.negative === "blurry, watermark", c.negative);

await open(plainSession);
c = await composer();
check("a session with no written prompts shows none",
  c.written.length === 0, JSON.stringify(c.written));
check("the instruction is this session's", c.instruction === B_TEXT, c.instruction);
check("the negative is this session's", c.negative === "text, logo", c.negative);
check("Generate would queue this session's prompt and nothing else",
  c.queued.length > 0 && c.queued.every(p => p.includes(B_TEXT)),
  JSON.stringify(c.queued).slice(0, 160));

await open(enriched);
c = await composer();
check("switching back restores the written prompts again",
  c.written.join("|") === "ENRICHED ONE|ENRICHED TWO", JSON.stringify(c.written));

// The box at the top is the original prompt, the only copy of it. Enrich used
// to copy it into a "your prompt" card below, and Generate sent the card: edit
// the top box afterwards and the two drifted apart, with the stale one queued
// and the count beside Generate calling it "1 prompt" as if nothing was wrong.
const setTop = text => page.evaluate(t => {
  const el = document.querySelector("#instruction");
  el.value = t;
  el.dispatchEvent(new Event("input", { bubbles: true }));
}, text);
await setTop("THE TOP BOX, EDITED");
c = await composer();
check("Generate sends the top box as it stands, next to the written prompts",
  c.queued.some(p => p.includes("THE TOP BOX, EDITED")) && !c.queued.some(p => p.includes(A_TEXT))
    && c.queued.includes("ENRICHED ONE") && c.queued.includes("ENRICHED TWO"),
  JSON.stringify(c.queued).slice(0, 200));
check("the negative box stays up after Enrich, as the original's negative",
  !(await page.$eval("#negative", el => el.hidden)));
const countText = () => page.$eval("#genCount", el => el.textContent);
check("the count names the prompts it is counting",
  /orig/.test(await countText()) && /V1/.test(await countText()), await countText());
// leaving the original out of the round, from its own line under the box
const origLine = () => page.evaluate(() => ({
  shown: !document.querySelector("#origLine").hidden,
  text: document.querySelector("#origText").textContent,
  on: document.querySelector("#includeOrig").checked,
}));
let o = await origLine();
check("with written prompts on screen, the original has a line of its own",
  o.shown && o.on && /every selected model/.test(o.text), JSON.stringify(o));
await page.click("#origToggle");
c = await composer();
check("its x leaves only the written ones",
  c.queued.join("|") === "ENRICHED ONE|ENRICHED TWO", JSON.stringify(c.queued));
check("and the count follows", !/orig/.test(await countText()), await countText());
check("and the line says it is left out", /left out/.test((await origLine()).text));
await page.click("#origToggle");
await setTop(A_TEXT);

// Enrich decides whether the original goes in: not when this session has
// already drafted it word for word, which is the round that queued 16 drafts
// of it next to the 4 of the one written prompt that was kept.
const enrichWithStub = () => page.evaluate(async () => {
  const real = window.api;
  window.api = (p, init) => p === "/api/enrich"
    ? Promise.resolve({ variants: [{ prompt: "STUB ONE", negative_prompt: "" }] })
    : real(p, init);
  try { await runEnrich(); } finally { window.api = real; }
});
await enrichWithStub();
check("Enrich keeps a never-drafted original in the round",
  (await origLine()).on && (await composer()).queued.some(p => p.includes(A_TEXT)));
await page.evaluate(key => genIndex.set("fake-orig", {
  id: "fake-orig", kind: "draft", variant_label: "orig", session_id: session.id,
  model_key: key, prompt: document.querySelector("#instruction").value.trim(),
}), withNeg.key);
await enrichWithStub();
o = await origLine();
c = await composer();
check("Enrich leaves an already-drafted original out",
  !o.on && !c.queued.some(p => p.includes(A_TEXT)) && c.queued.includes("STUB ONE"),
  JSON.stringify({ o, queued: c.queued }));
check("and says how often it was drafted", /drafted 1× in this session/.test(o.text), o.text);
await page.click("#origToggle");
check("+ add puts it back", (await composer()).queued.some(p => p.includes(A_TEXT)));
await api(`/api/sessions/${enriched}/prompts`, { method: "PUT", body: { instruction: A_TEXT, prompts: enrichedState } });
await open(plainSession);
check("a session with no written prompts shows no orig line",
  await page.$eval("#origLine", el => el.hidden));
await open(enriched);

// Removing a prompt has to reach the session, not just the screen. Enrich and
// Generate used to be the only savers, so a drop in a reopened session was
// undone by the next visit to it -- and a prompt you cannot get rid of leaves
// the round's prompts saying one thing and the session another.
const dropLast = () => page.evaluate(() => {
  const b = document.querySelectorAll("#variants .vdrop");
  b[b.length - 1].click();
});
await dropLast();
await sleep(400);
c = await composer();
check("dropping a prompt takes it off the screen",
  c.written.join("|") === "ENRICHED ONE", JSON.stringify(c.written));
await open(plainSession); await open(enriched);
c = await composer();
check("and the drop survives reopening the session",
  c.written.join("|") === "ENRICHED ONE", JSON.stringify(c.written));

// the last one too: a block with nothing left in it goes, and what Generate
// will use is then the prompt at the top and only that
await dropLast();
await sleep(400);
await open(plainSession); await open(enriched);
c = await composer();
check("the last prompt can be dropped and stays dropped",
  c.written.length === 0, JSON.stringify(c.written));
check("with the blocks gone Generate falls back to the prompt at the top",
  c.queued.length > 0 && c.queued.every(p => p.includes(A_TEXT)),
  JSON.stringify(c.queued).slice(0, 160));

// put the blocks back for the checks below
await api(`/api/sessions/${enriched}/prompts`, { method: "PUT", body: { instruction: A_TEXT, prompts: enrichedState } });
await open(plainSession); await open(enriched);
c = await composer();
check("an edited prompt is saved too",
  c.written.join("|") === "ENRICHED ONE|ENRICHED TWO", JSON.stringify(c.written));
await page.evaluate(() => {
  const t = document.querySelector("#variants .vgroup .vprompt");
  t.value = "EDITED AND SAVED";
  t.dispatchEvent(new Event("input", { bubbles: true }));
});
await sleep(1200);
await open(plainSession); await open(enriched);
c = await composer();
check("an edited prompt survives reopening the session",
  c.written[0] === "EDITED AND SAVED", JSON.stringify(c.written));

// Tidying a stale prompt out of a session must not retitle it: the instruction
// box is the session's name in the picker, and someone drafting their next
// idea above a session they are only reading has not asked for that.
await page.evaluate(() => { document.querySelector("#instruction").value = "SOMETHING ELSE ENTIRELY"; });
await dropLast();
await sleep(400);
const titles = await api("/api/sessions");
check("a composer save leaves the session's name alone",
  titles.find(t => t.id === enriched).instruction === A_TEXT,
  titles.find(t => t.id === enriched).instruction);

// Cleanup redraws the open session through the same function, and that must
// not behave like a switch: the boxes are only written back by Enrich and by
// Generate, so a reset here throws away whatever was typed since.
await page.evaluate(() => {
  document.querySelector("#variants .vgroup .vprompt").value = "EDITED BY HAND";
  document.querySelector("#negative").value = "hand-typed negative";
});
await page.evaluate(id => loadSession(id), enriched);
await sleep(400);
c = await composer();
check("redrawing the open session keeps unsaved edits",
  c.written[0] === "EDITED BY HAND" && c.negative === "hand-typed negative",
  JSON.stringify([c.written[0], c.negative]));

await open(bare);
c = await composer();
check("a session saved before composers were stored comes back empty",
  c.written.length === 0 && c.negative === "",
  JSON.stringify({ written: c.written, negative: c.negative }));
check("and with its own instruction", c.instruction === C_TEXT, c.instruction);

// Rename: a name of its own, shown in the picker, and never written into the
// prompt box -- Enrich and Generate rewrite that, which would undo it.
const optionText = id => page.$eval(`#sessionPicker option[value="${id}"]`, o => o.textContent);
const rename = async name => {
  await page.click("#renameBtn");
  await sleep(300);
  await page.$eval("#renameInput", el => { el.value = ""; });
  if (name) await page.type("#renameInput", name);
  await page.click("#renameSave");
  await sleep(500);
};
await open(plainSession);
check("the rename button is live with a session open",
  await page.$eval("#renameBtn", b => !b.disabled));
await rename("My cube shots");
check("the picker shows the new name", (await optionText(plainSession)).includes("My cube shots"),
  await optionText(plainSession));
check("the picker still has the renamed session selected",
  await page.$eval("#sessionPicker", el => el.value) === plainSession);
c = await composer();
check("renaming leaves the prompt box alone", c.instruction === B_TEXT, c.instruction);
await page.reload({ waitUntil: "networkidle2" });
await page.waitForSelector(`#sessionPicker option[value="${plainSession}"]`);
check("the name survives a reload", (await optionText(plainSession)).includes("My cube shots"));
check("the rename button is off with no session open",
  await page.$eval("#renameBtn", b => b.disabled));
await open(plainSession);
await rename("");
check("clearing the name shows the prompt again",
  (await optionText(plainSession)).includes(B_TEXT) && !(await optionText(plainSession)).includes("My cube"),
  await optionText(plainSession));

// A session is created by the first Generate (or Enrich) on a fresh page. The
// picker has to show it straight away: it used to stay on "new session", and
// choosing "new session" again then did nothing, because the picker's value
// never changed -- the only way to start another session was to switch away
// and back. The drafts request is answered here, so nothing is rendered.
await page.reload({ waitUntil: "networkidle2" });
await page.setRequestInterception(true);
page.on("request", req => {
  if (/\/api\/sessions\/[^/]+\/drafts$/.test(req.url()) && req.method() === "POST")
    return req.respond({ status: 200, contentType: "application/json", body: "[]" });
  req.continue();
});
const FRESH = "a lighthouse in a storm";
await page.$eval("#instruction", el => { el.value = ""; });
await page.type("#instruction", FRESH);
await page.click("#generateBtn");
await sleep(800);
const created = await page.$eval("#sessionPicker", el => el.value);
check("a session made by Generate is selected in the picker at once",
  created !== "" && (await optionText(created)).includes(FRESH), created);
check("and can be renamed without switching first",
  await page.$eval("#renameBtn", b => !b.disabled));
// Not checked here: that choosing "new session" then starts one. puppeteer's
// select() fires change even when the value does not move, which a real
// browser does not -- so it would pass with the bug in place. The picker's
// value above is the thing that decides it.

console.log(errors.length ? "PAGE ERRORS:\n" + errors.join("\n") : "no page errors");
await browser.close();
process.exit(failed || errors.length ? 1 : 0);
