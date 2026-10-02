/**
 * Smoke test for the Edit dialog and the comparison grid, in a real browser.
 *
 *   node scripts/ui_smoke.mjs [port] [session-id]
 *
 * Needs a running server with at least ten finished images in the session it
 * is pointed at, and puppeteer available (this looks for the copy that comes
 * with a global website-scraper-puppeteer, and falls back to a plain
 * `puppeteer` resolvable from here).
 *
 * It exists because the Edit dialog decides four things at once -- which modes
 * are offered, which engine they mean, which model runs it and at what
 * strength -- from two inputs, and that is not something reading the code
 * catches reliably. Written after two bugs it found on its first run: a stale
 * engine read before the mode fallback, and controls that stayed visible
 * because an author `display` beats the browser's own rule for [hidden].
 */
const require = (await import("node:module")).createRequire(import.meta.url);
const PUPPETEER = [
  "/opt/homebrew/lib/node_modules/website-scraper-puppeteer/node_modules/puppeteer",
  "puppeteer",
];
let puppeteer = null;
for (const path of PUPPETEER) { try { puppeteer = require(path); break; } catch {} }
if (!puppeteer) { console.error("puppeteer not found; npm i -g puppeteer"); process.exit(2); }

const PORT = process.argv[2] || "8500";
const SESSION = process.argv[3] || "";

let failed = 0;
const check = (name, ok, extra = "") => {
  if (!ok) failed++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? "  " + extra : ""}`);
};

(async () => {
  const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  page.on("requestfailed", r => errors.push("requestfailed: " + r.url()));
  page.on("response", r => { if (r.status() >= 400) errors.push(r.status() + " " + r.url()); });

  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "networkidle2" });
  // load the session that has finished images
  const session = SESSION || await page.$eval("#sessionPicker",
    el => [...el.options].map(o => o.value).filter(Boolean)[0]);
  await page.select("#sessionPicker", session);
  await page.waitForFunction(() => document.querySelectorAll("#rounds .card").length > 3, { timeout: 20000 });

  const cards = await page.$$eval("#rounds .card", els => els.length);
  console.log("cards on screen:", cards);

  // --- the Edit dialog ----------------------------------------------------
  const opened = await page.evaluate(() => {
    const card = [...document.querySelectorAll("#rounds .card")]
      .find(c => c.querySelector('[data-act="edit"]'));
    if (!card) return "no card with an Edit button";
    card.querySelector('[data-act="edit"]').click();
    return null;
  });
  if (opened) { console.log("FAIL open:", opened); process.exit(1); }
  await page.waitForSelector("#editText", { timeout: 5000 });

  check("one Edit button per card, no Instruct button",
    await page.$$eval("#rounds .card", els =>
      els.every(c => !c.querySelector('[data-act="instruct"]'))));

  const state = () => page.evaluate(() => ({
    mode: document.querySelector('input[name="editMode"]:checked')?.value,
    visibleModes: [...document.querySelectorAll(".modeOpt")]
      .filter(l => getComputedStyle(l).display !== "none")
      .map(l => l.querySelector("input").id || l.querySelector("input").value),
    strengthShown: getComputedStyle(document.querySelector("#editStrengthLabel")).display !== "none",
    strength: document.querySelector("#editStrength").value,
    steps: document.querySelector("#editSteps").value,
    model: document.querySelector("#editModel").value,
    modelCount: document.querySelector("#editModel").options.length,
    refShown: getComputedStyle(document.querySelector("#editRef")).display !== "none",
    hint: document.querySelector("#editHint").textContent.slice(0, 60),
    warn: document.querySelector("#editWarn").textContent.trim().slice(0, 40),
  }));

  // 1. no rectangle: instruction to the edit model
  let s = await state();
  check("no rectangle -> instruct mode", s.mode === "instruct", JSON.stringify(s.visibleModes));
  check("no rectangle -> instruct model selected", s.model === "qwen-edit", s.model);
  check("no rectangle -> no strength dial", !s.strengthShown);
  check("no rectangle -> no reference button", !s.refShown);

  // 2. "another take on it" switches to a checkpoint and shows 0.8
  await page.evaluate(() => {
    document.querySelector('input[name="editMode"][value="vary"]').click();
  });
  s = await state();
  check("vary -> strength 0.8 on a checkpoint", s.strengthShown && s.strength === "0.8" && s.model !== "qwen-edit",
    `${s.strength} ${s.model}`);

  // 3. drag a rectangle on the picture
  const rect = await page.evaluate(() => {
    const img = document.querySelector("#editImg");
    const r = img.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  });
  await page.mouse.move(rect.x + rect.w * 0.2, rect.y + rect.h * 0.6);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.w * 0.5, rect.y + rect.h * 0.9, { steps: 12 });
  await page.mouse.up();
  await new Promise(r => setTimeout(r, 200));

  s = await state();
  check("rectangle -> change mode, region options only",
    s.mode === "change" && s.visibleModes.includes("change") && s.visibleModes.includes("add")
    && !s.visibleModes.includes("instruct"), JSON.stringify(s.visibleModes));
  check("change -> 0.7 on a checkpoint", s.strength === "0.7" && s.model !== "qwen-edit", `${s.strength} ${s.model}`);
  check("change -> reference button offered", s.refShown);
  check("change -> no resize warning with a rectangle", s.warn === "", s.warn);
  const regionHint = await page.$eval("#editRegionHint", e => e.textContent);
  check("rectangle hint quotes its size", /\d+×\d+px/.test(regionHint), regionHint.slice(0, 70));

  // 4. add -> the instruction editor places it, and the fast toggle appears
  await page.evaluate(() => document.querySelector('input[name="editMode"][value="add"]').click());
  s = await state();
  check("add -> instruction editor", s.model === "qwen-edit", s.model);
  const editorSteps = await page.evaluate(() => String(MODELS.find(m => m.key === "qwen-edit").final.steps));
  check("add -> the editor's own steps", s.steps === editorSteps, `${s.steps} vs ${editorSteps}`);
  check("add -> no strength dial", !s.strengthShown);
  check("add -> only the two region modes are offered",
    s.visibleModes.length === 2, JSON.stringify(s.visibleModes));
  const placeHint = await page.$eval("#editHint", e => e.textContent);
  check("add -> hint promises the rectangle is the frame and the boundary",
    /not one pixel outside changes/i.test(placeHint) && /whole space the new thing needs/i.test(placeHint),
    placeHint.slice(0, 80));
  check("no img2img route to an addition survives",
    await page.evaluate(() => !document.querySelector("#editFastAdd")
      && ![...document.querySelectorAll("#editStrength option")].some(o => o.value === "0.85")));

  // 5. clearing the rectangle returns the whole-picture options
  await page.mouse.click(rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
  await new Promise(r => setTimeout(r, 200));
  s = await state();
  check("cleared rectangle -> whole-picture options return",
    s.visibleModes.includes("instruct") && !s.visibleModes.includes("change"), JSON.stringify(s.visibleModes));
  check("cleared rectangle -> mode fell back to instruct", s.mode === "instruct", s.mode);

  await page.evaluate(() => document.querySelector("#editCancel").click());

  // --- the composer's negative prompt -------------------------------------
  // it exists for hand-written prompts, which is most rounds: before this
  // there was nowhere to type one unless Enrich had run and made the boxes
  const negHidden = () => page.evaluate(() => document.querySelector("#negative")?.hidden);

  // a resumed session restores its prompt cards; this path is the one without
  // them, which is what most rounds are
  await page.evaluate(() => { document.querySelector("#variants").innerHTML = ""; });
  await page.evaluate(() => [...selectedModels].forEach(k => chipEls.get(k)?.click()));
  const negModel = await page.evaluate(() => {
    const m = MODELS.find(x => x.available && x.shown && !x.edit_only && x.uses_negative_prompt);
    if (m) chipEls.get(m.key).click();
    return m ? m.key : null;
  });
  check("a model that reads a negative shows the box", negModel && (await negHidden()) === false, String(negModel));

  const plainModel = await page.evaluate(() => {
    [...selectedModels].forEach(k => chipEls.get(k)?.click());
    const m = MODELS.find(x => x.available && x.shown && !x.edit_only && !x.uses_negative_prompt);
    if (m) chipEls.get(m.key).click();
    return m ? m.key : null;
  });
  check("a model that does not read one hides it", plainModel && (await negHidden()) === true, String(plainModel));

  // back to the model that reads one, and check what the button actually sends
  await page.evaluate(() => {
    [...selectedModels].forEach(k => chipEls.get(k)?.click());
    const m = MODELS.find(x => x.available && x.shown && !x.edit_only && x.uses_negative_prompt);
    chipEls.get(m.key).click();
    document.querySelector("#instruction").value = "a girl in a red coat";
    document.querySelector("#negative").value = "watermark, extra fingers";
  });
  const sent = await page.evaluate(async () => {
    const real = window.fetch;
    let body = null;
    window.fetch = async (url, opts) => {          // capture rather than queue
      if (String(url).includes("/drafts")) {
        body = opts && opts.body ? JSON.parse(opts.body) : null;
        return new Response("[]", { status: 200, headers: { "content-type": "application/json" } });
      }
      return real(url, opts);
    };
    document.querySelector("#generateBtn").click();
    await new Promise(r => setTimeout(r, 800));
    window.fetch = real;
    return body;
  });
  const negSent = JSON.stringify(sent || {}).includes("watermark, extra fingers");
  check("a hand-written negative reaches the request", negSent,
    negSent ? "" : JSON.stringify(sent).slice(0, 200));

  // a mixed selection keeps the box and says who acts on it
  const mixed = await page.evaluate(() => {
    const plain = MODELS.find(x => x.available && x.shown && !x.edit_only && !x.uses_negative_prompt);
    chipEls.get(plain.key).click();               // add one that cannot use it
    renderPromptPlan();
    const hint = document.querySelector("#negativeHint");
    const reader = MODELS.find(x => selectedModels.has(x.key) && x.uses_negative_prompt);
    const out = { boxHidden: document.querySelector("#negative").hidden,
                  hint: hint.hidden ? "" : hint.textContent, plain: plain.label, reader: reader.label };
    chipEls.get(plain.key).click();               // put the selection back
    renderPromptPlan();
    return out;
  });
  // the hint names the readers only: the one that ignores it is not listed,
  // and the line says the rest do
  check("a mixed selection keeps the box and names who reads it",
    mixed.boxHidden === false && mixed.hint.includes(mixed.reader)
      && !mixed.hint.includes(mixed.plain) && /the rest ignore it/.test(mixed.hint),
    mixed.hint.slice(0, 90));

  // and with Enrich's own card on screen there is exactly one box, not two
  const oneBox = await page.evaluate(() => {
    const box = document.querySelector("#negative");
    const d = document.createElement("div");
    d.className = "variant"; d.dataset.original = "1";
    d.innerHTML = '<textarea class="vprompt"></textarea><textarea class="neg vneg"></textarea>';
    document.querySelector("#variants").appendChild(d);
    renderPromptPlan();
    const hidden = box.hidden;
    d.remove(); renderPromptPlan();
    return hidden;
  });
  check("the composer box steps aside for Enrich's own", oneBox === true, String(oneBox));

  // --- compare: ten panes -------------------------------------------------
  const picks = await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#rounds .card .pickBtn")].filter(b => !b.hidden);
    btns.slice(0, 12).forEach(b => b.click());   // two more than the maximum
    return document.querySelectorAll("#rounds .card.picked").length;
  });
  check("compare caps at ten picks", picks === 10, String(picks));
  await page.evaluate(() => document.querySelector("#fCompare").click());
  await page.waitForSelector(".cmpPane", { timeout: 5000 });
  const panes = await page.$$eval(".cmpPane", els => els.length);
  const cols = await page.$eval("#cmpPanes", el => getComputedStyle(el).gridTemplateColumns.split(" ").length);
  check("ten panes are built", panes === 10, String(panes));
  check("ten panes lay out four across", cols === 4, String(cols));

  // the favicon 404 is the page's own and predates this script
  const real = errors.filter(e => !e.includes("favicon.ico"));
  console.log(real.length ? "PAGE ERRORS:\n" + real.join("\n") : "no page errors");
  await browser.close();
  process.exit(failed || real.length ? 1 : 0);
})();
