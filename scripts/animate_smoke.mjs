/**
 * Smoke test for the Animate dialog's per-model clip lengths, in a real browser.
 *
 *   node scripts/animate_smoke.mjs
 *
 * Self-contained like lora_smoke.mjs: its own server on a spare port and a
 * throwaway data directory. Nothing renders: the dialog is opened on a stand-in
 * still, and the two requests it makes are answered in the browser.
 *
 * What only a browser can see: switching model rebuilds the length list from
 * that model's own grid (Wan 17/33/49/81 at 16 fps, H3 22/56/73/124 at 24),
 * keeps roughly the length that was picked, and sends the model's fps where
 * the motion prompt needs it. H3
 * is first in the config, so it is what the dialog opens on.
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

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "draftpool-animate-smoke-"));
const server = spawn("uv", ["run", "uvicorn", "app.main:app", "--port", PORT], {
  cwd: path.join(import.meta.dirname, ".."),
  env: { ...process.env, DRAFTPOOL_DATA_DIR: dataDir },
  stdio: "ignore",
});
process.on("exit", () => { server.kill(); fs.rmSync(dataDir, { recursive: true, force: true }); });

for (let i = 0; ; i++) {
  try { const r = await fetch(BASE + "/api/video-models"); if (r.ok) break; } catch {}
  if (i > 60) { console.error("server never came up"); process.exit(2); }
  await sleep(500);
}

const browser = await launchBrowser();
try {
  const page = await browser.newPage();
  const sent = [];
  await page.setRequestInterception(true);
  page.on("request", req => {
    const u = req.url();
    if (req.method() === "POST" && /\/api\/generations\/[^/]+\/animate(-prompt)?$/.test(u)) {
      sent.push({ url: u, body: JSON.parse(req.postData() || "{}") });
      const body = u.endsWith("-prompt")
        ? { prompt: "hair drifts", used_image: false }
        : { id: "clip00000000", kind: "video", status: "queued", num_frames: 56, fps: 24 };
      return req.respond({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    }
    req.continue();
  });
  await page.goto(BASE, { waitUntil: "networkidle0" });
  // both models offered whether or not this machine has them installed: the
  // dialog's behaviour is the question, not Draw Things' folder
  const keys = await page.evaluate(() => {
    VIDEO_MODELS = ALL_VIDEO_MODELS.filter(v => ["fusionx", "minimax-h3"].includes(v.key));
    // addRound draws a card into a session; there is none in a throwaway pool
    window.addRound = () => {};
    showAnimate({ id: "abcdef123456", width: 1024, height: 1024, prompt: "a girl" });
    return VIDEO_MODELS.map(v => v.key);
  });
  check("H3 first, so it is the default", keys.join() === "minimax-h3,fusionx", keys.join());
  check("the dialog opens on H3", await page.$eval("#animModel", e => e.value) === "minimax-h3");

  const lengths = () => page.evaluate(() =>
    [...document.querySelectorAll("#animLen option")].map(o => [Number(o.value), o.selected, o.textContent]));
  let opts = await lengths();
  check("H3 offers its own grid", opts.map(o => o[0]).join() === "22,56,73,124", opts.map(o => o[0]).join());
  check("H3 defaults to about 2s", opts.find(o => o[1])?.[0] === 56, opts.find(o => o[1])?.[2]);
  check("labels in H3's seconds", opts.map(o => o[2]).includes("5.2s (124f)"), opts.map(o => o[2]).join(" | "));
  const est = await page.$eval("#animEst", e => e.textContent);
  check("an estimate is shown for H3", /Estimated \d+ minute/.test(est), est);

  await page.select("#animModel", "fusionx");
  opts = await lengths();
  check("FusionX keeps Wan's lengths", opts.map(o => o[0]).join() === "17,33,49,81", opts.map(o => o[0]).join());
  check("switching keeps about the same length", opts.find(o => o[1])?.[0] === 33, opts.find(o => o[1])?.[2]);

  await page.select("#animLen", "81");
  await page.select("#animModel", "minimax-h3");
  opts = await lengths();
  check("back to H3 keeps the long clip", opts.find(o => o[1])?.[0] === 124);

  await page.click("#animImprove");
  await sleep(300);
  await page.click("#animGo");
  await sleep(300);
  const imp = sent.find(s => s.url.endsWith("-prompt"));
  const go = sent.find(s => s.url.endsWith("/animate"));
  check("Improve sends H3's model and fps", imp?.body.model_key === "minimax-h3" && imp?.body.fps === 24,
        JSON.stringify(imp?.body));
  // the clip's fps is the model's, decided by the server: the page sends none
  check("Animate sends H3's length and no fps", go?.body.num_frames === 124
        && !("fps" in (go?.body || {})) && go?.body.model_key === "minimax-h3", JSON.stringify(go?.body));
} finally {
  await browser.close();
}
console.log(failed ? `${failed} FAILED` : "all passed");
process.exit(failed ? 1 : 0);
