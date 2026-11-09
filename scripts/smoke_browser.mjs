/**
 * The browser every smoke script drives, found the same way for all of them.
 *
 *   import { launchBrowser } from "./smoke_browser.mjs";
 *   const browser = await launchBrowser();
 *
 * Puppeteer: the current `puppeteer` package first, wherever Node resolves it
 * (`npm i -g puppeteer`, or one in a parent folder), and the copy bundled with
 * website-scraper-puppeteer only as a last resort -- that one is pinned to an
 * old release whose Chrome predates `:popover-open`, which the page uses for
 * its toasts, and every smoke that showed a toast failed on it.
 *
 * Chrome: $CHROME_PATH if set, else the installed Google Chrome, else the
 * browser Puppeteer downloaded. The installed one updates itself, so the smokes
 * run on what a user runs rather than on a copy frozen at install time.
 */
import { createRequire } from "node:module";
import fs from "node:fs";

const require = createRequire(import.meta.url);

const PUPPETEER = [
  "puppeteer",
  "/opt/homebrew/lib/node_modules/puppeteer",
  "/opt/homebrew/lib/node_modules/website-scraper-puppeteer/node_modules/puppeteer",
];

const CHROME = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];

export function loadPuppeteer() {
  for (const p of PUPPETEER) {
    try { return require(p); } catch {}
  }
  console.error("puppeteer not found; npm i -g puppeteer");
  process.exit(2);
}

export async function launchBrowser(options = {}) {
  const puppeteer = loadPuppeteer();
  const executablePath = CHROME.find(p => p && fs.existsSync(p));
  return puppeteer.launch({
    headless: true,
    args: ["--no-sandbox"],
    ...(executablePath ? { executablePath } : {}),
    ...options,
  });
}
