"use strict";

const assert = require("assert");
const fs = require("fs");
const playwrightRoot = process.env.PLAYWRIGHT_ROOT
  || [
    "/tmp/fr3-playwright/node_modules/playwright-core",
    "/tmp/zdtaichu-playwright/node_modules/playwright-core",
  ].find((candidate) => fs.existsSync(candidate));
if (!playwrightRoot) throw new Error("playwright-core not found; set PLAYWRIGHT_ROOT");
const { chromium } = require(playwrightRoot);

const baseUrl = process.env.ONEFORWARD_VERIFY_URL || "http://127.0.0.1:8000/";
const screenshot = process.env.ONEFORWARD_SCREENSHOT || "docs/assets/playground.png";
const mobileScreenshot = process.env.ONEFORWARD_MOBILE_SCREENSHOT || null;
const onePixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAZElEQVR4nO3PQQ3AIADAQEADH/x7Q8dE8Lgs6Slo5z17/NnSAa8a0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0D4XbAGTUCB/GwAAAABJRU5ErkJggg==",
  "base64",
);

(async () => {
  const browser = await chromium.launch({
    executablePath: "/opt/google/chrome/chrome",
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  try {
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.locator("#status.ready").waitFor({ timeout: 20_000 });
    await page.locator("#media-input").setInputFiles({
      name: "visual-evidence.png",
      mimeType: "image/png",
      buffer: onePixelPng,
    });
    await page.locator(".media-card").waitFor();
    await page.locator("#run").click();
    await page.locator("#results:not([hidden])").waitFor({ timeout: 60_000 });
    assert.strictEqual(await page.locator(".media-card").count(), 1);
    await page.screenshot({ path: screenshot, fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    assert(overflow <= 1, `mobile page overflows by ${overflow}px`);
    if (mobileScreenshot) {
      await page.screenshot({ path: mobileScreenshot, fullPage: true });
    }
    assert.deepStrictEqual(errors, [], errors.join("\n"));
    process.stdout.write(JSON.stringify({
      baseUrl,
      screenshots: [screenshot, mobileScreenshot].filter(Boolean),
      consoleErrors: errors,
    }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
