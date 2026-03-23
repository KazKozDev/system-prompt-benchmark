const http = require("http");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const { chromium } = require("playwright");

const repoRoot = path.resolve(__dirname, "..", "..");
const repoVenvPython = path.join(repoRoot, ".venv", "bin", "python");
const targetPort = Number(process.env.SPB_SMOKE_PORT || 8502);
const targetUrl =
  process.env.SPB_SMOKE_URL || `http://127.0.0.1:${targetPort}`;
const pythonBin =
  process.env.SPB_SMOKE_PYTHON ||
  process.env.PYTHON ||
  (fs.existsSync(repoVenvPython) ? repoVenvPython : "python");
const startupTimeoutMs = 45000;

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForServer(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const isReady = await new Promise((resolve) => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      });
      req.on("error", () => resolve(false));
      req.setTimeout(2000, () => {
        req.destroy();
        resolve(false);
      });
    });

    if (isReady) {
      return;
    }

    await delay(500);
  }

  throw new Error(`Timed out waiting for Streamlit at ${url}`);
}

function startStreamlit() {
  const child = spawn(
    pythonBin,
    [
      "-m",
      "streamlit",
      "run",
      "app.py",
      "--server.headless",
      "true",
      "--server.port",
      String(targetPort),
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  child.stdout.on("data", (chunk) => {
    process.stdout.write(`[streamlit] ${chunk}`);
  });
  child.stderr.on("data", (chunk) => {
    process.stderr.write(`[streamlit] ${chunk}`);
  });

  return child;
}

async function assertVisible(page, textOrRegex) {
  const locator = page.getByText(textOrRegex, { exact: false }).first();
  await locator.waitFor({ state: "visible", timeout: 15000 });
}

async function visitTab(page, tabName) {
  await page.getByRole("tab", { name: tabName }).click();
  await page.waitForTimeout(800);
}

async function runSmoke() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1600 } });

  try {
    await page.goto(targetUrl, {
      waitUntil: "networkidle",
      timeout: 30000,
    });
    await page.waitForTimeout(1500);

    await page.getByText("Help", { exact: true }).first().click();
    await page.waitForTimeout(1200);
    await assertVisible(page, /^Help Center$/);
    await assertVisible(page, /How To Use The Benchmark/);

    await page.getByText("Admin", { exact: true }).first().click();
    await page.waitForTimeout(1200);

    const expectedTabs = [
      "Jobs",
      "Results",
      "Webhooks",
      "Redis",
      "Metrics",
      "Plugins",
      "Smoke Tools",
      "Presets & Datasets",
    ];
    for (const tabName of expectedTabs) {
      await page
        .getByRole("tab", { name: tabName })
        .waitFor({ state: "visible", timeout: 15000 });
    }

    await visitTab(page, "Metrics");
    await assertVisible(page, /^Runtime$/);
    await assertVisible(page, /Backend (inprocess|redis)/);
    await assertVisible(page, /Workers\s*\d+/);
    await assertVisible(page, /Redis (healthy|unhealthy|disabled)/);

    await visitTab(page, "Plugins");
    await assertVisible(page, /^Registry$/);
    await assertVisible(page, /Plugins\s*\d+/);
    await assertVisible(page, /Providers\s*\d+/);
    await assertVisible(page, /Transforms\s*\d+/);
    await assertVisible(page, /Judges\s*\d+/);
    await assertVisible(page, /Exporters\s*\d+/);

    await visitTab(page, "Webhooks");
    await assertVisible(page, /Failed webhook deliveries and replay operations\./);

    await visitTab(page, "Results");
    await assertVisible(page, /Completed benchmark results with summary drill-down\./);
    await page.getByRole("tab", { name: "Result Files" }).click();
    await page.waitForTimeout(800);
    await assertVisible(
      page,
      /Load standalone result JSON files for summary or side-by-side comparison\./,
    );
    await assertVisible(page, /^Summarize$/);
    await assertVisible(page, /^Compare$/);

    await visitTab(page, "Redis");
    await assertVisible(page, /Redis Stream pending entries and replay operations\./);

    const bodyText = await page.locator("body").innerText();
    if (bodyText.includes("Traceback") || bodyText.includes("Uncaught app exception")) {
      throw new Error("Admin smoke detected an application error in the UI");
    }

    console.log("Admin smoke passed.");
  } finally {
    await browser.close();
  }
}

async function main() {
  const streamlit = startStreamlit();
  let streamlitExitedEarly = false;

  streamlit.once("exit", (code) => {
    streamlitExitedEarly = true;
    if (code && code !== 0) {
      console.error(`Streamlit exited early with code ${code}`);
    }
  });

  try {
    await waitForServer(targetUrl, startupTimeoutMs);
    if (streamlitExitedEarly) {
      throw new Error("Streamlit exited before the smoke test began");
    }
    await runSmoke();
  } finally {
    if (!streamlit.killed) {
      streamlit.kill("SIGTERM");
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
