// @ts-check
const { defineConfig } = require('@playwright/test');

/**
 * Playwright configuration for Streamlit CV app e2e tests.
 *
 * Usage:
 *   npx playwright test              # run all tests
 *   npx playwright test --headed     # run with visible browser
 *   npx playwright test --ui         # interactive UI mode
 */
module.exports = defineConfig({
  testDir: './tests',
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  fullyParallel: false,
  retries: 1,
  reporter: [['html', { open: 'never' }], ['list']],
  use: {
    baseURL: process.env.STREAMLIT_URL || 'http://localhost:8501',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
});
