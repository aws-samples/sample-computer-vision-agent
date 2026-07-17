// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

/**
 * E2E tests for the Streamlit Computer Vision app.
 *
 * Prerequisites:
 *   - Streamlit app running at http://localhost:8501
 *   - AWS credentials configured
 *   - Test image available at ../assets/test_image.png
 */

test.describe('Streamlit CV App', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for Streamlit to fully load
    await page.waitForSelector('[data-testid="stApp"]', {
      timeout: 30_000,
    });
  });

  test('app loads successfully', async ({ page }) => {
    // Verify the app title or header is present
    const appContent = page.locator('[data-testid="stApp"]');
    await expect(appContent).toBeVisible();

    // Check that the sidebar with model selector is present
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();
  });

  test('model selector is visible in sidebar', async ({ page }) => {
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();

    // Verify model selectbox is present
    const selectbox = sidebar.locator('[data-testid="stSelectbox"]');
    await expect(selectbox).toBeVisible();
  });

  test('file uploader is present', async ({ page }) => {
    const sidebar = page.locator('[data-testid="stSidebar"]');
    await expect(sidebar).toBeVisible();

    // Check for file uploader widget
    const uploader = sidebar.locator(
      '[data-testid="stFileUploader"]'
    );
    await expect(uploader).toBeVisible();
  });

  test('chat input is present', async ({ page }) => {
    // Streamlit chat input
    const chatInput = page.locator(
      '[data-testid="stChatInput"] textarea'
    );
    await expect(chatInput).toBeVisible();
  });

  test('can upload an image', async ({ page }) => {
    const sidebar = page.locator('[data-testid="stSidebar"]');
    const uploader = sidebar.locator(
      '[data-testid="stFileUploader"]'
    );
    await expect(uploader).toBeVisible();

    // Upload the test image
    const testImagePath = path.resolve(
      __dirname, '..', '..', 'assets', 'test_image.png'
    );
    const fileInput = uploader.locator('input[type="file"]');
    await fileInput.setInputFiles(testImagePath);

    // Wait for upload confirmation - Streamlit shows the filename
    await expect(
      sidebar.getByText('test_image.png')
    ).toBeVisible({ timeout: 15_000 });
  });

  test('can send a chat message', async ({ page }) => {
    const chatInput = page.locator(
      '[data-testid="stChatInput"] textarea'
    );
    await expect(chatInput).toBeVisible();

    // Type a message
    await chatInput.fill('hello');
    await chatInput.press('Enter');

    // Wait for assistant response to appear
    const assistantMessage = page.locator(
      '[data-testid="stChatMessage"]'
    ).last();
    await expect(assistantMessage).toBeVisible({ timeout: 30_000 });
  });

  test('upload image and ask about it', async ({ page }) => {
    // Upload image first
    const sidebar = page.locator('[data-testid="stSidebar"]');
    const uploader = sidebar.locator(
      '[data-testid="stFileUploader"]'
    );
    await expect(uploader).toBeVisible();

    const testImagePath = path.resolve(
      __dirname, '..', '..', 'assets', 'test_image.png'
    );
    const fileInput = uploader.locator('input[type="file"]');
    await fileInput.setInputFiles(testImagePath);

    // Wait for upload confirmation
    await expect(
      sidebar.getByText('test_image.png')
    ).toBeVisible({ timeout: 15_000 });

    // Ask about the image
    const chatInput = page.locator(
      '[data-testid="stChatInput"] textarea'
    );
    await chatInput.fill('describe this image');
    await chatInput.press('Enter');

    // Wait for a response (may take a while with Bedrock)
    const messages = page.locator(
      '[data-testid="stChatMessage"]'
    );
    await expect(messages.last()).toBeVisible({ timeout: 60_000 });

    // Verify the response contains some text content
    const lastMessage = messages.last();
    const responseText = await lastMessage.textContent();
    expect(responseText.length).toBeGreaterThan(10);
  });
});
