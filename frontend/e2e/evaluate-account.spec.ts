// evaluate_ui e9: #/account — authenticators and sessions self-service; a reviewer can view their
// authenticators/sessions and add a second authenticator via a step-up ceremony.
import { test, expect } from './fixtures';

test('e9: account page lists the current authenticator and session, and adds a second authenticator', async ({
  createInvitedUser,
  authenticator,
}) => {
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto('/#/account');
  await expect(reviewer.page.getByRole('heading', { name: 'Your account' })).toBeVisible();
  await expect(reviewer.page.getByText(reviewer.email)).toBeVisible();

  // One authenticator so far: the "only authenticator" warning and prompt banner show.
  await expect(reviewer.page.getByText('You have one authenticator.').first()).toBeVisible();
  await expect(reviewer.page.getByText('Used for this session')).toBeVisible();

  await expect(reviewer.page.getByText('This browser')).toBeVisible();

  // Step-up: re-authenticate with the existing authenticator, then register a NEW one. A WebAuthn
  // platform authenticator with a resident key returns/overwrites the SAME credential for a second
  // "create" ceremony on the same (rpId, user handle) — so registering a genuinely different
  // credential needs a second, distinct virtual authenticator "present" for that second ceremony
  // only (mirrors a reviewer plugging in a different physical key).
  // Chrome allows only one 'internal' virtual authenticator per environment, so the second (distinct
  // physical key) uses 'usb' transport, same as a real backup security key would.
  const cdp = await reviewer.page.context().newCDPSession(reviewer.page);
  const { authenticatorId: secondId } = await cdp.send('WebAuthn.addVirtualAuthenticator', {
    options: { protocol: 'ctap2', ctap2Version: 'ctap2_1', transport: 'usb', hasResidentKey: true, hasUserVerification: true, isUserVerified: true, automaticPresenceSimulation: false },
  });

  await reviewer.page.getByRole('button', { name: 'Add another authenticator' }).click();
  await reviewer.page.getByLabel('Name the new authenticator (optional)').fill('Backup key');
  await reviewer.page.getByRole('button', { name: 'Start' }).click(); // re-authenticates with the original authenticator
  await expect(reviewer.page.getByRole('button', { name: 'Register the new authenticator' })).toBeVisible({ timeout: 15_000 });

  await authenticator.setAutomaticPresence(false);
  await cdp.send('WebAuthn.setAutomaticPresenceSimulation', { authenticatorId: secondId, enabled: true });
  await reviewer.page.getByRole('button', { name: 'Register the new authenticator' }).click();
  await expect(reviewer.page.getByText('Added “Backup key”.')).toBeVisible({ timeout: 15_000 });
  await authenticator.setAutomaticPresence(true);

  // The prompt for a second authenticator is now satisfied.
  await reviewer.page.reload();
  await expect(reviewer.page.getByText('You have one authenticator.')).toHaveCount(0);
});
