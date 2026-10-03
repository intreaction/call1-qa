// evaluate_ui e12: OfflineBanner / connectivity detection. When Store is unreachable, the banner
// shows, and it clears when connectivity returns.
import { test, expect } from './fixtures';

test('e12: the offline banner appears when Store becomes unreachable and clears when it recovers', async ({
  createInvitedUser,
}) => {
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByText('Offline', { exact: true })).toHaveCount(0);

  // Every request to Store fails at the network level (StoreClient.send's catch -> 'unreachable').
  await reviewer.context.route('**/store/v1/**', (route) => route.abort('connectionfailed'));
  await reviewer.page.getByRole('link', { name: 'Metrics' }).click();
  await expect(reviewer.page.getByText("Can't reach Store.", { exact: false })).toBeVisible({ timeout: 15_000 });

  // Restore connectivity: the next answered request (the change poller, within one interval, or an
  // explicit retry) clears the banner without a page reload.
  await reviewer.context.unroute('**/store/v1/**');
  await expect(reviewer.page.getByText('Offline', { exact: true })).toHaveCount(0, { timeout: 15_000 });
});
