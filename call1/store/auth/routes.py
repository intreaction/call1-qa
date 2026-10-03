"""Auth-area handlers. Every contract operation this area owns is registered on ``router``.

Owned operations (``call1.store.routing._AUTH``): enrollBegin, enrollFinish, signInBegin,
signInFinish, getSession, signOut, addAuthenticatorBegin, addAuthenticatorFinish,
listOwnAuthenticators, renameOwnAuthenticator, removeOwnAuthenticator, listOwnSessions,
revokeOwnSession (``routes_self``); listAccounts, getAccount, updateAccount,
listAccountAuthenticators, revokeAccountAuthenticator, revokeAccountSessions, createInvitation,
listInvitations, revokeInvitation, listSetupCodes, listBreakGlass, registerInstallation,
listInstallations, retireInstallation, createServiceKey, listServiceKeys, rotateServiceKey,
revokeServiceKey (``routes_admin``).

The session cookie is set and cleared with ``principals.set_session_cookie`` /
``clear_session_cookie`` (they apply the dev-mode cookie name). The relying party is
``store.config.rp_id`` and the accepted origins ``store.config.allowed_origins``.
"""

from call1.store.routing import Area, AreaRouter

router = AreaRouter(Area.AUTH)

from . import routes_admin, routes_self  # noqa: E402,F401  (registers the handlers on ``router``)
