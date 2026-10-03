"""The auth area: passkey enrollment and sign-in (webauthn 3.x), server-side sessions and CSRF,
reviewer accounts, invitations, setup codes and break-glass, Process installations and service
keys.

Owner: the auth builder. Files: ``routes.py`` (handlers), ``backend.py`` (the ``AuthBackend`` the
principal guard calls on every request), ``cli.py`` (the host commands behind
``python -m call1.store setup-code`` and ``issue-service-key``), ``api.py`` (functions other areas
call), ``migrations/030_auth.sql`` (and 031-039).
"""
