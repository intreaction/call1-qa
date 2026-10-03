-- 030_auth: owned by the auth area (call1/store/auth/). Numbers 030-039 belong to the auth area.
-- Reviewer identity (passkey-only), server-side sessions, invitations, setup codes and break-glass,
-- Process installations and service keys. Store keeps hashes of every bearer secret (session
-- cookies, CSRF tokens, invitation tokens, setup codes, service keys), never the values. No
-- password verifier exists anywhere. Other areas reference these rows by ID only
-- (call1.store.auth.api), never by FOREIGN KEY.

-- Area-private random keys (decoy derivation, CSRF derivation). Generated on first use.
CREATE TABLE auth_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE auth_accounts (
    id                TEXT PRIMARY KEY,
    email             TEXT NOT NULL,
    email_key         TEXT NOT NULL UNIQUE,      -- lower-cased email: one account per address
    display_name      TEXT NOT NULL,
    role              TEXT NOT NULL CHECK (role IN ('reviewer', 'supervisor', 'admin')),
    status            TEXT NOT NULL CHECK (status IN ('pending_enrollment', 'active', 'disabled', 'reinvite_required')),
    user_handle       TEXT NOT NULL UNIQUE,      -- WebAuthn user.id, base64url random; never the email
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    last_sign_in_at   TEXT,
    legacy_auditor_id TEXT
);
CREATE INDEX auth_accounts_role_status ON auth_accounts (role, status);

-- WebAuthn credentials (attestation "none"; the public key is not secret). A revoked credential
-- keeps its row for the audit trail and never authenticates again.
CREATE TABLE auth_credentials (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES auth_accounts (id),
    credential_id   TEXT NOT NULL,               -- base64url WebAuthn credential ID
    public_key_cose TEXT NOT NULL,               -- base64url COSE key
    sign_count      INTEGER NOT NULL DEFAULT 0 CHECK (sign_count >= 0),
    transports_json TEXT NOT NULL DEFAULT '[]',
    aaguid          TEXT,
    nickname        TEXT,
    backup_eligible INTEGER,
    backup_state    INTEGER,
    created_at      TEXT NOT NULL,
    last_used_at    TEXT,
    revoked_at      TEXT,
    revoked_reason  TEXT
);
CREATE UNIQUE INDEX auth_credentials_live_credential_id ON auth_credentials (credential_id) WHERE revoked_at IS NULL;
CREATE INDEX auth_credentials_account ON auth_credentials (account_id, revoked_at);

-- Server-side sessions. The cookie carries a random secret kept here as sha256 only; the CSRF
-- token is derived from the cookie (HMAC under auth_settings.csrf_key) and kept as sha256 only.
CREATE TABLE auth_sessions (
    id                  TEXT PRIMARY KEY,
    account_id          TEXT NOT NULL REFERENCES auth_accounts (id),
    cookie_hash         TEXT NOT NULL UNIQUE,
    csrf_hash           TEXT NOT NULL,
    authenticator_id    TEXT NOT NULL REFERENCES auth_credentials (id),
    created_at          TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    idle_expires_at     TEXT NOT NULL,
    absolute_expires_at TEXT NOT NULL,
    revoked_at          TEXT,
    revoked_reason      TEXT
);
CREATE INDEX auth_sessions_account ON auth_sessions (account_id, revoked_at);

-- WebAuthn ceremonies: single-use challenges that expire after CHALLENGE_LIFETIME. A sign-in
-- ceremony for an unknown, disabled or re-invite-pending email has no account (decoys).
CREATE TABLE auth_ceremonies (
    id                TEXT PRIMARY KEY,
    kind              TEXT NOT NULL CHECK (kind IN ('enroll', 'sign_in', 'add_authenticator')),
    challenge         TEXT NOT NULL,             -- base64url; registration (enroll, add) or assertion (sign_in)
    step_up_challenge TEXT,                      -- add_authenticator: the re-authentication assertion challenge
    account_id        TEXT,                      -- the bound account (existing accounts only)
    session_id        TEXT,                      -- add_authenticator: the session that began it
    invitation_id     TEXT,
    setup_code_id     TEXT,
    new_account_id    TEXT,                      -- enroll of a new account: its ID and user handle
    new_user_handle   TEXT,
    nickname          TEXT,
    created_at        TEXT NOT NULL,
    expires_at        TEXT NOT NULL,
    used_at           TEXT
);
CREATE INDEX auth_ceremonies_expires ON auth_ceremonies (expires_at);

CREATE TABLE auth_invitations (
    id                     TEXT PRIMARY KEY,
    email                  TEXT NOT NULL,
    email_key              TEXT NOT NULL,
    display_name           TEXT NOT NULL,
    role                   TEXT NOT NULL CHECK (role IN ('reviewer', 'supervisor', 'admin')),
    status                 TEXT NOT NULL CHECK (status IN ('pending', 'redeemed', 'revoked')),  -- expired is derived
    delivery               TEXT NOT NULL,
    token_hash             TEXT NOT NULL UNIQUE,
    issued_by_account_id   TEXT,
    issued_at              TEXT NOT NULL,
    expires_at             TEXT NOT NULL,
    redeemed_at            TEXT,
    revoked_at             TEXT,
    account_id             TEXT,
    reinvite_of_account_id TEXT
);
CREATE INDEX auth_invitations_email ON auth_invitations (email_key, status);

-- First-admin and break-glass codes, issued only by the host command.
CREATE TABLE auth_setup_codes (
    id                       TEXT PRIMARY KEY,
    purpose                  TEXT NOT NULL CHECK (purpose IN ('first_admin', 'break_glass')),
    email                    TEXT NOT NULL,
    display_name             TEXT NOT NULL,
    target_account_id        TEXT,
    code_hash                TEXT NOT NULL UNIQUE,
    os_user                  TEXT NOT NULL,
    issued_at                TEXT NOT NULL,
    expires_at               TEXT NOT NULL,
    used_at                  TEXT,
    enrolled_account_id      TEXT,
    revoked_credential_count INTEGER NOT NULL DEFAULT 0,
    audit_event_id           TEXT NOT NULL,      -- setup_code_issued
    redeemed_audit_event_id  TEXT                -- setup_code_redeemed / break_glass_used
);

CREATE TABLE auth_installations (
    id                    TEXT PRIMARY KEY,
    label                 TEXT NOT NULL,
    primary_host          INTEGER NOT NULL DEFAULT 0 CHECK (primary_host IN (0, 1)),
    created_at            TEXT NOT NULL,
    created_by_account_id TEXT,
    retired_at            TEXT,
    retire_reason         TEXT
);
CREATE INDEX auth_installations_active ON auth_installations (retired_at, label);

CREATE TABLE auth_service_keys (
    id                    TEXT PRIMARY KEY,
    installation_id       TEXT NOT NULL REFERENCES auth_installations (id),
    label                 TEXT NOT NULL,
    scopes_json           TEXT NOT NULL,
    key_prefix            TEXT NOT NULL,
    key_hash              TEXT NOT NULL UNIQUE,
    created_at            TEXT NOT NULL,
    created_by_account_id TEXT,
    expires_at            TEXT,
    revoked_at            TEXT,
    revoke_reason         TEXT,
    rotated_from_key_id   TEXT,
    superseded_by_key_id  TEXT,
    grace_until           TEXT,
    last_used_at          TEXT
);
CREATE INDEX auth_service_keys_installation ON auth_service_keys (installation_id);
