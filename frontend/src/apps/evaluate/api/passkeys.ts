// Passkey ceremonies with @simplewebauthn/browser. Passkey-only: there is no password endpoint,
// no reset and no fallback (contracts/README.md "Principals", "Identity details").
//
// Each ceremony is split into `begin…` (a Store request) and `complete…` (the browser prompt plus
// the finish request), so the UI can run the browser prompt from a click when a browser demands
// a fresh user gesture. WebAuthn payloads go to Store exactly as the library returns them.

import {
  WebAuthnError,
  browserSupportsWebAuthn,
  startAuthentication,
  startRegistration,
  type PublicKeyCredentialCreationOptionsJSON,
  type PublicKeyCredentialRequestOptionsJSON,
} from '@simplewebauthn/browser';
import type { Output } from '@/contracts';
import type { StoreClient } from './client';
import type { SessionInfo } from './types';

export type RegistrationBegin = Output<'RegistrationBeginResponse'>;
export type AuthenticationBegin = Output<'AuthenticationBeginResponse'>;
export type AddAuthenticatorBegin = Output<'AddAuthenticatorBeginResponse'>;
export type RegistrationFinish = Output<'RegistrationFinishResponse'>;
type RegistrationCredentialJSON = Output<'RegistrationFinishRequest'>['credential'];
type AuthenticationCredentialJSON = Output<'AuthenticationFinishRequest'>['credential'];

export { browserSupportsWebAuthn };

// The contract's options are PublicKeyCredential*OptionsJSON as @simplewebauthn/browser takes
// them; the generated and library types differ only in how open-ended strings are typed.
function creationOptions(o: RegistrationBegin['options']): PublicKeyCredentialCreationOptionsJSON {
  return o as unknown as PublicKeyCredentialCreationOptionsJSON;
}
function requestOptions(o: AuthenticationBegin['options']): PublicKeyCredentialRequestOptionsJSON {
  return o as unknown as PublicKeyCredentialRequestOptionsJSON;
}

/** True once `expires_at` (the ceremony's challenge lifetime) has passed. */
export function ceremonyExpired(begin: { expires_at: string }, now = Date.now()): boolean {
  return Date.parse(begin.expires_at) <= now + 2000;
}

// --- sign-in (account-first: email, then the authenticator) ------------------------------------

export function beginSignIn(client: StoreClient, email: string): Promise<AuthenticationBegin> {
  return client.post('/store/v1/auth/sign-in/begin', { body: { email: email.trim() } });
}

export async function completeSignIn(client: StoreClient, begin: AuthenticationBegin): Promise<SessionInfo> {
  const credential = await startAuthentication({ optionsJSON: requestOptions(begin.options) });
  const res = await client.post('/store/v1/auth/sign-in/finish', {
    body: { ceremony_id: begin.ceremony_id, credential: credential as unknown as AuthenticationCredentialJSON },
  });
  client.setCsrfToken(res.session.csrf_token);
  return res.session;
}

// --- enrollment (an invitation token or a setup code) ------------------------------------------

export type EnrollmentSecret = { invitation_token: string } | { setup_code: string };

export function beginEnrollment(client: StoreClient, secret: EnrollmentSecret): Promise<RegistrationBegin> {
  return client.post('/store/v1/auth/enroll/begin', { body: secret });
}

export async function completeEnrollment(
  client: StoreClient,
  begin: RegistrationBegin,
  nickname?: string,
): Promise<RegistrationFinish> {
  const credential = await startRegistration({ optionsJSON: creationOptions(begin.options) });
  const res = await client.post('/store/v1/auth/enroll/finish', {
    body: {
      ceremony_id: begin.ceremony_id,
      credential: credential as unknown as RegistrationCredentialJSON,
      nickname: nickname?.trim() || null,
    },
  });
  if (res.signed_in) client.setCsrfToken(res.signed_in.session.csrf_token);
  return res;
}

// --- add another authenticator (step-up) --------------------------------------------------------

export function beginAddAuthenticator(client: StoreClient, nickname?: string): Promise<AddAuthenticatorBegin> {
  return client.post('/store/v1/auth/authenticators/begin', { body: { nickname: nickname?.trim() || null } });
}

/** Step 1: a fresh user-verified assertion from an existing authenticator. */
export async function reauthenticate(begin: AddAuthenticatorBegin): Promise<AuthenticationCredentialJSON> {
  const assertion = await startAuthentication({ optionsJSON: requestOptions(begin.reauthentication) });
  return assertion as unknown as AuthenticationCredentialJSON;
}

/** Step 2: register the new authenticator and finish. */
export async function registerAdditional(
  client: StoreClient,
  begin: AddAuthenticatorBegin,
  reauthentication: AuthenticationCredentialJSON,
  nickname?: string,
): Promise<RegistrationFinish> {
  const credential = await startRegistration({ optionsJSON: creationOptions(begin.options) });
  return client.post('/store/v1/auth/authenticators/finish', {
    body: {
      ceremony_id: begin.ceremony_id,
      credential: credential as unknown as RegistrationCredentialJSON,
      reauthentication,
      nickname: nickname?.trim() || null,
    },
  });
}

// --- browser-side errors -------------------------------------------------------------------------

/**
 * True when the browser refused to show its prompt because the call was not tied to a click
 * (Safari), so the UI should offer a button that retries from a user gesture.
 */
export function needsUserGesture(err: unknown): boolean {
  // The library passes NotAllowedError through with the original name (and keeps it as `cause`).
  const cause = (err as { cause?: unknown } | null)?.cause;
  return isNotAllowed(err) || isNotAllowed(cause);
}

function isNotAllowed(err: unknown): boolean {
  return err instanceof Error && err.name === 'NotAllowedError';
}

/** A human sentence for an error thrown by the browser prompt, or null if it is not one. */
export function describeWebAuthnError(err: unknown): string | null {
  if (err instanceof WebAuthnError) {
    switch (err.code) {
      case 'ERROR_CEREMONY_ABORTED':
        return 'The passkey prompt was cancelled.';
      case 'ERROR_AUTHENTICATOR_PREVIOUSLY_REGISTERED':
        return 'That authenticator is already registered to your account. Use a different one.';
      case 'ERROR_INVALID_DOMAIN':
      case 'ERROR_INVALID_RP_ID':
        return 'This page is not on the Store hostname passkeys are bound to. Open Evaluate at its usual address.';
      case 'ERROR_AUTHENTICATOR_MISSING_USER_VERIFICATION_SUPPORT':
        return 'This authenticator cannot verify you (PIN or biometric). Use one that can.';
      default:
        break;
    }
  }
  if (err instanceof Error) {
    if (err.name === 'NotAllowedError') return 'The passkey prompt was dismissed or timed out. Try again.';
    if (err.name === 'InvalidStateError') return 'That authenticator is already registered. Use a different one.';
    if (err.name === 'SecurityError') return 'The browser refused the passkey request for this address.';
    if (err instanceof WebAuthnError) return err.message;
  }
  return null;
}
