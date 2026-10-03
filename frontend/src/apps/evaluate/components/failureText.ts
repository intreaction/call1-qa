// Plain-language reasons for the job error codes Store reports on a failed result group
// (`ResultGroup.failure_code`, the closed `JobErrorCode` set in call1/contracts/errors.py). The
// Workbench shows these instead of the raw code; an unknown code falls back to its words.

/** For the audio stages (validation, transcription) `validation_rejected` means the recording itself. */
const RECORDING_REJECTED = 'the recording was rejected: too short, no speech detected, or the audio could not be read';

const REASONS: Record<string, string> = {
  validation_rejected: "the model's answer could not be used",
  provider_error: 'the model returned an error',
  provider_timeout: 'the model took too long to answer',
  context_limit_exceeded: 'the call is too long for the model',
  model_unavailable: 'the model is not available on this appliance',
  model_unqualified: 'the model has not passed qualification for this task',
  route_disabled: 'the model route is turned off',
  route_policy_rejected: 'the model route is not allowed by policy',
  credential_missing: 'a model credential is missing',
  configuration_error: 'a configuration problem',
  input_unavailable: 'an earlier step did not produce what this one needs',
  resource_unavailable: 'the appliance was out of capacity',
  lease_expired: 'the worker stopped responding',
  worker_crashed: 'the worker stopped unexpectedly',
  cancelled: 'it was cancelled',
  store_publication_failed: 'the result could not be saved',
};

/** "the model took too long to answer" for `provider_timeout`; words for an unknown code. `audio`
 * is true for the transcript group, where `validation_rejected` means the recording was rejected. */
export function failureReason(code: string | null | undefined, audio = false): string | null {
  if (!code) return null;
  if (audio && code === 'validation_rejected') return RECORDING_REJECTED;
  return REASONS[code] ?? code.replace(/_/g, ' ');
}

/** True when the recording itself was rejected (too short, silent or unreadable). */
export function isRecordingRejected(code: string | null | undefined): boolean {
  return code === 'validation_rejected';
}
