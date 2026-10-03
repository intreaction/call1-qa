import { useState } from 'react';
import { KeyRound } from 'lucide-react';
import { Button, Notice, TextInput } from './ui';

/**
 * Shown wherever a write is unavailable because this browser tab has no console credential
 * (call1/process/console.py). The credential is a URL-fragment secret printed once by
 * `python -m call1.process serve` / `console-token --rotate`; a new tab never saw it, so this
 * offers a manual paste as the documented fallback ("keep the token in memory or
 * sessionStorage" — README "Console credential").
 */
export function ConsoleTokenNotice({
  configured,
  onToken,
}: {
  /** Whether Process has a credential configured at all (false: none has been issued yet). */
  configured: boolean | undefined;
  onToken(token: string): void;
}) {
  const [value, setValue] = useState('');
  return (
    <Notice tone="yellow" icon={KeyRound}>
      <div className="flex flex-col gap-2">
        <p>
          {configured === false
            ? 'No console credential has been issued yet. Run python -m call1.process console-token --rotate on this machine and open the printed link.'
            : "This tab doesn't have the console credential, so changes (imports, retries, cancellations, model selection, and training) are unavailable. Open the link Process printed on start, or paste the token below."}
        </p>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            const trimmed = value.trim();
            if (trimmed) {
              onToken(trimmed);
              setValue('');
            }
          }}
        >
          <TextInput
            aria-label="Console token"
            placeholder="c1con_…"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            className="max-w-xs font-normal"
          />
          <Button type="submit" size="sm" variant="secondary" disabled={!value.trim()}>
            Connect
          </Button>
        </form>
      </div>
    </Notice>
  );
}
