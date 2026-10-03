import { useCallback, useEffect, useState } from 'react';

export type Theme = 'dark' | 'light';

const STORAGE_KEY = 'call1-theme';
const LIGHT_QUERY = '(prefers-color-scheme: light)';

function readStoredTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // storage denied — no stored choice
  }
  return null;
}

function systemTheme(): Theme {
  try {
    if (window.matchMedia && window.matchMedia(LIGHT_QUERY).matches) return 'light';
  } catch {
    // matchMedia unavailable — fall through to the default
  }
  return 'dark';
}

function readInitialTheme(): Theme {
  return readStoredTheme() ?? systemTheme();
}

/**
 * The sun/moon theme. An explicit toggle is persisted in localStorage `call1-theme`; until the
 * user toggles, the theme follows `prefers-color-scheme` (including live OS changes) and nothing
 * is written, so a later OS change is not masked by a value stored on the first visit.
 */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(readInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    // Atomic palette swap: suppress color transitions for the single style
    // pass that applies the new token set, then flush styles before
    // re-enabling transitions (see .theme-switching in index.css).
    root.classList.add('theme-switching');
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme;
    void document.body.offsetHeight; // forced style/layout flush
    root.classList.remove('theme-switching');
    window.dispatchEvent(new CustomEvent('call1:theme-change', { detail: { theme } }));
  }, [theme]);

  // Follow the OS setting while the user has not chosen a theme explicitly.
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    let media: MediaQueryList;
    try {
      media = window.matchMedia(LIGHT_QUERY);
    } catch {
      return undefined;
    }
    const onChange = () => {
      if (readStoredTheme() === null) setTheme(systemTheme());
    };
    media.addEventListener?.('change', onChange);
    return () => media.removeEventListener?.('change', onChange);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === 'light' ? 'dark' : 'light';
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch {
        // storage denied — session-only
      }
      return next;
    });
  }, []);

  return { theme, toggleTheme };
}
