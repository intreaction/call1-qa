import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Button, Notice } from './ui';

/**
 * Keeps a render error in one console tab from unmounting the whole Process console (React has no
 * other way to catch one). The header and the other tabs keep working; the tab shows what failed
 * and a retry. `App` keys it by tab, so switching tabs also resets it.
 */
export class ViewBoundary extends Component<{ name: string; children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`Process console: the ${this.props.name} tab failed to render`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="max-w-4xl mx-auto space-y-3">
        <Notice tone="red">
          The {this.props.name} tab could not be shown: {this.state.error.message || 'an unexpected error'}. The rest of the console
          still works.
        </Notice>
        <Button size="sm" onClick={() => this.setState({ error: null })}>
          Try again
        </Button>
      </div>
    );
  }
}
