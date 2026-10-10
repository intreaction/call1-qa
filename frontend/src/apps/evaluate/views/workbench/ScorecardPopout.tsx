import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';

/** A contextual, nonmodal inspection card. Its anchor stays visible and keyboard focus returns
 * there on dismissal. Reposition on scrolling, resizing, or expansion of the review form. */
export function ScorecardPopout({ anchor, title, eyebrow, children, footer, onClose }: {
  anchor: HTMLElement | null;
  title: string;
  eyebrow: string;
  children: ReactNode;
  footer: ReactNode;
  onClose: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const closeCallback = useRef(onClose);
  closeCallback.current = onClose;
  const titleId = useId();
  const [position, setPosition] = useState({ top: 0, left: 0, maxHeight: 520, above: true, pointer: 24 });

  useLayoutEffect(() => {
    if (!anchor || !panel.current) return;
    const update = () => {
      const bounds = anchor.getBoundingClientRect();
      if (bounds.bottom < 0 || bounds.top > window.innerHeight) {
        closeCallback.current();
        return;
      }
      const width = Math.min(440, window.innerWidth - 24);
      const above = bounds.top > window.innerHeight - bounds.bottom;
      const space = above ? bounds.top - 24 : window.innerHeight - bounds.bottom - 24;
      const maxHeight = Math.min(520, Math.max(160, space));
      const height = Math.min(panel.current!.getBoundingClientRect().height, maxHeight);
      const left = Math.max(12, Math.min(bounds.left - 16, window.innerWidth - width - 12));
      const top = Math.max(12, Math.min(above ? bounds.top - height - 10 : bounds.bottom + 10, window.innerHeight - height - 12));
      setPosition({ top, left, maxHeight, above, pointer: Math.max(16, Math.min(width - 16, bounds.left + bounds.width / 2 - left)) });
    };
    update();
    const resize = new ResizeObserver(update);
    resize.observe(panel.current);
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    close.current?.focus({ preventScroll: true });
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !panel.current?.contains(event.target) && !anchor.contains(event.target)) closeCallback.current();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        closeCallback.current();
      }
    };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => {
      resize.disconnect();
      window.removeEventListener('resize', update);
      window.removeEventListener('scroll', update, true);
      document.removeEventListener('pointerdown', outside);
      document.removeEventListener('keydown', escape);
      if (anchor.isConnected) anchor.focus({ preventScroll: true });
    };
  }, [anchor]);

  return createPortal(
    <div ref={panel} id="scorecard-criterion-popout" role="dialog" aria-modal="false" aria-labelledby={titleId}
      className="fixed z-50 flex flex-col w-[440px] max-w-[calc(100vw-24px)] rounded-xl border border-border bg-canvas shadow-2xl"
      style={{ top: position.top, left: position.left, maxHeight: position.maxHeight }}>
      <span aria-hidden="true" className={`absolute w-2.5 h-2.5 rotate-45 bg-canvas border-border ${position.above ? '-bottom-1.5 border-b border-r' : '-top-1.5 border-t border-l'}`} style={{ left: position.pointer - 5 }} />
      <header role="presentation" className="shrink-0 flex items-start justify-between gap-4 px-4 py-3 border-b border-border-muted">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold tracking-wide uppercase text-fg-muted truncate" title={eyebrow}>{eyebrow}</p>
          <h2 id={titleId} className="text-sm font-semibold text-fg mt-0.5">{title}</h2>
        </div>
        <button ref={close} type="button" onClick={onClose} aria-label="Close criterion details" className="shrink-0 rounded-md p-1 text-fg-muted hover:bg-canvas-inset hover:text-fg focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue">
          <X className="w-4 h-4" aria-hidden="true" />
        </button>
      </header>
      <div className="min-h-0 overflow-y-auto overscroll-contain p-4 space-y-3">{children}</div>
      <footer role="group" aria-label="Criterion navigation" className="shrink-0 flex items-center justify-between gap-2 px-3 py-2 border-t border-border-muted bg-canvas-subtle rounded-b-xl">{footer}</footer>
    </div>, document.body,
  );
}
