import { useEffect, useState } from 'react';
import { ArrowDown, Download, Layers, Play, RotateCcw } from 'lucide-react';
import { Button, Card, SelectInput, StatusPill, Switch } from './ui';

const CATALOG = [
  { id: 'retail-v1', name: 'Retail & e-commerce', description: 'Orders, returns, deliveries, and customer resolutions.' },
  { id: 'banking-v1', name: 'Banking & finance', description: 'Account questions, transactions, and service quality.' },
  { id: 'insurance-v1', name: 'Insurance', description: 'Policy questions, claims, and customer support.' },
];
const KEY = 'call1-demo-fine-tune-stack-v1';
const EVENT = 'call1-demo-fine-tune-stack-change';
interface Stack { installed: string[]; industry: string | null; privateFor: string | null; privateEnabled: boolean }
const EMPTY: Stack = { installed: [], industry: null, privateFor: null, privateEnabled: false };
function read(): Stack {
  try {
    const data = JSON.parse(localStorage.getItem(KEY) ?? 'null');
    if (!data || !Array.isArray(data.installed)) return EMPTY;
    const installed = CATALOG.filter((m) => data.installed.includes(m.id)).map((m) => m.id);
    const industry = installed.includes(data.industry) ? data.industry as string : null;
    return { installed, industry, privateFor: data.privateFor === (industry ?? 'base') ? data.privateFor : null, privateEnabled: data.privateEnabled === true };
  } catch { return EMPTY; }
}
export function useDemoStack() {
  const [stack, setStack] = useState(read);
  useEffect(() => {
    const update = () => setStack(read());
    window.addEventListener(EVENT, update);
    window.addEventListener('storage', update);
    return () => { window.removeEventListener(EVENT, update); window.removeEventListener('storage', update); };
  }, []);
  function save(next: Stack) {
    localStorage.setItem(KEY, JSON.stringify(next));
    setStack(next);
    window.dispatchEvent(new Event(EVENT));
  }
  const industryName = CATALOG.find((m) => m.id === stack.industry)?.name ?? null;
  return { stack, save, industryName, privateActive: stack.privateEnabled && stack.privateFor === (stack.industry ?? 'base') };
}

/** A labelled demo of package delivery and industry + customer layers. No weights or call data
 * are downloaded, uploaded, changed, or trained by this simulation. Real tools remain below. */
export function FineTuneExperience({ canWrite }: { canWrite: boolean }) {
  const { stack, save, industryName, privateActive } = useDemoStack();
  const [choice, setChoice] = useState(stack.industry ?? CATALOG[0].id);
  const [activity, setActivity] = useState<'pull' | 'train' | null>(null);
  const [message, setMessage] = useState('');
  const lineage = stack.industry ?? 'base';
  const hasPrivate = stack.privateFor === lineage;
  const selected = CATALOG.find((m) => m.id === choice)!;
  const installed = stack.installed.includes(choice);
  const disabled = !canWrite || activity !== null;

  async function act(kind: 'pull' | 'train') {
    setActivity(kind);
    setMessage('');
    await new Promise((resolve) => window.setTimeout(resolve, 800));
    const current = read();
    if (kind === 'pull') {
      save({ ...current, installed: [...new Set([...current.installed, choice])], industry: choice, privateFor: null, privateEnabled: false });
      setMessage(`${selected.name} is ready. Your private layer will train on this edition.`);
    } else if ((current.industry ?? 'base') === lineage) {
      save({ ...current, privateFor: lineage, privateEnabled: true });
      setMessage('Your private fine-tune is ready. Both LoRA adapters are active together.');
    }
    setActivity(null);
  }

  return (
    <div className="space-y-4" data-testid="fine-tune-experience">
      <p className="text-xs text-fg-muted"><StatusPill tone="magenta">Demo preview</StatusPill> Downloads and stacked training are simulated. Live processing stays on its installed model.</p>
      <Card title="Your model stack" icon={Layers} subtitle="One base model, two LoRA adapters working together">
        <ol aria-label="Model layers" className="flex flex-col items-center">
          {[
            { label: 'Gemma 4 E2B', detail: 'Included base model', tone: 'neutral' as const },
            { label: industryName ?? 'Call1 fine-tune', detail: industryName ? 'Call1 LoRA · active' : 'Choose an industry edition below', tone: industryName ? 'blue' as const : 'neutral' as const },
            { label: 'Your private fine-tune', detail: privateActive ? 'Private LoRA · active alongside Call1' : hasPrivate ? 'Paused · Call1 layer stays active' : 'Add your team’s knowledge below', tone: privateActive ? 'green' as const : 'neutral' as const },
          ].map((layer, i) => (
            <li key={layer.label} className="w-full max-w-xl">
              {i > 0 && <ArrowDown className="h-5 w-5 mx-auto my-1 text-fg-subtle" aria-hidden="true" />}
              <div className="rounded-md border border-border-muted bg-canvas-subtle px-4 py-3 flex flex-wrap justify-between items-center gap-2">
                <span className="text-sm font-semibold text-fg">{layer.label}</span>
                <StatusPill tone={layer.tone}>{layer.detail}</StatusPill>
              </div>
            </li>
          ))}
        </ol>
      </Card>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Get a Call1 fine-tune" icon={Download} subtitle="A ready-to-use edition for your industry">
          <div className="space-y-3">
            <label htmlFor="demo-industry" className="text-xs font-medium text-fg-muted">Industry edition</label>
            <SelectInput id="demo-industry" value={choice} disabled={disabled} onChange={(e) => { setChoice(e.target.value); setMessage(''); }}>
              {CATALOG.map((m) => <option key={m.id} value={m.id}>{m.name}{stack.installed.includes(m.id) ? ' · installed' : ''}</option>)}
            </SelectInput>
            <p className="text-sm text-fg-muted">{selected.description}</p>
            <p className="text-xs text-fg-muted">Your subscription includes the fine-tune, ready-to-use rubrics, and peer comparisons using the same questions and rubric/model versions.</p>
            <Button variant="primary" icon={Download} busy={activity === 'pull'} disabled={disabled || stack.industry === choice} onClick={() => void act('pull')}>{installed ? 'Use this edition' : 'Get fine-tune'}</Button>
            {stack.industry && <Button size="sm" variant="ghost" icon={RotateCcw} disabled={disabled} onClick={() => { save({ ...stack, industry: null, privateFor: null, privateEnabled: false }); setMessage('Using the Gemma base.'); }}>Use Gemma base</Button>}
          </div>
        </Card>
        <Card title="Fine-tune on your calls" icon={Play} subtitle="Built in · private to your contact center">
          <div className="space-y-3">
            <p className="text-sm text-fg-muted">Review and correct calls in Evaluate. Call1 learns from those corrections on your computer.</p>
            <p className="text-sm text-fg">Learns on top of <span className="font-medium">{industryName ? `Call1 ${industryName} LoRA` : 'a Call1 LoRA — choose an edition first'}</span>.</p>
            <Button variant="primary" icon={Play} busy={activity === 'train'} disabled={disabled || !stack.industry} onClick={() => void act('train')}>{hasPrivate ? 'Update private fine-tune' : 'Train private fine-tune'}</Button>
            {hasPrivate && <div className="flex items-center gap-2"><Switch label="Use private fine-tune" checked={privateActive} disabled={disabled} onChange={(enabled) => save({ ...stack, privateEnabled: enabled })} /><span className="text-sm text-fg">Use private fine-tune</span></div>}
            <p className="text-xs text-fg-muted">Both adapters contribute to the same model. Your private LoRA adds your team’s knowledge to the Call1 LoRA. Turning it off keeps the Call1 fine-tune. A new industry edition needs a fresh private fine-tune.</p>
          </div>
        </Card>
      </div>
      {message && <p role="status" className="text-sm text-primer-greenFg">{message}</p>}
      {!canWrite && <p className="text-sm text-fg-muted">Connect the console credential to try the fine-tuning demo.</p>}
    </div>
  );
}
