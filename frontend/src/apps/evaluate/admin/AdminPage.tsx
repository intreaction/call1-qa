import { BookMarked, Mail, Server, ShieldAlert, Users } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { EmptyState, NotBuiltYet, PageHeader } from '../components/ui';
import { useSession } from '../state/app';
import { DEFERRED_ADMIN_SECTIONS, href, isDeferredAdminSection, type AdminSection } from '../state/router';
import { AccountsPanel } from './AccountsPanel';
import { InstallationsPanel } from './InstallationsPanel';
import { InvitationsPanel } from './InvitationsPanel';
import { VocabularyPanel } from './VocabularyPanel';

const SECTIONS: { id: Exclude<AdminSection, keyof typeof DEFERRED_ADMIN_SECTIONS>; label: string; icon: LucideIcon }[] = [
  { id: 'accounts', label: 'Accounts', icon: Users },
  { id: 'invitations', label: 'Invitations', icon: Mail },
  { id: 'installations', label: 'Installations and keys', icon: Server },
  { id: 'vocabulary', label: 'Vocabulary', icon: BookMarked },
];

/** Admin area (admin role): identity, invitations, Process installations and service keys. */
export function AdminPage({ section }: { section: AdminSection }) {
  const { can } = useSession();
  if (!can('manage_accounts')) {
    return (
      <EmptyState icon={ShieldAlert} title="Admin role required">
        The admin area manages accounts, invitations and Process keys. Ask an admin if you need access.
      </EmptyState>
    );
  }
  return (
    <div className="max-w-5xl mx-auto">
      <PageHeader
        title="Admin"
        description="Every change here is an audited admin action recorded under your account."
      />
      <nav aria-label="Admin sections" className="mb-4 border-b border-border">
        <ul className="flex gap-1 overflow-x-auto">
          {SECTIONS.map((s) => {
            const active = s.id === section;
            const Icon = s.icon;
            return (
              <li key={s.id}>
                <a
                  href={href({ name: 'admin', section: s.id })}
                  aria-current={active ? 'page' : undefined}
                  className={`flex items-center gap-1.5 px-3 h-9 text-sm whitespace-nowrap border-b-2 -mb-px focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue ${
                    active ? 'border-primer-blue text-fg' : 'border-transparent text-fg-muted hover:text-fg'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" aria-hidden="true" />
                  {s.label}
                </a>
              </li>
            );
          })}
        </ul>
      </nav>
      {isDeferredAdminSection(section) && (
        <NotBuiltYet what={DEFERRED_ADMIN_SECTIONS[section]}>
          Evaluate has no screen for this yet. Nothing here is hidden or simulated.
        </NotBuiltYet>
      )}
      {section === 'accounts' && <AccountsPanel />}
      {section === 'invitations' && <InvitationsPanel />}
      {section === 'installations' && <InstallationsPanel />}
      {section === 'vocabulary' && <VocabularyPanel />}
      <p className="text-xs text-fg-subtle mt-6">
        Coming later: admin settings, the audit log, release trust, Pro1 key releases, usage and the price table.
      </p>
    </div>
  );
}
