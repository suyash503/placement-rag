import { useState } from "react";
import { Building2, ChevronDown, GraduationCap, MapPin } from "lucide-react";
import type { Citation } from "../lib/types";

const asText = (v: string | string[] | number | number[] | null | undefined) => {
  if (v === null || v === undefined) return null;
  if (Array.isArray(v)) {
    if (!v.length) return null;
    return v.length > 3 ? `${v.slice(0, 3).join(", ")} +${v.length - 3}` : v.join(", ");
  }
  return String(v);
};

export function CitationCard({ citation }: { citation: Citation }) {
  const [open, setOpen] = useState(false);
  const isProfile = citation.doc_type === "company";

  return (
    <div className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface)] text-sm transition-colors hover:border-[var(--color-accent)]/40">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2.5 p-3 text-left"
      >
        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded bg-[var(--color-accent-soft)] text-[11px] font-semibold text-[var(--color-accent)]">
          {citation.index}
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5 font-medium">
            {isProfile ? <Building2 size={13} className="shrink-0" /> : null}
            <span className="truncate">{citation.company}</span>
            {isProfile && (
              <span className="shrink-0 rounded bg-[var(--color-accent-soft)] px-1.5 py-px text-[10px] font-medium text-[var(--color-accent)]">
                profile
              </span>
            )}
          </span>

          <span className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--color-ink-muted)]">
            {citation.package_lpa != null && (
              <span className="font-medium text-[var(--color-ink)]">{citation.package_lpa} LPA</span>
            )}
            {citation.cgpa_cutoff != null && <span>CGPA ≥ {citation.cgpa_cutoff}</span>}
            {asText(citation.year) && <span>{asText(citation.year)}</span>}
            {asText(citation.college) && (
              <span className="flex items-center gap-1 truncate">
                <GraduationCap size={12} className="shrink-0" />
                <span className="truncate">{asText(citation.college)}</span>
              </span>
            )}
          </span>
        </span>

        <ChevronDown
          size={15}
          className={`mt-0.5 shrink-0 text-[var(--color-ink-muted)] transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div className="animate-fade-up border-t border-[var(--color-border-subtle)] px-3 pb-3 pt-2.5 text-xs">
          <dl className="grid gap-x-4 gap-y-1.5 sm:grid-cols-2">
            {citation.role && <Field label="Role" value={citation.role} />}
            {citation.branches.length > 0 && (
              <Field label="Branches" value={citation.branches.join(", ")} />
            )}
            {citation.active_backlogs && (
              <Field label="Active backlogs" value={citation.active_backlogs} />
            )}
            {citation.job_location && (
              <Field
                label="Location"
                value={
                  <span className="flex items-center gap-1">
                    <MapPin size={11} />
                    {citation.job_location}
                  </span>
                }
              />
            )}
          </dl>

          {citation.selection_rounds.length > 0 && (
            <div className="mt-2.5">
              <dt className="mb-1 text-[var(--color-ink-muted)]">Selection process</dt>
              <div className="flex flex-wrap items-center gap-1">
                {citation.selection_rounds.map((round, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span className="rounded bg-[var(--color-accent-soft)] px-1.5 py-0.5">
                      {round}
                    </span>
                    {i < citation.selection_rounds.length - 1 && (
                      <span className="text-[var(--color-ink-muted)]">→</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}

          <p className="mt-2.5 border-t border-[var(--color-border-subtle)] pt-2 leading-relaxed text-[var(--color-ink-muted)]">
            {citation.text}
          </p>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <dt className="shrink-0 text-[var(--color-ink-muted)]">{label}</dt>
      <dd className="min-w-0 truncate font-medium">{value}</dd>
    </div>
  );
}
