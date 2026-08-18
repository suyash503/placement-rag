import { useState } from "react";
import { Filter, Layers, Sparkles, Timer } from "lucide-react";
import type { RetrievalTrace, ScoredDoc } from "../lib/types";

export function TracePanel({ trace }: { trace: RetrievalTrace }) {
  const [open, setOpen] = useState(false);

  const vectorOnly = trace.vector_only ?? [];
  const fulltextOnly = trace.fulltext_only ?? [];
  const vectorSet = new Set(vectorOnly.map((d) => d.doc));
  const fulltextSet = new Set(fulltextOnly.map((d) => d.doc));

  const timings = trace.timings_ms ?? {};
  const totalSearch = Object.values(timings).reduce((a, b) => a + b, 0);

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)] transition-colors hover:text-[var(--color-accent)]"
      >
        <Layers size={13} />
        {open ? "Hide" : "Show"} retrieval trace
        {totalSearch > 0 && <span className="opacity-60">· {totalSearch}ms</span>}
      </button>

      {open && (
        <div className="animate-fade-up mt-2 space-y-3 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-canvas)] p-3 text-xs">
          {trace.condensed_question && (
            <Section icon={<Sparkles size={12} />} title="Rewritten as standalone question">
              <p className="italic text-[var(--color-ink-muted)]">"{trace.condensed_question}"</p>
            </Section>
          )}

          <Section icon={<Filter size={12} />} title="Extracted filters">
            {Object.keys(trace.filters ?? {}).length ? (
              <>
                <pre className="overflow-x-auto rounded bg-[var(--color-accent-soft)] p-2 font-mono text-[11px] leading-relaxed">
                  {JSON.stringify(trace.filters, null, 2)}
                </pre>
                <ul className="mt-1.5 space-y-0.5 text-[var(--color-ink-muted)]">
                  {(trace.filter_notes ?? []).map((n, i) => (
                    <li key={i}>· {n}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="text-[var(--color-ink-muted)]">
                No structured constraint found — pure semantic search.
              </p>
            )}
            {trace.filter_relaxed && (
              <p className="mt-1.5 rounded bg-amber-500/10 px-2 py-1 text-amber-700 dark:text-amber-400">
                Filter matched zero documents, so it was dropped and the search retried.
              </p>
            )}
          </Section>

          <Section icon={<Layers size={12} />} title="Pipeline">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
              <Stage label={`${trace.candidates_found ?? 0} candidates`} sub={trace.mode} />
              <Arrow />
              <Stage
                label={`rerank ${trace.rerank_top?.length ? "top " + trace.rerank_top.length : "off"}`}
                sub="cross-encoder"
              />
              <Arrow />
              <Stage label={`${trace.returned?.length ?? 0} to the model`} sub="deduplicated" />
            </div>
          </Section>

          {(vectorOnly.length > 0 || fulltextOnly.length > 0) && (
            <Section icon={<Layers size={12} />} title="What each leg found on its own">
              <div className="grid gap-3 sm:grid-cols-2">
                <Leg
                  title="Vector (cosine)"
                  docs={vectorOnly}
                  uniqueAgainst={fulltextSet}
                  formatScore={(s) => s.toFixed(3)}
                />
                <Leg
                  title="Full-text (BM25)"
                  docs={fulltextOnly}
                  uniqueAgainst={vectorSet}
                  formatScore={(s) => s.toFixed(2)}
                />
              </div>
              <p className="mt-2 text-[11px] text-[var(--color-ink-muted)]">
                Highlighted rows were found by only one leg. Reciprocal Rank Fusion merges both
                lists, so those documents survive instead of being lost.
              </p>
            </Section>
          )}

          {trace.rerank_top && trace.rerank_top.length > 0 && (
            <Section icon={<Sparkles size={12} />} title="After cross-encoder rerank">
              <ol className="space-y-0.5">
                {trace.rerank_top.map((d, i) => (
                  <li key={i} className="flex items-baseline justify-between gap-3">
                    <span className="truncate text-[var(--color-ink-muted)]">
                      {i + 1}. {d.doc}
                    </span>
                    <span className="shrink-0 font-mono text-[10px]">{d.score.toFixed(2)}</span>
                  </li>
                ))}
              </ol>
            </Section>
          )}

          {Object.keys(timings).length > 0 && (
            <Section icon={<Timer size={12} />} title="Timings">
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[var(--color-ink-muted)]">
                {Object.entries(timings).map(([k, v]) => (
                  <span key={k}>
                    {k} <span className="font-mono text-[var(--color-ink)]">{v}ms</span>
                  </span>
                ))}
              </div>
            </Section>
          )}
        </div>
      )}
    </div>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h4 className="mb-1.5 flex items-center gap-1.5 font-medium">
        {icon}
        {title}
      </h4>
      {children}
    </div>
  );
}

function Stage({ label, sub }: { label: string; sub?: string }) {
  return (
    <span className="rounded border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-2 py-1">
      <span className="font-medium">{label}</span>
      {sub && <span className="ml-1 text-[var(--color-ink-muted)]">({sub})</span>}
    </span>
  );
}

const Arrow = () => <span className="text-[var(--color-ink-muted)]">→</span>;

function Leg({
  title,
  docs,
  uniqueAgainst,
  formatScore,
}: {
  title: string;
  docs: ScoredDoc[];
  uniqueAgainst: Set<string>;
  formatScore: (s: number) => string;
}) {
  return (
    <div>
      <p className="mb-1 font-medium">{title}</p>
      <ol className="space-y-0.5">
        {docs.map((d, i) => {
          const unique = !uniqueAgainst.has(d.doc);
          return (
            <li
              key={i}
              className={`flex items-baseline justify-between gap-2 rounded px-1 ${
                unique ? "bg-[var(--color-accent-soft)]" : ""
              }`}
            >
              <span className="truncate text-[var(--color-ink-muted)]">{d.doc}</span>
              <span className="shrink-0 font-mono text-[10px]">{formatScore(d.score)}</span>
            </li>
          );
        })}
        {!docs.length && <li className="text-[var(--color-ink-muted)]">no hits</li>}
      </ol>
    </div>
  );
}
