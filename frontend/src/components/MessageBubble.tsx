import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AlertCircle, FileText } from "lucide-react";
import type { Message } from "../lib/types";
import { CitationCard } from "./CitationCard";
import { TracePanel } from "./TracePanel";

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="animate-fade-up flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[var(--color-accent)] px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  const showCitations = !message.streaming && (message.citations?.length ?? 0) > 0;

  return (
    <div className="animate-fade-up space-y-3">
      <div className="rounded-2xl rounded-bl-sm border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-4 py-3">
        {message.error ? (
          <p className="flex items-start gap-2 text-sm text-red-600 dark:text-red-400">
            <AlertCircle size={15} className="mt-0.5 shrink-0" />
            {message.error}
          </p>
        ) : message.content ? (
          <div className="prose-answer text-sm">
            <Markdown remarkPlugins={[remarkGfm]}>{message.content}</Markdown>
            {message.streaming && (
              <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-[var(--color-accent)] align-middle" />
            )}
          </div>
        ) : (
          <ThinkingDots />
        )}

        {message.latencyMs != null && !message.streaming && (
          <p className="mt-2 text-[11px] text-[var(--color-ink-muted)]">
            answered in {(message.latencyMs / 1000).toFixed(1)}s
          </p>
        )}
      </div>

      {showCitations && (
        <div>
          <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-[var(--color-ink-muted)]">
            <FileText size={12} />
            {message.citations!.length} records used
          </p>
          <div className="grid gap-1.5">
            {message.citations!.map((c) => (
              <CitationCard key={c.index} citation={c} />
            ))}
          </div>
        </div>
      )}

      {message.trace && !message.streaming && <TracePanel trace={message.trace} />}
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1 text-sm text-[var(--color-ink-muted)]">
      <span className="flex gap-1">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--color-ink-muted)]"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </span>
      searching placement records
    </div>
  );
}
