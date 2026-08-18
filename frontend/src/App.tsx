import { useEffect, useRef, useState } from "react";
import { ArrowUp, Database, Moon, Square, Sun } from "lucide-react";
import { fetchStats, streamChat } from "./lib/api";
import type { Message, Stats } from "./lib/types";
import { MessageBubble } from "./components/MessageBubble";

const SUGGESTIONS = [
  "Which companies offered more than 10 LPA?",
  "I have 7 CGPA and no backlogs — what am I eligible for?",
  "Show me Computer Engineering companies from 2018",
  "What is the selection process at Morgan Stanley?",
  "Tell me about TCS across all years",
  "Core mechanical companies between 2015 and 2017",
];

const newId = () => Math.random().toString(36).slice(2);

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [dark, setDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);

  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    fetchStats().then(setStats);
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || busy) return;

    const history = messages
      .filter((m) => !m.error && m.content)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

    const answerId = newId();
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", content: trimmed },
      { id: answerId, role: "assistant", content: "", streaming: true },
    ]);
    setInput("");
    setBusy(true);

    const patch = (fn: (m: Message) => Message) =>
      setMessages((prev) => prev.map((m) => (m.id === answerId ? fn(m) : m)));

    const controller = new AbortController();
    abortRef.current = controller;

    await streamChat(
      trimmed,
      history,
      {
        onMeta: (citations, trace) => patch((m) => ({ ...m, citations, trace })),
        onToken: (value) => patch((m) => ({ ...m, content: m.content + value })),
        onDone: (latencyMs) => patch((m) => ({ ...m, streaming: false, latencyMs })),
        onError: (message) => patch((m) => ({ ...m, streaming: false, error: message })),
      },
      controller.signal,
    );

    setBusy(false);
    abortRef.current = null;
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
  };

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between gap-4 border-b border-[var(--color-border-subtle)] py-3">
        <div className="min-w-0">
          <h1 className="text-sm font-semibold">Placement RAG Assistant</h1>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[11px] text-[var(--color-ink-muted)]">
            {stats ? (
              <>
                <span className="flex items-center gap-1">
                  <Database size={10} />
                  {stats.records.toLocaleString()} drives
                </span>
                <span>{stats.companies.toLocaleString()} companies</span>
                <span>{stats.colleges} colleges</span>
                {stats.years.length > 0 && (
                  <span>
                    {stats.years[0]}–{stats.years[stats.years.length - 1]}
                  </span>
                )}
              </>
            ) : (
              <span>connecting…</span>
            )}
          </p>
        </div>

        <button
          onClick={() => setDark((v) => !v)}
          aria-label="Toggle theme"
          className="rounded-lg border border-[var(--color-border-subtle)] p-2 text-[var(--color-ink-muted)] transition-colors hover:text-[var(--color-ink)]"
        >
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </header>

      <main className="flex-1 space-y-4 overflow-y-auto py-5">
        {messages.length === 0 ? (
          <div className="pt-8">
            <h2 className="text-lg font-medium">Ask about campus placements</h2>
            <p className="mt-1 text-sm text-[var(--color-ink-muted)]">
              Hybrid search over {stats?.records.toLocaleString() ?? "thousands of"} recruitment
              drives. Every answer cites the records it used.
            </p>
            <div className="mt-5 grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-surface)] px-3 py-2.5 text-left text-sm transition-colors hover:border-[var(--color-accent)]/50 hover:bg-[var(--color-accent-soft)]"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} />)
        )}
        <div ref={endRef} />
      </main>

      <footer className="pb-4">
        <div className="flex items-end gap-2 rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-surface)] p-2 focus-within:border-[var(--color-accent)]/50">
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
                if (textareaRef.current) textareaRef.current.style.height = "auto";
              }
            }}
            placeholder="Ask about companies, packages, eligibility…"
            className="max-h-36 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-[var(--color-ink-muted)]"
          />

          {busy ? (
            <button
              onClick={stop}
              aria-label="Stop"
              className="rounded-lg bg-[var(--color-ink-muted)] p-2 text-white"
            >
              <Square size={15} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={() => send(input)}
              disabled={!input.trim()}
              aria-label="Send"
              className="rounded-lg bg-[var(--color-accent)] p-2 text-white transition-opacity disabled:opacity-30"
            >
              <ArrowUp size={15} />
            </button>
          )}
        </div>

        <p className="mt-2 text-center text-[11px] text-[var(--color-ink-muted)]">
          Company, college and year data is real. Packages, CGPA cutoffs and selection rounds are
          synthetic demo values.
        </p>
      </footer>
    </div>
  );
}
