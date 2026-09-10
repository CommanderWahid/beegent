import { Component, OnInit, computed, signal } from '@angular/core';
import { Beegent, ChatReply, CountryCount, Link, Message, Resolved, RunFields, RunView }
  from './api/beegent.service';
import { ChatPaneComponent } from './panes/chat-pane.component';
import { CatalogPaneComponent } from './panes/catalog-pane.component';
import { LogPaneComponent } from './panes/log-pane.component';

type Mode = 'chat' | 'catalog' | 'logs';

/** A log line to plain words. The chat says WHAT is happening; the Logs tab keeps the detail. */
const STEPS: [RegExp, string][] = [
  [/^fetch_page\b/, 'Reading a page…'],
  [/^web_search\b/, 'Searching the web…'],
  [/^probe_url\b/, 'Checking a download…'],
  [/^\[plan\]/, 'Planning routes…'],
  [/^\[(catalog|memory)\]/, 'Checking what I already hold…'],
  [/^\[geofetch\] angle (\d+)\/(\d+)/, 'Working on route $1 of $2…'],
  [/^\[geofetch\]/, 'Following a route…'],
  [/^\[(gate|critic)\]/, 'Reviewing what was found…'],
  [/^\[error\]/, 'Something went wrong — see the logs.'],
];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [ChatPaneComponent, CatalogPaneComponent, LogPaneComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  readonly mode = signal<Mode>('chat');
  readonly countries = signal<CountryCount[]>([]);
  readonly country = signal('');
  readonly catalog = signal<Link[]>([]);
  readonly catalogLoading = signal(false);
  readonly greetingLead = signal('');
  readonly greetingRest = signal('');
  readonly greetingPlain = signal('');
  readonly details = signal(false);
  readonly messages = signal<Message[]>([]);
  readonly draft = signal('');
  readonly busy = signal(false);
  readonly run = signal<RunView | null>(null);
  readonly log = signal<string[]>([]);
  readonly resolved = signal<Resolved | null>(null);
  private pending?: { unsubscribe(): void };
  readonly waited = signal(0);
  private ticker?: ReturnType<typeof setInterval>;

  /** The front door: centred until the conversation actually starts. */
  readonly blank = computed(() =>
    this.mode() === 'chat' && !this.messages().length && !this.run());
  readonly running = computed(() => !!this.run()?.running);
  /** A chat turn or a whole run: either way the agent is busy and cannot take another. */
  readonly locked = computed(() => this.busy() || this.running());

  constructor(private api: Beegent) {}

  ngOnInit(): void {
    this.refresh();
    this.api.greeting().subscribe((g) => this.lede(g.reply));
    this.api.config().subscribe((c) => this.resolved.set(c));
  }

  /** One line up front; the rest is the same text, folded away - never a second copy of it. */
  private lede(reply: string): void {
    const [lead, ...rest] = reply.split('\n\n');
    this.greetingLead.set(this.markdown(lead));
    this.greetingRest.set(this.markdown(rest.join('\n\n')));
    this.greetingPlain.set(rest.join(' ').replace(/\*\*/g, ''));
  }

  /** The model the chat is actually waiting on - the front door uses the planner role. */
  chatModel = () => this.resolved()?.models?.['planner'] || '';

  private wait(on: boolean): void {
    clearInterval(this.ticker);
    this.waited.set(0);
    if (!on) return;
    const started = Date.now();
    this.ticker = setInterval(() => this.waited.set(Math.round((Date.now() - started) / 1000)), 1000);
  }

  total = () => this.countries().reduce((n, c) => n + c.links, 0);

  /** Browsing REPLACES the view. It must never append to the transcript. */
  pick(country: string): void {
    this.country.set(country);
    this.mode.set('catalog');
    this.catalogLoading.set(true);
    this.api.links(country).subscribe({
      next: (links) => { this.catalog.set(links); this.catalogLoading.set(false); },
      error: () => { this.catalog.set([]); this.catalogLoading.set(false); },
    });
  }

  /** Enter sends; Shift+Enter is a new line, which a multi-line box has to allow. */
  key(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) this.send(event);
  }

  send(event: Event): void {
    event.preventDefault();
    const text = this.draft().trim();
    if (!text || this.locked()) return;
    this.mode.set('chat');
    this.messages.update((m) => [...m, { who: 'you', html: this.escape(text), text }]);
    this.draft.set('');
    this.busy.set(true);
    this.wait(true);

    this.pending = this.api.chat(text, this.history()).subscribe({
      next: (r) => { this.busy.set(false); this.wait(false); this.answer(r); },
      error: () => {
        this.busy.set(false); this.wait(false); this.say('The backend did not answer.');
      },
    });
  }

  start(fields: RunFields): void {
    this.log.set([]);
    // The confirmation is consumed by pressing Search - it must not stay live in history.
    this.messages.update((m) => m.map((msg) => ({ ...msg, confirm: undefined })));
    this.api.start(fields).subscribe({
      next: ({ run_id }) => {
        this.run.set({ id: run_id, country: fields.country,
                       use_case: fields.sent_as || fields.use_case,
                       running: true, step: 'Starting…' });
        this.wait(true);
        this.api.logs(run_id, (line) => this.line(line), () => this.finish(run_id));
      },
      // 409: the API allows one run at a time, because two would interleave store writes.
      error: (e) => this.say(e.status === 409
        ? 'A run is already in flight — one at a time.' : 'Could not start the run.'),
    });
  }

  /** A run cancels for real; a chat turn only stops us waiting, and says so. */
  stop(): void {
    const id = this.run()?.id;
    if (id && this.running()) {
      this.run.update((r) => r && { ...r, step: 'Stopping…' });
      this.api.stop(id).subscribe({ error: () => {} });
      return;
    }
    this.pending?.unsubscribe();
    this.pending = undefined;
    this.busy.set(false);
    this.wait(false);
    this.say('Stopped waiting. That model call may still finish on the server.');
  }

  private line(raw: string): void {
    this.log.update((l) => [...l, raw]);
    const text = raw.trim();
    for (const [pattern, label] of STEPS) {
      const found = text.match(pattern);
      if (!found) continue;
      // Built from the label alone: replacing IN the line would keep the URL after it.
      const step = label.replace(/\$(\d)/g, (_, i) => found[+i] ?? '');
      this.run.update((r) => r && { ...r, step });
      return;
    }
  }

  private finish(runId: string): void {
    this.run.update((r) => r && { ...r, running: false, step: '' });
    this.wait(false);
    this.refresh();
    this.api.result(runId).subscribe({
      next: (summary) => this.run.update((r) => r && { ...r, summary }),
      error: () => {},
    });
  }

  private answer(r: ChatReply): void {
    // A gated request shows the four fields BEFORE spending: the confirmation is the product.
    const confirm = r.ready
      ? { country: r.country, use_case: r.use_case, format: r.format, vintage: r.vintage,
          channel: r.channel, sent_as: r.sent_as }
      : undefined;
    this.say(r.reply || (confirm ? 'Ready when you are.' : '…'), r.links, confirm, r.country);
  }

  /** Text only, so the model is never fed back its own rendered cards. */
  private history = () => this.messages().slice(-6)
    .map((m) => ({ role: m.who === 'you' ? 'user' : 'assistant', content: m.text }));

  private refresh(): void {
    this.api.countries().subscribe((c) => this.countries.set(c));
    if (this.mode() === 'catalog') this.pick(this.country());
  }

  private say(text: string, links?: Link[], confirm?: Message['confirm'], country?: string): void {
    this.messages.update((m) =>
      [...m, { who: 'agent', html: this.markdown(text), text, links, confirm, country }]);
  }

  private escape(s: string): string {
    return s.replace(/[&<>"]/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
  }

  /** Escape FIRST, then the two marks the agent actually uses. Never render raw model output. */
  private markdown(s: string): string {
    return this.escape(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }
}
