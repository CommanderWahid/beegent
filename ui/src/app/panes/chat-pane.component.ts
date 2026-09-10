import { Component, ElementRef, effect, input, output, viewChild } from '@angular/core';
import { Link, Message, RunFields, RunView } from '../api/beegent.service';
import { LinkCardComponent } from './link-card.component';

/** The conversation. A run appears as ONE card that grows, never as a line per message. */
@Component({
  selector: 'chat-pane',
  standalone: true,
  imports: [LinkCardComponent],
  template: `
    <div class="thread" #thread>
      <div class="column">
      @for (m of messages(); track $index) {
        @if (m.who === 'you') {
          <p class="you" [innerHTML]="m.html"></p>
        } @else {
          <div class="agent">
            <div [innerHTML]="m.html"></div>

            @if (m.links?.length) {
              <div class="cards">
                @for (l of preview(m.links!); track l.link_uid) { <link-card [link]="l" /> }
              </div>
              @if (m.links!.length > 3) {
                <button class="quiet more" (click)="openCatalog.emit(m.country || '')">
                  See all {{ m.links!.length }} in the catalog →
                </button>
              }
            }

            @if (m.confirm) {
              <div class="confirm">
                <dl>
                  <dt>country</dt><dd>{{ m.confirm.country }}</dd>
                  <dt>use case</dt><dd>{{ m.confirm.use_case }}</dd>
                  <dt>format</dt>
                  <dd>{{ m.confirm.format
                         || (m.confirm.channel === 'api'
                             ? 'API endpoint (a live service)' : 'any the publisher serves') }}</dd>
                  <dt>vintage</dt><dd>{{ m.confirm.vintage || 'latest' }}</dd>
                  @if (m.confirm.sent_as) {
                    <dt>sent as</dt><dd class="sent">{{ m.confirm.sent_as }}</dd>
                  }
                </dl>
                <footer>
                  <span>Up to 120 model calls.</span>
                  <button (click)="start.emit(m.confirm!)" [disabled]="!!run()?.running">Search</button>
                </footer>
              </div>
            }
          </div>
        }
      }

      @if (busy()) {
        <div class="agent waiting">
          <img src="beegent_logo.svg" alt="" width="20" height="20" class="bee" />
          <span>Thinking…</span>
          @if (waited() > 2) { <span class="elapsed">{{ waited() }}s</span> }
          @if (waited() > 20 && model()) {
            <p class="slow">Still waiting on <b>{{ model() }}</b>. A local reasoning model can
              take minutes for one reply; a hosted backend answers in seconds.</p>
          }
        </div>
      }

      @if (run(); as r) {
        <section class="run">
          @if (r.running) {
            <p class="working">
              <img src="beegent_logo.svg" alt="" width="20" height="20" />
              <span>{{ r.step }}</span>
              @if (waited() > 2) { <span class="elapsed">{{ waited() }}s</span> }
              <button class="quiet logs" (click)="openLogs.emit()">View logs</button>
            </p>
          }

          @if (r.summary; as s) {
            <p class="done"><b>{{ status(s) }}</b> — {{ verified(s) }} of
              {{ s.candidates?.length || 0 }} candidate(s) independently probed</p>
            <!-- Effort, not cost: tokens belong in the log, not in the conversation. -->
            <p class="effort">{{ s.totals?.angles_run || 0 }} route(s) ·
               {{ s.totals?.http_requests || 0 }} request(s)</p>
            @if (s.critic_note) { <p class="note">{{ s.critic_note }}</p> }
          }
        </section>
      }
      </div>
    </div>
  `,
  styles: [`
    :host { display: flex; flex-direction: column; flex: 1; min-height: 0; }
    .thread { flex: 1; min-height: 0; overflow-y: auto; padding: 28px 32px 10px; }
    /* One column, so the transcript lines up with the composer below it. */
    .column { max-width: var(--measure); margin: 0 auto;
              display: flex; flex-direction: column; gap: 18px; }

    /* Only DATA gets a container; agent prose is just text. */
    .agent { font-size: 14.5px; }
    .you { align-self: flex-end; max-width: 46ch; margin: 0; padding: 8px 14px;
           background: var(--accent); color: #fff; border-radius: 14px 14px 4px 14px; }

    .cards { display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
    .cards ::ng-deep .card { background: none; border: none; padding: 0;
                             border-top: 1px solid var(--line); padding-top: 11px; }
    .more { margin-top: 8px; padding: 5px 10px; font-size: 12.5px; color: var(--accent); }

    /* No panel in the conversation: the page ground carries it, spacing does the rest. */
    .confirm { margin-top: 12px; }
    .confirm dl { display: grid; grid-template-columns: 88px 1fr; gap: 5px 12px;
                  margin: 0; font-size: 13px; }
    .confirm footer { display: flex; align-items: center; gap: 14px;
                      margin-top: 12px; font-size: 12px; color: var(--dim); }
    dt { color: var(--faint); }
    .sent { color: var(--dim); }
    dd { margin: 0; }

    /* The run reads as part of the conversation, never as a panel dropped into it. */
    .working { display: flex; align-items: center; gap: 10px; margin: 0; font-size: 13.5px; }
    .working img { animation: hover 1.6s ease-in-out infinite; }
    @keyframes hover { 50% { transform: translateY(-2px); } }
    .done { margin: 0 0 8px; font-size: 14px; }

    .elapsed { color: var(--faint); font-size: 12px; }
    .logs { padding: 3px 10px; font-size: 12px; color: var(--accent); }

    .effort { margin: 2px 0 0; font-size: 12.5px; color: var(--faint); }
    .note { margin: 8px 0 0; font-size: 13px; color: var(--dim); }

    .waiting { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
    .elapsed { font-size: 12px; color: var(--faint); }
    .slow { flex-basis: 100%; margin: 0; font-size: 12.5px; color: var(--dim); }
    .waiting { font-size: 13.5px; }
    .waiting .bee { animation: hover 1.6s ease-in-out infinite; }
  `],
})
export class ChatPaneComponent {
  readonly messages = input.required<Message[]>();
  readonly busy = input(false);
  readonly waited = input(0);
  readonly model = input('');
  readonly run = input<RunView | null>(null);
  readonly start = output<RunFields>();
  readonly openCatalog = output<string>();
  readonly openLogs = output<void>();
  private readonly thread = viewChild<ElementRef<HTMLElement>>('thread');

  constructor() {
    effect(() => {
      this.messages(); this.run()?.step; this.busy();
      const el = this.thread()?.nativeElement;
      if (el) queueMicrotask(() => (el.scrollTop = el.scrollHeight));
    });
  }

  /** A long list belongs in the catalog, not stacked down the transcript. */
  preview = (links: Link[]) => links.slice(0, 3);

  status = (s: any) => ({ ok: 'Finished', stopped: 'Stopped' }[s.status as string]
    || 'Needs human review');

  verified = (s: any) => (s.candidates || []).filter((c: any) => c.resource_url).length;

}
