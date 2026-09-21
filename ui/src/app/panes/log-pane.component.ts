import { Component, ElementRef, effect, input, viewChild } from '@angular/core';
import { Stage } from './stages';

/** The raw stream. The chat shows milestones; everything the pipeline logged lands here. */
@Component({
  selector: 'log-pane',
  standalone: true,
  template: `
    @if (!lines().length) {
      <p class="none">No run yet. The full log appears here while one is in flight.</p>
    } @else {
      <pre #box>{{ lines().join('\n') }}</pre>

      <!-- A sibling of the log, never inside it: the pre auto-scrolls to the tail. -->
      <aside class="steps">
        @for (s of stages(); track s.id) {
          <div class="step" [class.active]="s.state === 'active'" [class.done]="s.state === 'done'">
            @if (s.agent) {
              <img src="beegent_logo.svg" alt="agent step" width="16" height="16" class="bee" />
            } @else {
              <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" class="cog">
                <g fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round">
                  <circle cx="8" cy="8" r="2.4" />
                  <path d="M8 1.2v2M8 12.8v2M1.2 8h2M12.8 8h2M3.2 3.2l1.4 1.4M11.4
                           11.4l1.4 1.4M12.8 3.2l-1.4 1.4M4.6 11.4l-1.4 1.4" />
                </g>
              </svg>
            }
            <span class="name">{{ s.name }}</span>
            <span class="detail">{{ s.detail }}</span>
          </div>
        }
      </aside>
    }
  `,
  styles: [`
    :host { display: flex; gap: 16px; flex: 1; min-height: 0; padding: 20px 24px 8px; }
    pre { margin: 0; flex: 1; min-height: 0; overflow: auto; white-space: pre-wrap; word-break: break-word;
          font-size: 11.5px; line-height: 1.6; color: var(--dim);
          background: var(--sunken); border: 1px solid var(--line);
          border-radius: var(--radius); padding: 14px 16px; }
    .none { color: var(--faint); font-size: 13px; }

    .steps { flex: 0 0 210px; overflow: auto; display: flex; flex-direction: column; gap: 2px; }
    /* Shape says which kind of step it is; colour says what it is doing right now. */
    .step { display: grid; grid-template-columns: 18px 1fr; gap: 2px 8px; align-items: center;
            padding: 7px 10px; border-radius: var(--radius-sm); color: var(--faint);
            font-size: 12px; }
    .step.done { color: var(--dim); }
    .step.active { color: var(--accent); background: var(--accent-soft); }
    .name { font-weight: 500; }
    .detail { grid-column: 2; font-size: 11px; color: var(--faint); }
    .step.active .detail { color: var(--accent); }

    /* An img cannot inherit currentColor, so the bee shows its state by coming alive. */
    .bee { filter: grayscale(1); opacity: .45; }
    .step.active .bee { filter: none; opacity: 1; animation: pulse 1.6s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: .5; } }

    /* Narrow panes stack it above the log rather than squeezing both. */
    @media (max-width: 900px) {
      :host { flex-direction: column-reverse; }
      .steps { flex: 0 0 auto; flex-direction: row; flex-wrap: wrap; }
    }
  `],
})
export class LogPaneComponent {
  readonly lines = input.required<string[]>();
  readonly stages = input.required<Stage[]>();
  private readonly box = viewChild<ElementRef<HTMLElement>>('box');

  constructor() {
    // Follow the tail, which is the only part anyone watches during a run.
    effect(() => {
      this.lines();
      const el = this.box()?.nativeElement;
      if (el) queueMicrotask(() => (el.scrollTop = el.scrollHeight));
    });
  }
}
