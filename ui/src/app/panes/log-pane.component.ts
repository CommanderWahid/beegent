import { Component, ElementRef, effect, input, viewChild } from '@angular/core';

/** The raw stream. The chat shows milestones; everything the pipeline logged lands here. */
@Component({
  selector: 'log-pane',
  standalone: true,
  template: `
    @if (!lines().length) {
      <p class="none">No run yet. The full log appears here while one is in flight.</p>
    } @else {
      <pre #box>{{ lines().join('\n') }}</pre>
    }
  `,
  styles: [`
    :host { display: flex; flex-direction: column; flex: 1; min-height: 0;
            padding: 20px 24px 8px; }
    pre { margin: 0; flex: 1; min-height: 0; overflow: auto; white-space: pre-wrap; word-break: break-word;
          font-size: 11.5px; line-height: 1.6; color: var(--dim);
          background: var(--sunken); border: 1px solid var(--line);
          border-radius: var(--radius); padding: 14px 16px; }
    .none { color: var(--faint); font-size: 13px; }
  `],
})
export class LogPaneComponent {
  readonly lines = input.required<string[]>();
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
