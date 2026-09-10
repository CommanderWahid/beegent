import { Component, input, output } from '@angular/core';
import { Link } from '../api/beegent.service';
import { LinkCardComponent } from './link-card.component';

/** Browsing is a VIEW: picking a country re-renders it, it never appends to the chat. */
@Component({
  selector: 'catalog-pane',
  standalone: true,
  imports: [LinkCardComponent],
  template: `
    <header>
      <h2>{{ country() || 'All countries' }}</h2>
      <p>{{ links().length }} verified link{{ links().length === 1 ? '' : 's' }} ·
         re-probed on every run that matches</p>
    </header>

    @if (loading()) {
      @for (i of [1, 2, 3]; track i) { <div class="skeleton"></div> }
    } @else if (!links().length) {
      <div class="none">
        <p>Nothing stored{{ country() ? ' for ' + country() : '' }} yet.</p>
        <button class="quiet" (click)="ask.emit()">Ask for some data</button>
      </div>
    } @else {
      <div class="grid">
        @for (l of links(); track l.link_uid) { <link-card [link]="l" /> }
      </div>
    }
  `,
  styles: [`
    :host { flex: 1; min-height: 0; overflow-y: auto; padding: 26px 32px 8px; }
    header { max-width: var(--measure); margin: 0 auto 18px; }
    h2 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -.02em; }
    header p { margin: 3px 0 0; font-size: 12.5px; color: var(--dim); }
    .grid { display: flex; flex-direction: column; gap: 10px;
            max-width: var(--measure); margin: 0 auto; }

    .skeleton { height: 92px; max-width: var(--measure); border-radius: var(--radius);
                margin: 0 auto 10px; background: var(--surface);
                border: 1px solid var(--line); animation: pulse 1.4s ease-in-out infinite; }
    @keyframes pulse { 50% { opacity: .45; } }

    .none { max-width: var(--measure); margin: 0 auto; text-align: center; padding: 46px 0;
            color: var(--dim); border: 1px dashed var(--line-firm); border-radius: var(--radius); }
    .none p { margin: 0 0 12px; }
  `],
})
export class CatalogPaneComponent {
  readonly links = input.required<Link[]>();
  readonly country = input('');
  readonly loading = input(false);
  readonly ask = output<void>();
}
