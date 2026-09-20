import { Component, computed, inject, input, signal } from '@angular/core';
import { Beegent, Link } from '../api/beegent.service';
import { LinkMapComponent } from './link-map.component';
import { mapHint, mappable } from './link-map.util';

/** One stored link. `measured` and `claimed` are styled apart, never merged. */
@Component({
  selector: 'link-card',
  standalone: true,
  // LinkMapComponent is referenced ONLY inside the @defer block below. Any other reference
  // - even an unused one - makes Angular bundle deck.gl eagerly, silently and with no error.
  imports: [LinkMapComponent],
  template: `
    <article class="card" [class.dead]="link().status !== 'ok'">
      <h3>{{ link().dataset || 'untitled dataset' }}</h3>
      <a class="mono" [href]="link().resource_url" target="_blank" rel="noreferrer">
        {{ link().resource_url }}
      </a>

      <div class="measured mono">
        <span class="chip" [class.bad]="link().status !== 'ok'">
          HTTP {{ link().verification.status ?? '—' }}
        </span>
        <span>{{ link().verification.payload_type || 'unknown payload' }}</span>
        <span>{{ size(link().verification.total_size_bytes) }}</span>
        @if (link().verification.feature_count != null) {
          <span>{{ link().verification.feature_count }}{{ link().verification.count_is_exact ? '' : '+' }} features</span>
        }
      </div>

      <div class="actions">
        <button class="quiet" [disabled]="!drawable()" [title]="hint()"
                (click)="showMap.set(!showMap())">
          {{ showMap() ? 'Hide map' : 'Show on map' }}
        </button>
      </div>

      @if (showMap()) {
        @defer (on immediate) {
          <link-map [link]="link()" />
        } @placeholder {
          <p class="pending">loading the map engine…</p>
        } @error {
          <p class="pending bad">the map engine failed to load</p>
        }
      }

      <p class="claimed">
        <span>as advertised</span>
        {{ link().claim['edition'] || link().claim['vintage_date'] || 'nothing stated' }}
        @if (link().last_verified) { · found {{ link().last_verified.slice(0, 10) }} }
      </p>
    </article>
  `,
  styles: [`
    .card { background: var(--surface); border: 1px solid var(--line);
            border-radius: var(--radius); padding: 13px 15px; }
    .dead { opacity: .55; }
    h3 { margin: 0 0 4px; font-size: 15px; font-weight: 600; letter-spacing: -.01em; }
    .dead h3::after { content: " · no longer reachable"; color: var(--bad); font-weight: 400;
                      font-size: 12px; letter-spacing: 0; }
    a { display: block; font-size: 11.5px; word-break: break-all; text-decoration: none; }
    a:hover { text-decoration: underline; }

    /* Measured off the wire: mono, so it reads as an instrument reading. */
    .measured { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
                margin-top: 10px; font-size: 11.5px; color: var(--text); }
    .chip { border: 1px solid color-mix(in srgb, var(--ok) 45%, transparent);
            color: var(--ok); border-radius: 5px; padding: 1px 6px; font-size: 11px; }
    .chip.bad { border-color: color-mix(in srgb, var(--bad) 45%, transparent); color: var(--bad); }

    .actions { margin-top: 10px; }
    button { font: inherit; font-size: 11.5px; padding: 3px 9px; border-radius: 6px;
             border: 1px solid var(--line); background: transparent; color: var(--text);
             cursor: pointer; }
    button:hover:not(:disabled) { border-color: var(--dim); }
    button:disabled { color: var(--faint); cursor: not-allowed; }
    .pending { margin: 10px 0 0; font-size: 12px; color: var(--dim); }
    .pending.bad { color: var(--bad); }

    /* Model-written: dim prose, visibly a different kind of fact. */
    .claimed { margin: 6px 0 0; font-size: 12px; color: var(--dim); }
    .claimed span { color: var(--faint); margin-right: 6px; }
  `],
})
export class LinkCardComponent {
  readonly link = input.required<Link>();
  readonly showMap = signal(false);

  private readonly api = inject(Beegent);

  /** Format only. Rot and the size cap are the server verdict, so each has ONE owner. */
  readonly drawable = computed(() =>
    (this.api.previewConfig()?.enabled ?? true) && mappable(this.link().verification));

  readonly hint = computed(() => {
    const cfg = this.api.previewConfig();
    return mapHint(this.link(), cfg?.enabled ?? true, cfg?.max_bytes ?? 0);
  });

  /** Bytes are meaningless for a service: one page's length is not the dataset's size. */
  size(bytes: number | null | undefined): string {
    if (bytes == null) return 'size n/a';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0, n = bytes;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
  }
}
