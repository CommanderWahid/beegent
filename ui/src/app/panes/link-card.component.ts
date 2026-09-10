import { Component, input } from '@angular/core';
import { Link } from '../api/beegent.service';

/** One stored link. `measured` and `claimed` are styled apart, never merged. */
@Component({
  selector: 'link-card',
  standalone: true,
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

    /* Model-written: dim prose, visibly a different kind of fact. */
    .claimed { margin: 6px 0 0; font-size: 12px; color: var(--dim); }
    .claimed span { color: var(--faint); margin-right: 6px; }
  `],
})
export class LinkCardComponent {
  readonly link = input.required<Link>();

  /** Bytes are meaningless for a service: one page's length is not the dataset's size. */
  size(bytes: number | null | undefined): string {
    if (bytes == null) return 'size n/a';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    let i = 0, n = bytes;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i ? 1 : 0)} ${u[i]}`;
  }
}
