import {
  AfterViewInit, Component, ElementRef, NgZone, OnDestroy, input, signal, viewChild,
} from '@angular/core';
import { Deck, WebMercatorViewport } from '@deck.gl/core';
import { BitmapLayer, GeoJsonLayer } from '@deck.gl/layers';
import { TileLayer } from '@deck.gl/geo-layers';

import { Beegent, Link } from '../api/beegent.service';
import {
  Bbox, bboxOf, capFeatures, describeBbox, esriToGeoJson, looksWgs84, sourceCrs, toWgs84,
} from './link-map.util';

/**
 * The basemap: OpenStreetMap standard, with Esri dark gray canvas behind the button. Named
 * constants, so repointing them is one line - which matters, because the last provider we tried
 * started requiring a key without warning.
 *
 * NOTE the two paths are NOT the same shape: OSM is `{z}/{x}/{y}`, Esri is `{z}/{y}/{x}` - y
 * before x. Reversing either returns HTTP 200 and a coherent map of the wrong place, verified.
 * deck.gl substitutes only {x}, {y}, {z} and {-y}, so a provider template carrying `{s}` or `{r}`
 * keeps those braces in the request and an unresolvable `{s}` host fails every tile in silence.
 *
 * And why neither of these is CARTO: their keyless endpoint answers 200 with a valid PNG that has
 * API KEY REQUIRED stamped across it. Status, content type and size all looked right. A basemap is
 * verified by LOOKING at a tile, never by its status code.
 */
const BASE_LIGHT = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const BASE_DARK = 'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/' +
                  'World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}';

/** One layer of a payload. A GeoPackage yields several; everything else yields exactly one. */
interface Layer {
  name: string;
  count: number;
  fc: any;
  /** What this arrived in. Empty means degrees already, or converted during parsing. */
  crs: string;
  /** The loader applied the file own SRS. Distinct from a payload that declared nothing. */
  preconverted?: boolean;
}

/**
 * The verification a probe cannot do: does this dataset actually cover the right country.
 *
 * Everything here runs OUTSIDE the Angular zone. deck.gl installs a requestAnimationFrame
 * loop and pointer listeners, so inside the zone every frame of a pan would trigger a full
 * change-detection pass over the whole app.
 */
@Component({
  selector: 'link-map',
  standalone: true,
  template: `
    <div class="map" [class.tall]="tall()" #host>
      <canvas #cv></canvas>

      @if (layers().length) {
        <div class="chips">
          @for (l of layers(); track l.name; let i = $index) {
            <button [class.on]="i === selected()" (click)="show(i)"
                    [title]="i === selected()
                      ? 'Hide ' + l.name
                      : l.name + ' — ' + l.count + ' feature(s)'">
              {{ l.name }} <span class="n">{{ l.count }}</span>
            </button>
          }
        </div>
      }

      @if (message()) { <p class="msg" [class.bad]="failed()">{{ message() }}</p> }

      <!-- Inside the container on purpose: only the fullscreened subtree renders. -->
      @if (readout()) { <p class="readout mono">{{ readout() }}</p> }

      <div class="corner">
        @if (basemapDark()) {
          <a href="https://www.esri.com/en-us/legal/copyright-trademarks"
             target="_blank" rel="noreferrer">Esri, HERE, Garmin</a>
        }
        <a href="https://www.openstreetmap.org/copyright"
           target="_blank" rel="noreferrer">© OpenStreetMap contributors</a>
        <button class="grow" (click)="flipBasemap()"
                [title]="basemapDark() ? 'Light basemap' : 'Dark basemap'">
          {{ basemapDark() ? '☀' : '☾' }}
        </button>
        <button class="grow" (click)="bigger()"
                [title]="full() ? 'Exit full screen' : 'Full screen'">
          {{ full() ? '⛶ exit' : '⛶' }}
        </button>
      </div>
    </div>
  `,
  styles: [`
    /* resize needs a non-visible overflow; hidden keeps the canvas from spawning scrollbars. */
    .map { position: relative; height: 320px; margin-top: 10px; overflow: hidden;
           resize: vertical; min-height: 200px;
           border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    .map.tall { height: 640px; }
    /* Without this the map stays 320px tall on an otherwise black screen. */
    .map:fullscreen { height: 100%; width: 100%; margin: 0; border: 0; border-radius: 0;
                      resize: none; }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; }

    .chips { position: absolute; top: 6px; left: 6px; z-index: 2; display: flex; flex-wrap: wrap;
             gap: 4px; max-width: calc(100% - 12px); }
    .chips button { font: inherit; font-size: 11px; padding: 2px 8px; border-radius: 999px;
                    border: 1px solid var(--line); cursor: pointer; color: var(--text);
                    background: color-mix(in srgb, var(--surface) 88%, transparent); }
    .chips button:hover { border-color: var(--dim); }
    .chips button.on { border-color: var(--text); background: var(--surface); font-weight: 600; }
    .chips .n { color: var(--dim); font-weight: 400; }

    .msg { position: absolute; inset: 0; margin: 0; display: flex; align-items: center;
           justify-content: center; padding: 0 18px; text-align: center; font-size: 12.5px;
           color: var(--dim); background: var(--surface); z-index: 3; }
    .msg.bad { color: var(--bad); }

    /* Measured, like the card's own instrument readings - so it reads as one. */
    .readout { position: absolute; left: 6px; bottom: 5px; margin: 0; z-index: 2;
               font-size: 11px; color: var(--text); padding: 1px 6px; border-radius: 4px;
               max-width: calc(100% - 290px);
               background: color-mix(in srgb, var(--surface) 80%, transparent); }

    .corner { position: absolute; right: 4px; bottom: 3px; z-index: 2; display: flex;
              align-items: center; gap: 6px; font-size: 10px;
              background: color-mix(in srgb, var(--surface) 80%, transparent);
              padding: 1px 5px; border-radius: 4px; }
    .corner a { color: var(--dim); text-decoration: none; }
    .corner a:hover { text-decoration: underline; }
    .grow { font: inherit; font-size: 11px; line-height: 1; padding: 1px 4px; cursor: pointer;
            border: 1px solid var(--line); border-radius: 4px; background: transparent;
            color: var(--text); }
  `],
})
export class LinkMapComponent implements AfterViewInit, OnDestroy {
  readonly link = input.required<Link>();

  private readonly host = viewChild.required<ElementRef<HTMLDivElement>>('host');
  private readonly canvas = viewChild.required<ElementRef<HTMLCanvasElement>>('cv');

  readonly message = signal('loading the map engine…');
  readonly failed = signal(false);
  readonly readout = signal('');
  readonly layers = signal<Layer[]>([]);
  /** null means nothing is drawn - the basemap alone, which is how you check what is under it. */
  readonly selected = signal<number | null>(0);
  readonly full = signal(false);
  readonly tall = signal(false);

  private deck?: Deck<any>;
  private observer?: ResizeObserver;
  /** The FeatureCollection currently on screen, so the basemap can be swapped without it. */
  private drawn: any = null;
  private readonly abort = new AbortController();

  /** Light Positron unless asked otherwise. A per-map choice, not a system one. */
  readonly basemapDark = signal(false);

  private readonly onFullscreen = () =>
    this.zone.run(() => this.full.set(document.fullscreenElement === this.host().nativeElement));

  constructor(private zone: NgZone, private api: Beegent) {}

  ngAfterViewInit(): void {
    this.zone.runOutsideAngular(() => {
      document.addEventListener('fullscreenchange', this.onFullscreen);
      void this.build();
    });
  }

  ngOnDestroy(): void {
    this.abort.abort();
    this.observer?.disconnect();
    document.removeEventListener('fullscreenchange', this.onFullscreen);
    // Not optional: a leaked WebGL context per collapse/expand hits the browser cap after a
    // dozen toggles, and the map then stops drawing with only a console warning.
    this.deck?.finalize();
  }

  /** Swap the tiles and nothing else - re-rendering here would refit the camera. */
  flipBasemap(): void {
    this.basemapDark.set(!this.basemapDark());
    this.deck?.setProps({ layers: this.stack() });
  }

  /** Full screen where the browser has it, a taller box where it does not (iOS Safari). */
  bigger(): void {
    const el = this.host().nativeElement;
    if (!el.requestFullscreen) { this.tall.set(!this.tall()); return; }
    if (document.fullscreenElement === el) void document.exitFullscreen();
    else void el.requestFullscreen().catch(() => this.tall.set(!this.tall()));
  }

  /**
   * Clicking the visible layer hides it; clicking another swaps. Never two at once.
   *
   * No refetch and no re-parse either way - every table was parsed on the way in.
   */
  show(index: number): void {
    this.selected.set(index === this.selected() ? null : index);
    this.zone.runOutsideAngular(() => void this.render());
  }

  private say(text: string, bad = false): void {
    this.zone.run(() => { this.message.set(text); this.failed.set(bad); });
  }

  private async build(): Promise<void> {
    let found: Layer[];
    try {
      found = await this.load();
    } catch (err: any) {
      if (this.abort.signal.aborted) return;
      this.say(`${err?.message || err}`, true);
      return;
    }
    if (!found.length) {
      this.say('nothing to draw: the payload held no features', true);
      return;
    }
    this.zone.run(() => { this.layers.set(found); this.selected.set(0); });
    await this.render();
  }

  /** Cap first, then convert: reprojecting features that will not be drawn is wasted work. */
  private async render(): Promise<void> {
    const index = this.selected();
    if (index === null) {
      // Hidden: keep the camera exactly where it is, so showing it again is not a reset.
      this.drawn = null;
      this.say('');
      this.zone.run(() => this.readout.set(''));
      this.deck?.setProps({ layers: this.stack() });
      return;
    }

    const layer = this.layers()[index];
    if (!layer) return;
    this.say('');

    const { data, shown, total } = capFeatures(layer.fc);
    if (!shown) {
      this.say(`nothing to draw: ${layer.name} held no features`, true);
      return;
    }

    let converted;
    try {
      converted = await toWgs84(data, layer.crs);
    } catch (err: any) {
      this.say(`could not convert ${layer.crs} to WGS84: ${err?.message || err}`, true);
      return;
    }

    const bbox = bboxOf(converted.data);
    if (!bbox) {
      this.say('nothing to draw: no coordinates in the payload', true);
      return;
    }
    if (!looksWgs84(bbox)) {
      // Everything with a declared CRS was converted above, so reaching here means the payload
      // is projected and says nothing about how. There is no source to convert FROM.
      // Three different situations, and a reader responds differently to each.
      const why = converted.reason
        || (layer.preconverted ? 'its own CRS was already applied' : 'it declares no CRS');
      this.say(`these coordinates are not degrees (${describeBbox(bbox)}) and ${why}, ` +
               `so there is nothing to convert from and no map is drawn.`, true);
      return;
    }

    try {
      this.draw(converted.data, bbox);
    } catch (err: any) {
      this.say(`the map engine failed to start: ${err?.message || err}`, true);
      return;
    }

    const partial = shown < total ? ` — first ${shown} of ${total}, a partial view` : '';
    const paged = this.link().verification?.access === 'api' ? ' · one page of a paged service' : '';
    const from = converted.from ? ` · converted from ${converted.from}` : '';
    const name = this.layers().length > 1 ? `${layer.name} · ` : '';
    this.zone.run(() => this.readout.set(
      `${name}${this.link().country}: ${describeBbox(bbox)} · ` +
      `${shown} feature(s)${paged}${partial}${from}`));
  }

  /** Fetched with fetch(), not HttpClient: a refusal keeps the detail the server wrote. */
  private async load(): Promise<Layer[]> {
    this.say('fetching the data…');
    const res = await fetch(this.api.previewUrl(this.link().link_uid), { signal: this.abort.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `the server returned HTTP ${res.status}`);
    }
    const v = this.link().verification || {};
    if (v.payload_type === 'sqlite/geopackage') return this.loadGeoPackage(res);

    const doc = await res.json();
    const fc = v.shape === 'esrijson_featureset' ? esriToGeoJson(doc) : doc;
    return [{ name: 'features', count: fc?.features?.length || 0, fc, crs: sourceCrs(doc) }];
  }

  /** A GeoPackage IS a SQLite database, so reading one means running SQL - here, in WASM. */
  private async loadGeoPackage(res: Response): Promise<Layer[]> {
    this.say('reading the GeoPackage…');
    const [{ parse }, { GeoPackageLoader }] = await Promise.all([
      import('@loaders.gl/core'),
      import('@loaders.gl/geopackage'),
    ]);
    const parsed: any = await parse(await res.arrayBuffer(), GeoPackageLoader as any, {
      // Both defaults reach for a third-party CDN: sqlJsCDN is a URL PREFIX that locateFile
      // appends the filename to, and the worker bundle comes from unpkg. Neither is wanted
      // in an app that is otherwise localhost-only.
      worker: false,
      geopackage: { sqlJsCDN: 'assets/', shape: 'tables' },
      // The file carries its own SRS in gpkg_spatial_ref_sys, and the loader bundles proj4 - so
      // a GeoPackage arrives here already in degrees, whatever country it came from. It is the
      // one payload whose CRS needs no lookup table at all.
      gis: { reproject: true, _targetCrs: 'WGS84' },
    } as any);

    const tables: any[] = parsed?.tables || [];
    if (!tables.length) throw new Error('the GeoPackage held no feature table');
    // Every table was parsed and reprojected on the way in, so switching between them later
    // costs nothing. That is what makes the picker worth having rather than a second download.
    return tables.map((t) => ({
      name: t.name,
      count: t.table?.features?.length || 0,
      fc: t.table,
      crs: '',
      preconverted: true,
    }));
  }

  /** Basemap first, then the payload if one is showing. Rebuilt whenever either changes. */
  private stack(): any[] {
    const layers: any[] = [this.basemap()];
    if (this.drawn) {
      layers.push(new GeoJsonLayer({
        // The SAME id across switches, so deck.gl diffs the data instead of rebuilding.
        id: 'payload',
        data: this.drawn,
        filled: true,
        stroked: true,
        pickable: false,
        getFillColor: [37, 99, 235, 55],
        getLineColor: [37, 99, 235, 210],
        getPointRadius: 4,
        pointRadiusUnits: 'pixels',
        lineWidthMinPixels: 1,
      }));
    }
    return layers;
  }

  private draw(data: any, bbox: Bbox): void {
    const el = this.host().nativeElement;
    const { width, height } = el.getBoundingClientRect();
    const view = new WebMercatorViewport({ width: width || 600, height: height || 320 })
      .fitBounds([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], { padding: 24 });
    const viewState = { longitude: view.longitude, latitude: view.latitude, zoom: view.zoom };
    this.drawn = data;

    if (this.deck) {
      // A fresh initialViewState object is what makes deck.gl refit to the new layer.
      this.deck.setProps({ layers: this.stack(), initialViewState: viewState });
      return;
    }

    this.deck = new Deck({
      canvas: this.canvas().nativeElement,
      width: width || 600,
      height: height || 320,
      controller: true,
      initialViewState: viewState,
      layers: this.stack(),
    });

    // A canvas with no layout size draws nothing and throws nothing - the classic blank map.
    // This also covers the resize handle, and entering or leaving full screen.
    this.observer = new ResizeObserver(() => {
      const box = el.getBoundingClientRect();
      if (box.width && box.height) this.deck?.setProps({ width: box.width, height: box.height });
    });
    this.observer.observe(el);
  }

  private basemap(): TileLayer<any> {
    return new TileLayer({
      id: 'basemap',
      data: this.basemapDark() ? BASE_DARK : BASE_LIGHT,
      minZoom: 0,
      maxZoom: 19,
      tileSize: 256,
      renderSubLayers: (props: any) => {
        // deck.gl 9 exposes [[w,s],[e,n]]; an 8.x example uses tile.bbox and draws nothing.
        const [[west, south], [east, north]] = props.tile.boundingBox;
        return new BitmapLayer({
          ...props,
          data: undefined,
          image: props.data,
          bounds: [west, south, east, north],
        });
      },
    });
  }
}
