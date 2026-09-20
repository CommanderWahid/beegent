import { Link, Verification } from '../api/beegent.service';

/** Drawn at most, whichever binds first. Vertex count is what actually costs, not features. */
export const MAX_FEATURES = 20_000;
export const MAX_COORDS = 2_000_000;

/** Everything is shown in this, always. deck.gl reads lon/lat degrees and nothing else. */
export const WGS84 = 'EPSG:4326';

/** Esri geometry names mapped to the OGC ones. Mirrors _ESRI_GEOMETRY in beegent/web_tools.py. */
const ESRI_GEOMETRY: Record<string, string> = {
  esriGeometryPoint: 'Point',
  esriGeometryMultipoint: 'MultiPoint',
  esriGeometryPolyline: 'MultiLineString',
  esriGeometryPolygon: 'Polygon',
};

/**
 * Whether a browser could draw this at all - keyed on what the HARNESS measured, never on
 * the URL or a file extension. The server re-checks the same thing, so a crafted request
 * cannot spend the download budget on a PDF.
 */
export function mappable(v: Verification | undefined): boolean {
  if (!v) return false;
  if (v.payload_type === 'sqlite/geopackage') return true;
  return v.payload_type === 'json-text' &&
    (v.shape === 'geojson_featurecollection' || v.shape === 'esrijson_featureset');
}

/** Why the button is greyed out, in the words of whatever actually blocked it. */
export function mapHint(link: Link, previewEnabled = true, maxBytes = 0): string {
  const v = link.verification || {};
  if (!previewEnabled) return 'No map: previews are off, PREVIEW_DIR is not set';
  if (mappable(v)) {
    const size = v.total_size_bytes;
    if (maxBytes && size != null && size > maxBytes) {
      return `Too large to preview: ${mb(size)}, the cap is ${mb(maxBytes)} (PREVIEW_MAX_BYTES)`;
    }
    return v.payload_type === 'sqlite/geopackage'
      ? 'Draw it on a map. GeoPackage support is experimental'
      : 'Draw it on a map';
  }
  if (v.payload_type === 'xml/html-text') {
    if (v.note) return `No map: ${v.note}`;
    if (v.shape === 'wfs_featurecollection') return 'No map: GML has no maintained browser parser';
    return 'No map: the probe saw a text or error page, not data';
  }
  if (v.payload_type === 'parquet') return 'No map yet: GeoParquet in the browser is experimental';
  if (!v.payload_type) return 'No map: the payload was never identified';
  return `No map: a ${v.payload_type} cannot be drawn in the browser`;
}

export function mb(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

/** An Esri FeatureSet as GeoJSON. Coordinates are left in whatever CRS they arrived in. */
export function esriToGeoJson(doc: any): any {
  const kind = ESRI_GEOMETRY[doc?.geometryType] || 'Polygon';
  const features = (doc?.features || []).map((f: any) => {
    const g = f.geometry || {};
    let geometry: any = null;
    if (kind === 'Point' && g.x != null) geometry = { type: 'Point', coordinates: [g.x, g.y] };
    else if (g.rings) geometry = { type: 'Polygon', coordinates: g.rings };
    else if (g.paths) geometry = { type: 'MultiLineString', coordinates: g.paths };
    else if (g.points) geometry = { type: 'MultiPoint', coordinates: g.points };
    return { type: 'Feature', geometry, properties: f.attributes || {} };
  });
  return { type: 'FeatureCollection', features: features.filter((f: any) => f.geometry) };
}

// --- coordinate reference systems -------------------------------------------

/** `urn:ogc:def:crs:EPSG::28992`, `http://…/def/crs/EPSG/0/28992`, `EPSG:28992` → `EPSG:28992`. */
function normaliseCrsName(raw: string): string {
  const name = raw.trim();
  if (/CRS84/i.test(name)) return 'CRS84';
  const epsg = name.match(/EPSG[:/]{1,2}(?:0\/)?(\d{4,6})/i) || name.match(/(?:^|:)(\d{4,6})$/);
  return epsg ? `EPSG:${epsg[1]}` : name;
}

/**
 * The CRS a payload DECLARES, or "" when it declares none.
 *
 * Only a declaration counts. Coordinates alone cannot identify a projection, so a payload that
 * says nothing is treated as CRS84 - which is what RFC 7946 requires of GeoJSON anyway.
 */
export function sourceCrs(doc: any): string {
  const sr = doc?.spatialReference;               // Esri FeatureSet
  if (sr) {
    const code = sr.latestWkid || sr.wkid;
    if (code) return `EPSG:${code}`;
  }
  const name = doc?.crs?.properties?.name;        // GeoJSON 2008 legacy member
  if (typeof name === 'string' && name) return normaliseCrsName(name);
  return '';
}

/**
 * Whether coordinates in this CRS are already lon/lat degrees.
 *
 * A declared EPSG:4326 is treated as lon/lat, not the lat/lon axis order the register
 * specifies. Every GeoJSON in the wild writes lon/lat, and honouring the register here would
 * put Dutch data in Somalia. This is an assumption, not something the data can confirm - a
 * swap that stays inside valid ranges is undetectable.
 */
export function isDegrees(crs: string): boolean {
  return !crs || crs === 'CRS84' || crs === 'EPSG:4326' || crs === 'EPSG:4269';
}

function mapCoords(node: any, fn: (p: number[]) => number[]): any {
  if (!Array.isArray(node)) return node;
  if (typeof node[0] === 'number') return fn(node as number[]);
  return node.map((child) => mapCoords(child, fn));
}

function mapGeometry(geom: any, fn: (p: number[]) => number[]): any {
  if (!geom) return geom;
  if (geom.type === 'GeometryCollection') {
    return { ...geom, geometries: (geom.geometries || []).map((g: any) => mapGeometry(g, fn)) };
  }
  return { ...geom, coordinates: mapCoords(geom.coordinates, fn) };
}

export interface Converted {
  data: any;
  /** The CRS converted FROM, empty when nothing had to be done. */
  from: string;
  /** Why it could not be converted, empty on success. */
  reason: string;
}

/**
 * Every geometry ends up in EPSG:4326 degrees, whatever it arrived as.
 *
 * proj4 needs a DEFINITION, not a code, and ships only a handful (4326, 4269, 3857). The full
 * table is ~1MB, so it is imported here, lazily, and only when a payload actually declares a
 * projected CRS - a GeoPackage never reaches this (its own gpkg_spatial_ref_sys already told
 * the loader what to do) and neither does anything already in degrees.
 */
export async function toWgs84(fc: any, crs: string): Promise<Converted> {
  if (isDegrees(crs)) return { data: fc, from: '', reason: '' };

  const proj4 = ((await import('proj4')) as any).default;
  if (!proj4.defs(crs)) {
    const mod: any = await import('proj4-list');
    const list = mod.default ?? mod;
    const entry = list[crs] || list[crs.toUpperCase()];
    if (!entry) return { data: fc, from: crs, reason: `no definition for ${crs}` };
    proj4.defs(entry[0], entry[1]);
  }

  const project = proj4(crs, WGS84);
  const move = (p: number[]) => {
    const [x, y] = project.forward([p[0], p[1]]);
    // A third ordinate is an elevation, not a coordinate this projection touches.
    return p.length > 2 ? [x, y, p[2]] : [x, y];
  };
  const features = (fc?.features || []).map((f: any) => ({
    ...f, geometry: mapGeometry(f.geometry, move),
  }));
  return { data: { type: 'FeatureCollection', features }, from: crs, reason: '' };
}

function countCoords(node: any, budget: { n: number }): void {
  if (!Array.isArray(node) || budget.n > MAX_COORDS) return;
  if (typeof node[0] === 'number') { budget.n++; return; }
  for (const child of node) countCoords(child, budget);
}

/** Take features from the front until either cap binds. A partial view, honestly labelled. */
export function capFeatures(fc: any): { data: any; shown: number; total: number } {
  const all: any[] = fc?.features || [];
  const kept: any[] = [];
  const budget = { n: 0 };
  for (const f of all) {
    if (kept.length >= MAX_FEATURES || budget.n >= MAX_COORDS) break;
    countCoords(f?.geometry?.coordinates, budget);
    kept.push(f);
  }
  return { data: { type: 'FeatureCollection', features: kept }, shown: kept.length, total: all.length };
}

export type Bbox = [number, number, number, number];

/** [west, south, east, north] over every coordinate, or null if there are none. */
export function bboxOf(fc: any): Bbox | null {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  const walk = (node: any) => {
    if (!Array.isArray(node)) return;
    if (typeof node[0] === 'number') {
      const [x, y] = node;
      if (x < w) w = x;
      if (x > e) e = x;
      if (y < s) s = y;
      if (y > n) n = y;
      return;
    }
    for (const child of node) walk(child);
  };
  for (const f of fc?.features || []) walk(f?.geometry?.coordinates);
  return w === Infinity ? null : [w, s, e, n];
}

/**
 * Slop for the degree check. Converting 4326 to WGS84 is nearly an identity transform, but proj4
 * still does the arithmetic, and a coordinate on the antimeridian comes back as
 * 180.00000000000011 - measured on the Eurostat countries file, an overshoot of 1.1e-13 degrees.
 * A millionth of a degree is about 0.1 m, five orders of magnitude away from projected units, so
 * this buys robustness and costs nothing in discrimination.
 */
const DEGREE_SLOP = 1e-6;

/**
 * The last line of defence, not the policy: anything with a declared CRS has already been
 * converted. This catches a payload that is projected AND declares nothing, where there is
 * no source to convert from.
 *
 * Tolerant by design - clamping the coordinates to satisfy it would be bending the data to fit
 * the check, and deck.gl draws 180.00000000000011 quite happily.
 */
export function looksWgs84(bbox: Bbox): boolean {
  const [w, s, e, n] = bbox;
  const lon = 180 + DEGREE_SLOP;
  const lat = 90 + DEGREE_SLOP;
  return Math.abs(w) <= lon && Math.abs(e) <= lon && Math.abs(s) <= lat && Math.abs(n) <= lat;
}

export function describeBbox(bbox: Bbox): string {
  const [w, s, e, n] = bbox;
  const fix = (v: number) => (Math.abs(v) > 1000 ? v.toFixed(0) : v.toFixed(3));
  return `lon ${fix(w)} to ${fix(e)}, lat ${fix(s)} to ${fix(n)}`;
}
