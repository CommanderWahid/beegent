import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/** What the harness measured off the wire. Never merged with `claim`. */
export interface Verification {
  ok?: boolean; status?: number; payload_type?: string;
  total_size_bytes?: number | null; first_bytes_hex?: string;
  access?: string; shape?: string; feature_count?: number; count_is_exact?: boolean;
}

export interface Link {
  link_uid: string; country: string; dataset: string; url: string; resource_url: string;
  confidence: number | null; last_verified: string; status: string;
  /** Model-written and UNVERIFIED. Display it, never present it as measured. */
  claim: Record<string, any>;
  verification: Verification;
}

export interface CountryCount { country: string; links: number; ok: number; }

/** What would answer a request. Names only - never key material. */
export interface Resolved { backend: string; models: Record<string, string>; }

/** What the chat pane renders. `links` are store rows, never prose about them. */
export interface Message {
  who: 'you' | 'agent';
  html: string;
  /** The unrendered text, which is what the model gets back as history. */
  text: string;
  links?: Link[];
  country?: string;
  confirm?: RunFields;
}

/** What a run is given. `sent_as` is the composed string the pipeline actually reads. */
export interface RunFields {
  country: string; use_case: string; format: string; vintage: string;
  channel?: string; sent_as?: string;
}

export interface RunView {
  id: string;
  country: string;
  use_case: string;
  running: boolean;
  /** The CURRENT activity in plain words - replaced, never accumulated. Raw lines go to Logs. */
  step: string;
  summary?: any;
}

export interface ChatReply extends RunFields {
  intent: 'search' | 'catalog' | 'other';
  missing: string[]; ready?: boolean; reply: string;
  links?: Link[];
}

@Injectable({ providedIn: 'root' })
export class Beegent {
  /** Mirrors the API's single-run lock, so the UI can disable Search while one is in flight. */
  readonly running = signal(false);

  constructor(private http: HttpClient) {}

  countries(): Observable<CountryCount[]> {
    return this.http.get<CountryCount[]>('/api/countries');
  }

  links(country = ''): Observable<Link[]> {
    return this.http.get<Link[]>('/api/links', { params: country ? { country } : {} });
  }

  config(): Observable<Resolved> {
    return this.http.get<Resolved>('/api/config');
  }

  greeting(): Observable<{ reply: string }> {
    return this.http.get<{ reply: string }>('/api/chat');
  }

  chat(message: string, history: any[] = []): Observable<ChatReply> {
    return this.http.post<ChatReply>('/api/chat', { message, history });
  }

  /** All four fields: the API folds format, vintage and channel into the use case. */
  start(fields: RunFields): Observable<{ run_id: string }> {
    return this.http.post<{ run_id: string }>('/api/runs', fields);
  }

  /** Cooperative cancel: the pipeline checks between angles and agent steps. */
  stop(runId: string): Observable<unknown> {
    return this.http.post(`/api/runs/${runId}/stop`, {});
  }

  /** The finished run: status, candidates and what it cost. 202 while it is still going. */
  result(runId: string): Observable<any> {
    return this.http.get<any>(`/api/runs/${runId}`);
  }

  /** SSE, not polling: a run is minutes long and the log IS the progress indicator. */
  logs(runId: string, onLine: (line: string) => void, onDone: () => void): EventSource {
    const es = new EventSource(`/api/runs/${runId}/logs`);
    es.onmessage = (e) => onLine(e.data);
    es.addEventListener('done', () => { es.close(); onDone(); });
    es.onerror = () => { es.close(); onDone(); };
    return es;
  }
}
