'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

/* Muted, desaturated span colours — observability tooling, not a demo. */
const KIND_COLORS = {
  retrieve: '#56c8dc', generate: '#7d84f0', verify: '#4cc38a',
  planning: '#d4a24e', tool: '#d38ec2', perceive: '#9d8cff',
  transcribe: '#e07856',
};
const BAND_COLORS = { return: '#4cc38a', verify: '#d9a94e', escalate: '#e5534b' };

async function api(path, options) {
  const res = await fetch(`/api${path}`, options);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* Span offsets derived from the dependency graph: a node starts
   when its slowest dependency ends. */
function waterfall(nodes) {
  const byId = Object.fromEntries(nodes.map((n) => [n.node_id, n]));
  const end = {};
  const start = (n) => {
    if (end[n.node_id] !== undefined) return end[n.node_id] - (n.latency_ms || 0);
    const deps = (n.depends_on || []).map((d) => byId[d]).filter(Boolean);
    const s = deps.length ? Math.max(...deps.map((d) => start(d) + (d.latency_ms || 0))) : 0;
    end[n.node_id] = s + (n.latency_ms || 0);
    return s;
  };
  const rows = nodes.map((n) => ({ ...n, start: start(n) }));
  return { rows, total: Math.max(1, ...rows.map((r) => r.start + (r.latency_ms || 0))) };
}

const score = (c) => c?.confidence_score ?? c?.score ?? 0;
const fmtMs = (v) => (v == null ? '—' : `${Math.round(v)} ms`);
const fmtCost = (v) => (v == null ? '—' : `$${Number(v).toFixed(6)}`);

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <rect x="0.5" y="0.5" width="25" height="25" rx="6.5" fill="#12151b" stroke="#2b313f" />
      <circle cx="7.5" cy="13" r="2.1" fill="#6a70e8" />
      <circle cx="18.5" cy="6.8" r="1.7" fill="#56c8dc" />
      <circle cx="18.5" cy="19.2" r="1.7" fill="#4cc38a" />
      <path d="M9.4 12.2 16.9 7.6M9.4 13.8 16.9 18.4" stroke="#3a4152" strokeWidth="1.3" />
    </svg>
  );
}

export default function Home() {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [result, setResult] = useState(null);
  const [traces, setTraces] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [span, setSpan] = useState(null);
  const fileRef = useRef(null);
  const inputRef = useRef(null);

  const refresh = useCallback(async () => {
    // Each feed settles independently: with Promise.all, one failing
    // endpoint rejected the whole batch and blanked every panel.
    const [h, t, m] = await Promise.allSettled([
      api('/health'), api('/traces'), api('/metrics/summary')]);
    if (h.status === 'fulfilled') setHealth(h.value);
    if (t.status === 'fulfilled') setTraces(t.value.traces || t.value);
    if (m.status === 'fulfilled') setMetrics(m.value);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  // Keep the console live: metrics/traces also reflect queries run
  // elsewhere (dashboard, curl, MCP), not just this tab.
  useEffect(() => {
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);
  useEffect(() => { inputRef.current?.focus(); }, []);

  async function upload(ev) {
    const file = ev.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    try {
      const out = await api('/documents', { method: 'POST', body: form });
      setAttachments((a) => [...new Set([...a, out.source])]);
      refresh();
    } catch (e) { setError(`Upload failed: ${e.message}`); }
    ev.target.value = '';
  }

  async function run() {
    if (!query.trim() || busy) return;
    setBusy(true); setError(''); setSpan(null);
    try {
      const out = await api('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, attachments }),
      });
      setResult(out);
      refresh();
    } catch (e) { setError(`Query failed: ${e.message}`); }
    setBusy(false);
  }

  const wf = result ? waterfall(result.execution_graph || []) : null;
  const conf = result ? score(result.confidence) : 0;
  const band = result?.confidence?.decision || 'return';
  const sysRows = health ? [
    ['Mode', health.model_mode], ['Router', health.decision_model],
    ['Planner', health.plan_model], ['Embeddings', health.embeddings],
    ['Vector store', health.vector_store], ['Vision', health.vision_model],
  ] : [];
  const stratDist = metrics?.strategies
    ? Object.entries(metrics.strategies).sort((a, b) => b[1] - a[1]) : [];
  const stratMax = Math.max(1, ...stratDist.map(([, n]) => n));

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <Logo />
          <div className="name">NeuroRoute <span>console</span></div>
        </div>
        <nav className="nav">
          <a href="#console" className="active">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 17l6-5-6-5M12 19h8" /></svg>
            Console</a>
          <a href="#traces">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h4l3-8 4 16 3-8h4" /></svg>
            Traces</a>
          <a href="#metrics">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 20V10M10 20V4M16 20v-8M22 20H2" /></svg>
            Metrics</a>
        </nav>
        <div className="sys">
          <div className="side-label">System</div>
          {health == null && <div className="sys-row"><span className="dot off" /><span className="k">API</span><span className="v">unreachable</span></div>}
          {sysRows.map(([k, v]) => (
            <div className="sys-row" key={k}>
              <span className="dot" /><span className="k">{k}</span><span className="v">{v}</span>
            </div>
          ))}
        </div>
        <div className="side-foot">
          <a href="https://github.com/Anurag0018/neuroroute" target="_blank" rel="noreferrer">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.55v-2.15c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.69 1.25 3.35.96.1-.75.4-1.25.72-1.54-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.24 2.76.12 3.05.74.81 1.18 1.83 1.18 3.09 0 4.41-2.68 5.38-5.24 1.28 1.06.91 2.4 0 1.08-.01 2.2 0 .79.27.54 1.83-2.33-5.53 1.36-10.83 6.83-10.83A11.5 11.5 0 0 0 12 .5Z" transform="scale(0.92) translate(1 1)" /></svg>
            Anurag0018/neuroroute</a>
        </div>
      </aside>

      <div className="main">
        <div className="wrap">
          <div className="topbar">
            <h1><span className="crumb">NeuroRoute /</span> Console</h1>
            <span className="env"><span className={`dot${health ? '' : ' off'}`} />
              {health ? `${health.model_mode} · ${health.decision_model} router` : 'api offline'}</span>
          </div>

          {/* ---------- composer ---------- */}
          <div className="panel" id="console" style={{ padding: 0 }}>
            <div className="composer">
              <span className="glyph">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 17l6-5-6-5M12 19h8" /></svg>
              </span>
              <input ref={inputRef} type="text"
                placeholder="Ask anything — arithmetic, dates, a document, an image…"
                value={query} onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') run(); }} />
              <button className="btn ghost" onClick={() => fileRef.current?.click()}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.4 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" /></svg>
                Attach</button>
              <input ref={fileRef} type="file" hidden
                accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp,.gif,.mp3,.wav,.m4a,.ogg,.flac,.webm"
                onChange={upload} />
              <button className="btn" onClick={run} disabled={busy}>
                {busy ? <><span className="spinner" />Running</> : <>Run<kbd>↵</kbd></>}
              </button>
            </div>
            {attachments.length > 0 && (
              <div className="attach-row">
                {attachments.map((a) => (
                  <span key={a} className="attach"><b>{a}</b>
                    <button aria-label={`Remove ${a}`}
                      onClick={() => setAttachments((x) => x.filter((y) => y !== a))}>×</button>
                  </span>
                ))}
              </div>
            )}
            {error && <p className="err">{error}</p>}
          </div>

          {/* ---------- result ---------- */}
          {busy && !result && (
            <div className="panel"><div className="skel">
              <div style={{ width: '38%' }} /><div style={{ width: '82%' }} /><div style={{ width: '64%' }} />
            </div></div>
          )}
          {result && (
            <div className="panel">
              <div className="panel-h">
                <span className="panel-t">Response</span>
                <div className="meta-line">
                  <span className="tag">{result.strategy}</span>
                  {result.decision?.source && (
                    <span className="decision-note">routed by {result.decision.source}
                      {result.decision.reason ? ` — ${result.decision.reason}` : ''}</span>)}
                </div>
              </div>
              <div className="panel-b">
                <p className="answer">{result.answer}</p>
                {(result.escalation_history || []).length > 0 && (
                  <div className="esc">
                    Escalated {result.escalation_history.map((h, i) => (
                      <span key={i}><b>{h.from}</b> → <b>{h.to}</b>{i < result.escalation_history.length - 1 ? ', ' : ''}</span>
                    ))} after the first attempt scored below the confidence threshold.
                  </div>
                )}
              </div>
              <div className="stats">
                <div className="stat"><div className="k">Latency</div>
                  <div className="v">{Math.round(result.metrics?.latency_ms || 0)}<small> ms</small></div></div>
                <div className="stat"><div className="k">Tokens</div>
                  <div className="v">{result.metrics?.tokens ?? '—'}</div></div>
                <div className="stat"><div className="k">Cost</div>
                  <div className="v">{fmtCost(result.metrics?.cost_usd)}</div></div>
                <div className="stat"><div className="k">Escalations</div>
                  <div className="v">{result.metrics?.escalations ?? 0}</div></div>
              </div>
              <div className="conf">
                <div className="conf-top">
                  <span className="k">Confidence</span>
                  <span className="v" style={{ color: BAND_COLORS[band] }}>
                    {Math.round(conf * 100)}% · {band}</span>
                </div>
                <div className="conf-track">
                  <div className="conf-fill" style={{ width: `${conf * 100}%`, background: BAND_COLORS[band] }} />
                </div>
                <div className="conf-grid">
                  <div className="cell"><div className="k">Evidence coverage</div>
                    <div className="v">{result.confidence?.evidence_coverage?.toFixed(2) ?? '—'}</div></div>
                  <div className="cell"><div className="k">Retrieval relevance</div>
                    <div className="v">{result.confidence?.retrieval_relevance?.toFixed(2) ?? '—'}</div></div>
                  <div className="cell"><div className="k">Validation</div>
                    <div className="v">{result.confidence?.validation_passed ? 'passed' : 'failed'}</div></div>
                  <div className="cell"><div className="k">Model agreement</div>
                    <div className="v">{result.confidence?.model_agreement?.toFixed(2) ?? '—'}</div></div>
                </div>
              </div>
              {wf && wf.rows.length > 0 && (
                <>
                  <div className="panel-h" style={{ borderTop: '1px solid var(--line)' }}>
                    <span className="panel-t">Execution — span waterfall</span>
                    <span className="decision-note">{wf.rows.length} spans · {fmtMs(wf.total)}</span>
                  </div>
                  <div className="wf">
                    <div className="wf-ruler">
                      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                        <span key={f} style={{ left: `${f * 100}%` }}>{Math.round(wf.total * f)}</span>
                      ))}
                    </div>
                    {wf.rows.map((n) => (
                      <div key={n.node_id}
                        className={`wf-row${span?.node_id === n.node_id ? ' sel' : ''}`}
                        onClick={() => setSpan(span?.node_id === n.node_id ? null : n)}>
                        <div className="wf-label">
                          <span className="id">{n.node_id}</span>
                          <span className="kind">{n.kind}{n.model_tier ? ` · ${n.model_tier}` : ''}</span>
                        </div>
                        <div className="wf-track">
                          <div className={`wf-bar${n.status === 'failed' ? ' failed' : ''}`} style={{
                            left: `${(n.start / wf.total) * 100}%`,
                            width: `${Math.max(0.5, ((n.latency_ms || 0) / wf.total) * 100)}%`,
                            background: KIND_COLORS[n.kind] || '#888',
                          }} />
                        </div>
                        <div className="wf-ms">{n.latency_ms} ms</div>
                      </div>
                    ))}
                  </div>
                  {span && (
                    <div className="span-detail">
                      <div className="cell"><div className="k">Span</div><div className="v">{span.node_id}</div></div>
                      <div className="cell"><div className="k">Kind</div><div className="v">{span.kind}</div></div>
                      <div className="cell"><div className="k">Tier</div><div className="v">{span.model_tier || '—'}</div></div>
                      <div className="cell"><div className="k">Status</div><div className="v">{span.status}</div></div>
                      <div className="cell"><div className="k">Duration</div><div className="v">{span.latency_ms} ms</div></div>
                      <div className="cell"><div className="k">Depends on</div>
                        <div className="v">{(span.depends_on || []).join(', ') || '—'}</div></div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ---------- metrics ---------- */}
          {metrics === null && (
            <div id="metrics" className="panel">
              <div className="empty">
                <div className="big">Metrics unavailable</div>
                The API did not answer — check the backend is running on port 8000.
              </div>
            </div>
          )}
          {metrics && metrics.requests === 0 && (
            <div id="metrics" className="panel">
              <div className="empty">
                <div className="big">No metrics yet</div>
                The backend collects metrics in memory, so they start at zero
                every time it restarts. Run a query and this strip populates.
              </div>
            </div>
          )}
          {metrics && metrics.requests > 0 && (
            <div id="metrics">
              <div className="metric-strip">
                <div className="stat"><div className="k">Requests</div><div className="v">{metrics.requests}</div></div>
                <div className="stat"><div className="k">Avg latency</div>
                  <div className="v">{Math.round(metrics.avg_latency_ms)}<small> ms</small></div></div>
                <div className="stat"><div className="k">Avg tokens</div><div className="v">{Math.round(metrics.avg_tokens)}</div></div>
                <div className="stat"><div className="k">Avg cost</div><div className="v">{fmtCost(metrics.avg_cost_usd)}</div></div>
                <div className="stat"><div className="k">Escalation rate</div>
                  <div className="v">{Math.round(metrics.escalation_rate * 100)}<small> %</small></div></div>
              </div>
              {stratDist.length > 0 && (
                <div className="panel" style={{ marginTop: -8 }}>
                  <div className="dist" style={{ borderTop: 0 }}>
                    {stratDist.map(([name, n]) => (
                      <div className="dist-row" key={name}>
                        <span className="name">{name}</span>
                        <span className="track"><span className="fill" style={{ width: `${(n / stratMax) * 100}%`, display: 'block' }} /></span>
                        <span className="n">{n}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ---------- traces ---------- */}
          <div className="panel" id="traces">
            <div className="panel-h">
              <span className="panel-t">Traces</span>
              <span className="decision-note">{traces.length} recorded this session</span>
            </div>
            {traces.length === 0 ? (
              <div className="empty">
                <div className="big">No traces yet</div>
                Run a query and its full execution trace will appear here.
              </div>
            ) : (
              <div className="panel-b flush">
                <table>
                  <thead><tr>
                    <th>Request</th><th>Query</th><th>Strategy</th><th>Confidence</th>
                    <th className="num">Latency</th><th className="num">Tokens</th><th className="num">Cost</th>
                  </tr></thead>
                  <tbody>
                    {traces.slice(0, 14).map((t) => {
                      const s = score(t.confidence);
                      const b = t.confidence?.decision || 'return';
                      return (
                        <tr key={t.request_id}
                          className={`clickable${result?.request_id === t.request_id ? ' sel' : ''}`}
                          onClick={() => { setResult(t); setSpan(null); window.scrollTo({ top: 0, behavior: 'smooth' }); }}>
                          <td><span className="rid">{(t.request_id || '').slice(0, 8)}</span></td>
                          <td className="q">{t.query}</td>
                          <td><span className="strat">{t.strategy}</span></td>
                          <td><span className="mini-conf">
                            <span className="mini-track"><span className="mini-fill" style={{ width: `${s * 100}%`, background: BAND_COLORS[b], display: 'block' }} /></span>
                            {Math.round(s * 100)}%</span></td>
                          <td className="num">{t.metrics?.latency_ms}</td>
                          <td className="num">{t.metrics?.tokens}</td>
                          <td className="num">{fmtCost(t.metrics?.cost_usd)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <p className="foot">NeuroRoute — adaptive inference orchestration. Every request routed, verified and priced.</p>
        </div>
      </div>
    </div>
  );
}
