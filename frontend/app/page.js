'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

const KIND_COLORS = {
  retrieve: '#2dd4bf', generate: '#5b6cff', verify: '#43d17c',
  planning: '#e0a458', tool: '#e879c6', perceive: '#c084fc',
};

async function api(path, options) {
  const res = await fetch(`/api${path}`, options);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// Span offsets derived from the dependency graph, mirroring the
// built-in dashboard's waterfall: a node starts when its slowest
// dependency ends.
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

export default function Home() {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [result, setResult] = useState(null);
  const [traces, setTraces] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [h, t, m] = await Promise.all([
        api('/health'), api('/traces'), api('/metrics/summary')]);
      setHealth(h); setTraces(t.traces || t); setMetrics(m);
    } catch { /* backend not up yet */ }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

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
    if (!query.trim()) return;
    setBusy(true); setError('');
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

  return (
    <main>
      <h1>Neuro<span>Route</span> Console</h1>
      <p className="sub">Adaptive LLM inference orchestration — every request routed,
        verified and priced. This Next.js frontend talks to the same FastAPI API as the
        built-in dashboard.</p>

      {health && (
        <div className="chips" style={{ marginBottom: 16 }}>
          <span className="chip">mode <b>{health.model_mode}</b></span>
          <span className="chip">router <b>{health.decision_model}</b></span>
          <span className="chip">planner <b>{health.plan_model}</b></span>
          <span className="chip">embeddings <b>{health.embeddings}</b></span>
          <span className="chip">vector store <b>{health.vector_store}</b></span>
          <span className="chip">vision <b>{health.vision_model}</b></span>
        </div>
      )}

      <div className="card">
        <div className="row">
          <input type="text" placeholder="Ask anything — try “What is 15% of 84,000?” or upload an image and ask about it"
            value={query} onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && run()} />
          <button onClick={run} disabled={busy}>{busy ? 'Running…' : 'Run'}</button>
          <button className="ghost" onClick={() => fileRef.current?.click()}>Upload</button>
          <input ref={fileRef} type="file" hidden
            accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp,.gif" onChange={upload} />
        </div>
        {attachments.length > 0 && (
          <div className="chips">
            {attachments.map((a) => <span key={a} className="chip">📎 <b>{a}</b></span>)}
            <span className="chip" style={{ cursor: 'pointer' }}
              onClick={() => setAttachments([])}>clear</span>
          </div>
        )}
        {error && <p className="err">{error}</p>}
      </div>

      {result && (
        <div className="card">
          <div className="chips" style={{ marginTop: 0 }}>
            <span className="chip">strategy <b>{result.strategy}</b></span>
            <span className="chip">confidence <b>{Math.round((result.confidence?.score || 0) * 100)}%</b>
              {' '}({result.confidence?.decision})</span>
            <span className="chip">latency <b>{result.metrics?.latency_ms}ms</b></span>
            <span className="chip">tokens <b>{result.metrics?.tokens}</b></span>
            <span className="chip">cost <b>${(result.metrics?.cost_usd || 0).toFixed(6)}</b></span>
            {result.decision?.source && (
              <span className="chip">decision <b>{result.decision.source}</b></span>)}
            {(result.escalation_history || []).map((h, i) => (
              <span key={i} className="chip">escalated <b>{h.from} → {h.to}</b></span>))}
          </div>
          <p className="answer">{result.answer}</p>
          {wf && (
            <div>
              <p className="small">Execution graph — span waterfall</p>
              {wf.rows.map((n) => (
                <div className="bar" key={n.node_id}>
                  <div style={{
                    left: `${(n.start / wf.total) * 100}%`,
                    width: `${Math.max(0.4, ((n.latency_ms || 0) / wf.total) * 100)}%`,
                    background: KIND_COLORS[n.kind] || '#888',
                  }} />
                  <span style={{ left: `calc(${(n.start / wf.total) * 100}% + ${Math.max(0.4, ((n.latency_ms || 0) / wf.total) * 100)}% + 8px)` }}>
                    {n.node_id} · {n.kind}{n.model_tier ? ` · ${n.model_tier}` : ''} · {n.latency_ms}ms
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {metrics && metrics.requests > 0 && (
        <div className="card">
          <div className="chips" style={{ marginTop: 0 }}>
            <span className="chip">requests <b>{metrics.requests}</b></span>
            <span className="chip">avg latency <b>{metrics.avg_latency_ms}ms</b></span>
            <span className="chip">avg tokens <b>{metrics.avg_tokens}</b></span>
            <span className="chip">avg cost <b>${metrics.avg_cost_usd}</b></span>
            <span className="chip">escalation rate <b>{Math.round(metrics.escalation_rate * 100)}%</b></span>
          </div>
        </div>
      )}

      {traces.length > 0 && (
        <div className="card">
          <p className="small" style={{ marginTop: 0 }}>Recent traces — click one to inspect it</p>
          <table>
            <thead><tr><th>Query</th><th>Strategy</th><th>Conf.</th>
              <th>Latency</th><th>Tokens</th><th>Cost</th></tr></thead>
            <tbody>
              {traces.slice(0, 12).map((t) => (
                <tr key={t.request_id} className="clickable" onClick={() => setResult(t)}>
                  <td>{(t.query || '').slice(0, 52)}</td>
                  <td>{t.strategy}</td>
                  <td>{Math.round((t.confidence?.score || 0) * 100)}%</td>
                  <td>{t.metrics?.latency_ms}ms</td>
                  <td>{t.metrics?.tokens}</td>
                  <td>${(t.metrics?.cost_usd || 0).toFixed(6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
