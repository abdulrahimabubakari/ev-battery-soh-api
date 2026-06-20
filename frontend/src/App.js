import { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";
import axios from "axios";

const API = "https://battery-soh-predictor.onrender.com";

const DEFAULTS = {
  discharge_duration: 5500,
  voltage_mean: 3.47,
  voltage_min: 2.85,
  voltage_drop: 1.35,
  voltage_std: 0.18,
  current_mean: 0.99,
  temp_mean: 6.5,
  temp_max: 9.2,
  temp_rise: 2.1,
  temp_std: 0.8,
  energy_discharged: 18500,
  voltage_slope: -0.00012,
  time_below_3_5v: 0.42,
  time_above_35c: 0.0,
  cycle_number: 50,
};

function CSVUpload({ apiStatus }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [csvResult, setCsvResult] = useState(null);
  const [csvError, setCsvError] = useState(null);

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith('.csv')) {
      setCsvError('Please upload a .csv file');
      return;
    }
    setUploading(true);
    setCsvError(null);
    setCsvResult(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post(`${API}/predict-from-csv`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setCsvResult(res.data);
    } catch (e) {
      setCsvError('Upload failed — check the CSV format');
    }
    setUploading(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const status = csvResult ? getStatus(csvResult.soh_percent) : null;

  return (
    <div style={s.panel}>
      <div style={s.panelTitle}>CSV upload — raw cycle data</div>
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => document.getElementById('csv-input').click()}
        style={{
          border: `1.5px dashed ${dragging ? '#185FA5' : '#ddd'}`,
          borderRadius: 8,
          padding: '1.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragging ? '#E6F1FB' : 'transparent',
          transition: 'all 0.15s'
        }}
      >
        <input
          id="csv-input"
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />
        <div style={{ fontSize: 24, marginBottom: 8 }}>📂</div>
        <div style={{ fontSize: 13, color: '#888' }}>
          {uploading ? 'Processing...' : 'Drop a battery cycle CSV or click to browse'}
        </div>
        <div style={{ fontSize: 11, color: '#bbb', marginTop: 4 }}>
          Requires: Voltage_measured, Current_measured, Temperature_measured, Time
        </div>
      </div>

      {csvError && (
        <div style={{ ...s.errorBar, marginTop: 10, marginBottom: 0 }}>{csvError}</div>
      )}

      {csvResult && (
        <div style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 13, color: '#888' }}>{csvResult.filename}</span>
            <span style={{ fontSize: 11, color: '#bbb' }}>{csvResult.rows_processed} rows · {csvResult.inference_time_ms}ms</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ fontSize: 36, fontWeight: 500, color: getSoHColor(csvResult.soh_percent) }}>
              {csvResult.soh_percent.toFixed(1)}%
            </div>
            <div>
              <div style={{ fontSize: 13, color: '#888' }}>State of health</div>
              <span style={{ display: 'inline-block', padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 500, background: status.bg, color: status.color, marginTop: 3 }}>
                {status.label}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const FEATURE_IMPORTANCE = [
  { name: "energy discharged", pct: 57 },
  { name: "discharge duration", pct: 34 },
  { name: "cycle number", pct: 22 },
  { name: "voltage std", pct: 16 },
  { name: "voltage slope", pct: 14 },
  { name: "time below 3.5V", pct: 10 },
];

const INPUT_LABELS = {
  discharge_duration: "discharge duration (s)",
  voltage_mean: "voltage mean (V)",
  voltage_min: "voltage min (V)",
  voltage_drop: "voltage drop (V)",
  voltage_std: "voltage std",
  current_mean: "current mean (A)",
  temp_mean: "temp mean (°C)",
  temp_max: "temp max (°C)",
  temp_rise: "temp rise (°C)",
  temp_std: "temp std",
  energy_discharged: "energy discharged",
  voltage_slope: "voltage slope",
  time_below_3_5v: "time below 3.5V",
  time_above_35c: "time above 35°C",
  cycle_number: "cycle number",
};

function getSoHColor(soh) {
  if (soh >= 90) return "#1D9E75";
  if (soh >= 80) return "#BA7517";
  return "#E24B4A";
}

function getStatus(soh) {
  if (soh >= 90) return { label: "Healthy", bg: "#EAF3DE", color: "#3B6D11" };
  if (soh >= 80) return { label: "Degraded", bg: "#FAEEDA", color: "#854F0B" };
  return { label: "End of life", bg: "#FCEBEB", color: "#A32D2D" };
}

const s = {
  dash: { padding: "1.5rem", fontFamily: "system-ui, -apple-system, sans-serif" },
  header: { display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "1.5rem" },
  h1: { fontSize: 20, fontWeight: 500, color: "#111", margin: 0 },
  subtitle: { fontSize: 13, color: "#888", marginTop: 3 },
  liveBadge: { display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#1D9E75", background: "#EAF3DE", padding: "4px 10px", borderRadius: 20 },
  liveDot: { width: 6, height: 6, background: "#1D9E75", borderRadius: "50%" },
  offlineBadge: { display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#A32D2D", background: "#FCEBEB", padding: "4px 10px", borderRadius: 20 },
  metrics: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: "1.5rem" },
  metric: { background: "#f5f5f3", borderRadius: 8, padding: "12px 14px" },
  metricLabel: { fontSize: 11, color: "#888", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" },
  metricValue: { fontSize: 22, fontWeight: 500, color: "#111" },
  metricSub: { fontSize: 11, color: "#aaa", marginTop: 2 },
  mainGrid: { display: "grid", gridTemplateColumns: "1fr 300px", gap: 16 },
  panel: { background: "#fff", border: "0.5px solid #e5e5e5", borderRadius: 12, padding: "1.25rem" },
  panelTitle: { fontSize: 11, fontWeight: 500, color: "#888", marginBottom: "1rem", textTransform: "uppercase", letterSpacing: "0.06em" },
  tabRow: { display: "flex", gap: 4, marginBottom: "1rem" },
  tab: { padding: "5px 12px", fontSize: 12, borderRadius: 8, border: "0.5px solid transparent", cursor: "pointer", color: "#888", background: "transparent" },
  tabActive: { padding: "5px 12px", fontSize: 12, borderRadius: 8, border: "0.5px solid #e5e5e5", cursor: "pointer", color: "#111", background: "#f5f5f3" },
  inputGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 },
  label: { display: "block", fontSize: 11, color: "#888", marginBottom: 3 },
  input: { width: "100%", padding: "6px 8px", fontSize: 13, border: "0.5px solid #ddd", borderRadius: 8, background: "#fff", color: "#111", boxSizing: "border-box" },
  btnRow: { display: "flex", gap: 8, marginTop: 12 },
  btnPrimary: { flex: 1, padding: 9, fontSize: 13, fontWeight: 500, borderRadius: 8, border: "none", background: "#185FA5", color: "#fff", cursor: "pointer" },
  btnSecondary: { flex: 1, padding: 9, fontSize: 13, fontWeight: 500, borderRadius: 8, border: "0.5px solid #ddd", background: "transparent", color: "#111", cursor: "pointer" },
  sohDisplay: { textAlign: "center", padding: "1.5rem 0" },
  sohLabel: { fontSize: 13, color: "#888", marginTop: 6 },
  divider: { border: "none", borderTop: "0.5px solid #e5e5e5", margin: "1rem 0" },
  inference: { fontSize: 12, color: "#aaa", textAlign: "center" },
  thresholdRow: { display: "flex", alignItems: "center", gap: 8, marginBottom: 8 },
  thresholdDot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  thresholdName: { fontSize: 13, color: "#111", flex: 1 },
  thresholdRange: { fontSize: 12, color: "#888" },
  featureBar: { display: "flex", alignItems: "center", gap: 8, marginBottom: 7 },
  featureName: { fontSize: 12, color: "#888", width: 110, flexShrink: 0 },
  featureTrack: { flex: 1, height: 4, background: "#f0f0f0", borderRadius: 2 },
  featurePct: { fontSize: 11, color: "#aaa", width: 32, textAlign: "right" },
  errorBar: { background: "#FCEBEB", color: "#A32D2D", padding: "8px 14px", borderRadius: 8, fontSize: 13, marginBottom: "1rem" },
};

export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [simLoading, setSimLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("curve");
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    axios.get(`${API}/health`).then(() => {
      setApiStatus("live");
      return axios.get(`${API}/model/info`);
    }).then(r => setModelInfo(r.data))
      .catch(() => setApiStatus("offline"));
  }, []);

  const predict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API}/predict`, {
        ...form,
        cycle_number: parseInt(form.cycle_number),
      });
      setResult(res.data);
      setHistory(prev => [...prev, {
        cycle: form.cycle_number,
        soh: res.data.soh_percent,
      }].sort((a, b) => a.cycle - b.cycle));
    } catch {
      setError("Prediction failed — the API may be waking up, try again in 30s");
    }
    setLoading(false);
  }, [form]);

  const simulate = useCallback(async () => {
    setSimLoading(true);
    setError(null);
    const cycles = Array.from({ length: 20 }, (_, i) => {
      const c = (i + 1) * 10;
      return {
        ...form,
        cycle_number: c,
        energy_discharged: Math.max(5000, 18500 - c * 60),
        discharge_duration: Math.max(2000, 5500 - c * 15),
        voltage_mean: Math.max(3.2, 3.47 - c * 0.001),
      };
    });
    try {
      const res = await axios.post(`${API}/batch-predict`, { cycles });
      const newHistory = res.data.predictions.map((p, i) => ({
        cycle: cycles[i].cycle_number,
        soh: p.soh_percent,
      }));
      setHistory(newHistory);
      setResult(res.data.predictions[res.data.predictions.length - 1]);
      setActiveTab("curve");
    } catch {
      setError("Simulation failed — try again");
    }
    setSimLoading(false);
  }, [form]);

  const status = result ? getStatus(result.soh_percent) : null;

  return (
    <div style={s.dash}>
      <div style={s.header}>
        <div>
          <h1 style={s.h1}>Battery intelligence</h1>
          <p style={s.subtitle}>State-of-health prediction · LightGBM · NASA dataset</p>
        </div>
        {apiStatus === "live"
          ? <div style={s.liveBadge}><div style={s.liveDot}></div>API live</div>
          : apiStatus === "offline"
            ? <div style={s.offlineBadge}>API offline</div>
            : <div style={{ fontSize: 12, color: "#aaa" }}>Connecting...</div>}
      </div>

      {error && <div style={s.errorBar}>{error}</div>}

      <div style={s.metrics}>
        {[
          { label: "Model RMSE", value: modelInfo ? `${modelInfo.val_rmse_percent}%` : "4.8%", sub: "SoH error" },
          { label: "R² score", value: modelInfo ? modelInfo.val_r2 : "0.89", sub: "variance explained" },
          { label: "P95 latency", value: "11ms", sub: "at 50 users" },
          { label: "Throughput", value: "151/s", sub: "requests per sec" },
        ].map(m => (
          <div key={m.label} style={s.metric}>
            <div style={s.metricLabel}>{m.label}</div>
            <div style={s.metricValue}>{m.value}</div>
            <div style={s.metricSub}>{m.sub}</div>
          </div>
        ))}
      </div>

      <div style={s.mainGrid}>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

          <div style={s.panel}>
            <div style={s.tabRow}>
              {["curve", "importance"].map(t => (
                <button key={t} style={activeTab === t ? s.tabActive : s.tab} onClick={() => setActiveTab(t)}>
                  {t === "curve" ? "Degradation curve" : "Feature importance"}
                </button>
              ))}
            </div>

            {activeTab === "curve" && (
              history.length > 0 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={history} margin={{ top: 4, right: 16, bottom: 16, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="cycle" tick={{ fontSize: 11, fill: "#aaa" }} label={{ value: "Cycle", position: "insideBottom", offset: -8, fontSize: 11, fill: "#aaa" }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#aaa" }} />
                    <Tooltip formatter={v => [`${v.toFixed(1)}%`, "SoH"]} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                    <ReferenceLine y={80} stroke="#E24B4A" strokeDasharray="4 3" strokeWidth={0.8} label={{ value: "EOL 80%", fontSize: 10, fill: "#E24B4A", position: "insideTopLeft" }} />
                    <ReferenceLine y={90} stroke="#BA7517" strokeDasharray="4 3" strokeWidth={0.8} />
                    <Line type="monotone" dataKey="soh" stroke="#185FA5" strokeWidth={2} dot={{ r: 3, fill: "#185FA5" }} activeDot={{ r: 5 }} name="SoH" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: "#ccc", fontSize: 13 }}>
                  Run a prediction or simulate degradation to see the curve
                </div>
              )
            )}

            {activeTab === "importance" && (
              <div style={{ paddingTop: 8 }}>
                {FEATURE_IMPORTANCE.map(f => (
                  <div key={f.name} style={s.featureBar}>
                    <span style={s.featureName}>{f.name}</span>
                    <div style={s.featureTrack}>
                      <div style={{ ...s.featureTrack, width: `${f.pct}%`, background: "#185FA5", height: 4 }} />
                    </div>
                    <span style={s.featurePct}>{f.pct}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={s.panel}>
            <div style={s.panelTitle}>Cycle parameters</div>
            <div style={s.inputGrid}>
              {Object.entries(form).map(([key, val]) => (
                <div key={key}>
                  <label style={s.label}>{INPUT_LABELS[key]}</label>
                  <input
                    type="number"
                    value={val}
                    step="any"
                    style={s.input}
                    onChange={e => setForm(f => ({ ...f, [key]: parseFloat(e.target.value) || 0 }))}
                  />
                </div>
              ))}
            </div>
            <div style={s.btnRow}>
              <button style={s.btnPrimary} onClick={predict} disabled={loading}>
                {loading ? "Predicting..." : "Predict SoH"}
              </button>
              <button style={s.btnSecondary} onClick={simulate} disabled={simLoading}>
                {simLoading ? "Simulating..." : "Simulate degradation"}
              </button>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={s.panel}>
            <div style={s.panelTitle}>Prediction</div>
            {result ? (
              <>
                <div style={s.sohDisplay}>
                  <div style={{ fontSize: 64, fontWeight: 500, lineHeight: 1, color: getSoHColor(result.soh_percent) }}>
                    {result.soh_percent.toFixed(1)}%
                  </div>
                  <div style={s.sohLabel}>State of health</div>
                  <div style={{ marginTop: 10 }}>
                    <span style={{ display: "inline-block", padding: "5px 14px", borderRadius: 20, fontSize: 12, fontWeight: 500, background: status.bg, color: status.color }}>
                      {status.label}
                    </span>
                  </div>
                </div>
                <hr style={s.divider} />
                <div style={s.inference}>Inference · {result.inference_time_ms}ms</div>
                <hr style={s.divider} />
              </>
            ) : (
              <div style={{ textAlign: "center", color: "#ccc", fontSize: 13, padding: "2rem 0" }}>
                Run a prediction to see results
              </div>
            )}
            <div>
              {[
                { dot: "#1D9E75", name: "Healthy", range: "> 90%" },
                { dot: "#BA7517", name: "Degraded", range: "80 – 90%" },
                { dot: "#E24B4A", name: "End of life", range: "< 80%" },
              ].map(t => (
                <div key={t.name} style={s.thresholdRow}>
                  <div style={{ ...s.thresholdDot, background: t.dot }} />
                  <span style={s.thresholdName}>{t.name}</span>
                  <span style={s.thresholdRange}>{t.range}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={s.panel}>
            <div style={s.panelTitle}>API endpoints</div>
            {[
              { method: "GET", path: "/health" },
              { method: "GET", path: "/model/info" },
              { method: "POST", path: "/predict" },
              { method: "POST", path: "/batch-predict" },
            ].map(e => (
              <div key={e.path} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 500, color: e.method === "GET" ? "#1D9E75" : "#185FA5", background: e.method === "GET" ? "#EAF3DE" : "#E6F1FB", padding: "2px 6px", borderRadius: 4, width: 36, textAlign: "center" }}>{e.method}</span>
                <span style={{ fontSize: 12, color: "#555", fontFamily: "monospace" }}>{e.path}</span>
              </div>
            ))}
          </div>
        </div>
      <div style={{ marginTop: 16 }}>
        <CSVUpload apiStatus={apiStatus} />
      </div>
      </div>
    </div>
  );
}