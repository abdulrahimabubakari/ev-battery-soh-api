import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer
} from "recharts";
import axios from "axios";

const API = "http://127.0.0.1:8000";

function StatusBadge({ status }) {
  const colors = {
    HEALTHY: { bg: "#d1fae5", text: "#065f46", label: "Healthy" },
    DEGRADED: { bg: "#fef3c7", text: "#92400e", label: "Degraded" },
    END_OF_LIFE: { bg: "#fee2e2", text: "#991b1b", label: "End of Life" },
  };
  const c = colors[status] || colors.HEALTHY;
  return (
    <span style={{
      background: c.bg, color: c.text,
      padding: "4px 12px", borderRadius: 20,
      fontWeight: 700, fontSize: 13
    }}>{c.label}</span>
  );
}

function SoHGauge({ value }) {
  const color = value >= 90 ? "#10b981" : value >= 80 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ textAlign: "center", padding: 20 }}>
      <div style={{ fontSize: 64, fontWeight: 800, color }}>{value}%</div>
      <div style={{ fontSize: 14, color: "#6b7280" }}>State of Health</div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState({
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
  });

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get(`${API}/model/info`)
      .then(r => setModelInfo(r.data))
      .catch(() => setError("API offline — start uvicorn first"));
  }, []);

  const predict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API}/predict`, {
        ...form,
        cycle_number: parseInt(form.cycle_number),
      });
      const pred = res.data;
      setResult(pred);
      setHistory(prev => [...prev, {
        cycle: form.cycle_number,
        soh: pred.soh_percent,
        status: pred.status
      }]);
    } catch (e) {
      setError("Prediction failed — check API is running");
    }
    setLoading(false);
  };

  const simulateDegradation = async () => {
    setLoading(true);
    setError(null);
    const cycles = [];
    for (let c = 1; c <= 200; c += 10) {
      cycles.push({
        ...form,
        cycle_number: c,
        energy_discharged: Math.max(5000, 18500 - c * 60),
        discharge_duration: Math.max(2000, 5500 - c * 15),
        voltage_mean: Math.max(3.2, 3.47 - c * 0.001),
      });
    }
    try {
      const res = await axios.post(`${API}/batch-predict`, { cycles });
      const newHistory = res.data.predictions.map((p, i) => ({
        cycle: cycles[i].cycle_number,
        soh: p.soh_percent,
        status: p.status
      }));
      setHistory(newHistory);
      setResult(res.data.predictions[res.data.predictions.length - 1]);
    } catch (e) {
      setError("Simulation failed");
    }
    setLoading(false);
  };

  return (
    <div style={{ fontFamily: "Inter, sans-serif", background: "#f9fafb", minHeight: "100vh", padding: 24 }}>
      <div style={{ maxWidth: 1100, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: "#111827", margin: 0 }}>
            🔋 EV Battery SoH Prediction
          </h1>
          <p style={{ color: "#6b7280", marginTop: 4 }}>
            Physics-informed ML · LightGBM · NASA Battery Dataset
          </p>
          {modelInfo && (
            <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 4 }}>
              Model RMSE: {modelInfo.val_rmse_percent}% · R²: {modelInfo.val_r2} · Features: {modelInfo.n_features}
            </div>
          )}
          {error && (
            <div style={{ background: "#fee2e2", color: "#991b1b", padding: "8px 16px", borderRadius: 8, marginTop: 8 }}>
              {error}
            </div>
          )}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

          {/* Input Form */}
          <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginTop: 0 }}>Cycle Parameters</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {Object.entries(form).map(([key, val]) => (
                <div key={key}>
                  <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 2 }}>
                    {key.replace(/_/g, " ")}
                  </label>
                  <input
                    type="number"
                    value={val}
                    step="any"
                    onChange={e => setForm(f => ({ ...f, [key]: parseFloat(e.target.value) || 0 }))}
                    style={{
                      width: "100%", padding: "6px 8px", borderRadius: 6,
                      border: "1px solid #e5e7eb", fontSize: 13, boxSizing: "border-box"
                    }}
                  />
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
              <button onClick={predict} disabled={loading} style={{
                flex: 1, background: "#2563eb", color: "#fff", border: "none",
                borderRadius: 8, padding: "10px 0", fontWeight: 700, cursor: "pointer", fontSize: 14
              }}>
                {loading ? "Predicting..." : "Predict SoH"}
              </button>
              <button onClick={simulateDegradation} disabled={loading} style={{
                flex: 1, background: "#7c3aed", color: "#fff", border: "none",
                borderRadius: 8, padding: "10px 0", fontWeight: 700, cursor: "pointer", fontSize: 14
              }}>
                Simulate Degradation
              </button>
            </div>
          </div>

          {/* Result Panel */}
          <div style={{ background: "#fff", borderRadius: 12, padding: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginTop: 0 }}>Prediction Result</h2>
            {result ? (
              <>
                <SoHGauge value={result.soh_percent} />
                <div style={{ textAlign: "center", marginBottom: 12 }}>
                  <StatusBadge status={result.status} />
                </div>
                <div style={{ fontSize: 12, color: "#9ca3af", textAlign: "center" }}>
                  Inference: {result.inference_time_ms}ms
                </div>
                <div style={{ marginTop: 16, padding: 12, background: "#f3f4f6", borderRadius: 8, fontSize: 13 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>🟢 Healthy</span><span>&gt; 90% SoH</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span>🟡 Degraded</span><span>80–90% SoH</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                    <span>🔴 End of Life</span><span>&lt; 80% SoH</span>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ textAlign: "center", color: "#9ca3af", marginTop: 60 }}>
                Enter cycle parameters and click Predict
              </div>
            )}
          </div>
        </div>

        {/* Degradation Chart */}
        {history.length > 0 && (
          <div style={{ background: "#fff", borderRadius: 12, padding: 24, marginTop: 24, boxShadow: "0 1px 3px rgba(0,0,0,0.1)" }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, marginTop: 0 }}>SoH Degradation Curve</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                <XAxis dataKey="cycle" label={{ value: "Cycle Number", position: "insideBottom", offset: -4 }} />
                <YAxis domain={[0, 100]} label={{ value: "SoH (%)", angle: -90, position: "insideLeft" }} />
                <Tooltip formatter={(v) => [`${v}%`, "SoH"]} />
                <Legend />
                <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="4 4" label="EOL (80%)" />
                <ReferenceLine y={90} stroke="#f59e0b" strokeDasharray="4 4" label="Degraded (90%)" />
                <Line type="monotone" dataKey="soh" stroke="#2563eb" strokeWidth={2} dot={{ r: 4 }} name="State of Health" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}