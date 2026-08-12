import { useEffect, useState } from "react";
import { getPortfolioStats } from "../api";

export default function PortfolioDashboard({ refreshKey }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getPortfolioStats().then(setStats).catch((e) => setError(e.message));
  }, [refreshKey]);

  if (error) return <p className="error-text">{error}</p>;
  if (!stats) return <p className="empty-state">Loading…</p>;

  if (stats.total_scored === 0) {
    return (
      <div className="panel">
        <p className="empty-state">No applicants scored yet. Score one from the "Score applicant" tab.</p>
      </div>
    );
  }

  return (
    <>
      <div className="stat-grid">
        <div className="stat-box">
          <div className="stat-label">Total scored</div>
          <div className="stat-value">{stats.total_scored}</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Predicted default rate</div>
          <div className="stat-value">{(stats.overall_default_rate * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-box">
          <div className="stat-label">High risk</div>
          <div className="stat-value" style={{ color: "var(--high)" }}>
            {stats.by_risk_band.high ?? 0}
          </div>
        </div>
        <div className="stat-box">
          <div className="stat-label">Low risk</div>
          <div className="stat-value" style={{ color: "var(--low)" }}>
            {stats.by_risk_band.low ?? 0}
          </div>
        </div>
      </div>

      <div className="panel">
        <p className="panel-title">Recent predictions</p>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Score</th>
              <th>Band</th>
              <th>Predicted</th>
              <th>Scored at</th>
            </tr>
          </thead>
          <tbody>
            {stats.recent.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{(r.risk_score * 100).toFixed(1)}%</td>
                <td style={{ color: `var(--${r.risk_band})` }}>{r.risk_band}</td>
                <td>{r.predicted_default ? "default" : "no default"}</td>
                <td>{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
