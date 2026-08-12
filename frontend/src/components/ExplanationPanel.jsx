function bandFromScore(score) {
  if (score < 0.2) return "low";
  if (score < 0.5) return "medium";
  return "high";
}

export default function ExplanationPanel({ prediction, explanation }) {
  if (!prediction) return null;

  const band = bandFromScore(prediction.risk_score);
  const maxAbs = explanation
    ? Math.max(...explanation.top_factors.map((f) => Math.abs(f.shap_value)), 0.0001)
    : 1;

  return (
    <div className="panel">
      <p className="panel-title">Result — applicant #{prediction.id}</p>
      <div className="score-display">
        <span className="score-number">{(prediction.risk_score * 100).toFixed(1)}%</span>
        <span className={`risk-badge ${band}`}>{band} risk</span>
      </div>
      <p className="subtitle" style={{ marginBottom: 20 }}>
        Estimated probability of default next month
      </p>

      {explanation ? (
        <>
          <p className="panel-title">Top factors (SHAP)</p>
          {explanation.top_factors.map((f) => {
            const pct = (Math.abs(f.shap_value) / maxAbs) * 50;
            const positive = f.shap_value >= 0;
            return (
              <div className="factor-row" key={f.feature}>
                <span className="factor-name">{f.feature}</span>
                <div className="factor-bar-track">
                  <div className="factor-bar-center" />
                  <div
                    className="factor-bar-fill"
                    style={{
                      background: positive ? "var(--high)" : "var(--low)",
                      left: positive ? "50%" : `${50 - pct}%`,
                      width: `${pct}%`,
                    }}
                  />
                </div>
                <span className="factor-value">{f.value}</span>
              </div>
            );
          })}
          <p className="subtitle" style={{ marginTop: 16, marginBottom: 0 }}>
            Red bars push risk up, teal bars push risk down. Value column shows the applicant's raw input.
          </p>
        </>
      ) : (
        <p className="empty-state">Loading explanation…</p>
      )}
    </div>
  );
}
