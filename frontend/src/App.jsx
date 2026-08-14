import { useState } from "react";
import ScoreForm from "./components/ScoreForm";
import ExplanationPanel from "./components/ExplanationPanel";
import PortfolioDashboard from "./components/PortfolioDashboard";
import { scoreApplicant, explainPrediction } from "./api";

export default function App() {
  const [tab, setTab] = useState("score");
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleSubmit = async (form) => {
    setLoading(true);
    setError(null);
    setExplanation(null);
    try {
      const pred = await scoreApplicant(form);
      setPrediction(pred);
      setRefreshKey((k) => k + 1);
      const exp = await explainPrediction(pred.id);
      setExplanation(exp);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="sidebar">
        <div className="brand">
          <strong>Credit Risk Desk</strong>
          XGBoost + SHAP
        </div>
        <div
          className={`nav-item ${tab === "score" ? "active" : ""}`}
          onClick={() => setTab("score")}
        >
          Score applicant
        </div>
        <div
          className={`nav-item ${tab === "portfolio" ? "active" : ""}`}
          onClick={() => setTab("portfolio")}
        >
          Portfolio
        </div>
      </div>

      <div className="main">
        {tab === "score" && (
          <>
            <h1>Score an applicant</h1>
            <p className="subtitle">
              Based on Yeh &amp; Lien (2009): Trained on 30,000 Taiwanese credit card accounts
            </p>
            <div className="panel">
              <p className="panel-title">Applicant details</p>
              <ScoreForm onSubmit={handleSubmit} loading={loading} />
              {error && <p className="error-text">{error}</p>}
            </div>
            {prediction && <ExplanationPanel prediction={prediction} explanation={explanation} />}
          </>
        )}

        {tab === "portfolio" && (
          <>
            <h1>Portfolio overview</h1>
            <p className="subtitle">All applicants scored so far, from the audit log</p>
            <PortfolioDashboard refreshKey={refreshKey} />
          </>
        )}
      </div>
    </div>
  );
}
