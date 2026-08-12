const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const scoreApplicant = (data) =>
  request("/predict", { method: "POST", body: JSON.stringify(data) });

export const explainPrediction = (id) => request(`/explain/${id}`);

export const getPortfolioStats = () => request("/portfolio/stats");
