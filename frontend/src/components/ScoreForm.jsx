import { useState } from "react";

const DEFAULTS = {
  LIMIT_BAL: 120000, SEX: 2, EDUCATION: 2, MARRIAGE: 2, AGE: 26,
  PAY_1: -1, PAY_2: 2, PAY_3: 0, PAY_4: 0, PAY_5: 0, PAY_6: 2,
  BILL_AMT1: 2682, BILL_AMT2: 1725, BILL_AMT3: 2682, BILL_AMT4: 3272,
  BILL_AMT5: 3455, BILL_AMT6: 3261,
  PAY_AMT1: 0, PAY_AMT2: 1000, PAY_AMT3: 1000, PAY_AMT4: 1000,
  PAY_AMT5: 0, PAY_AMT6: 2000,
};

const PAY_STATUS_OPTIONS = [
  { value: -1, label: "-1 (paid duly)" },
  { value: 0, label: "0 (revolving)" },
  { value: 1, label: "1 (1mo delay)" },
  { value: 2, label: "2 (2mo delay)" },
  { value: 3, label: "3 (3mo delay)" },
  { value: 6, label: "6 (6mo delay)" },
];

export default function ScoreForm({ onSubmit, loading }) {
  const [form, setForm] = useState(DEFAULTS);

  const set = (key) => (e) => {
    const val = e.target.type === "number" || e.target.tagName === "SELECT"
      ? Number(e.target.value)
      : e.target.value;
    setForm((f) => ({ ...f, [key]: val }));
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(form);
      }}
    >
      <div className="form-grid">
        <div className="field">
          <label>Credit limit (NT$)</label>
          <input type="number" value={form.LIMIT_BAL} onChange={set("LIMIT_BAL")} />
        </div>
        <div className="field">
          <label>Age</label>
          <input type="number" value={form.AGE} onChange={set("AGE")} />
        </div>
        <div className="field">
          <label>Sex</label>
          <select value={form.SEX} onChange={set("SEX")}>
            <option value={1}>Male</option>
            <option value={2}>Female</option>
          </select>
        </div>
        <div className="field">
          <label>Education</label>
          <select value={form.EDUCATION} onChange={set("EDUCATION")}>
            <option value={1}>Graduate school</option>
            <option value={2}>University</option>
            <option value={3}>High school</option>
            <option value={4}>Other</option>
          </select>
        </div>
        <div className="field">
          <label>Marital status</label>
          <select value={form.MARRIAGE} onChange={set("MARRIAGE")}>
            <option value={1}>Married</option>
            <option value={2}>Single</option>
            <option value={3}>Other</option>
          </select>
        </div>
        <div />

        {[1, 2, 3, 4, 5, 6].map((n) => (
          <div className="field" key={`pay-${n}`}>
            <label>Repayment status, month -{n}</label>
            <select value={form[`PAY_${n}`]} onChange={set(`PAY_${n}`)}>
              {PAY_STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        ))}

        {[1, 2, 3, 4, 5, 6].map((n) => (
          <div className="field" key={`bill-${n}`}>
            <label>Bill amount, month -{n}</label>
            <input type="number" value={form[`BILL_AMT${n}`]} onChange={set(`BILL_AMT${n}`)} />
          </div>
        ))}

        {[1, 2, 3, 4, 5, 6].map((n) => (
          <div className="field" key={`pamt-${n}`}>
            <label>Prior payment, month -{n}</label>
            <input type="number" value={form[`PAY_AMT${n}`]} onChange={set(`PAY_AMT${n}`)} />
          </div>
        ))}
      </div>

      <button className="btn" type="submit" disabled={loading}>
        {loading ? "Scoring…" : "Score applicant"}
      </button>
    </form>
  );
}
