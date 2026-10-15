const form = document.getElementById("form");
const resultEl = document.getElementById("result");
const submitBtn = document.getElementById("submit");

const LABELS = {
  tenure: "Tenure", Contract: "Contract", MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges", InternetService: "Internet", PaymentMethod: "Payment method",
  OnlineSecurity: "Online security", TechSupport: "Tech support", OnlineBackup: "Online backup",
  DeviceProtection: "Device protection", StreamingTV: "Streaming TV", StreamingMovies: "Streaming movies",
  PaperlessBilling: "Paperless billing", SeniorCitizen: "Senior citizen", Partner: "Partner",
  Dependents: "Dependents", PhoneService: "Phone service", MultipleLines: "Multiple lines", gender: "Gender",
};

const PRESETS = {
  risky: {
    tenure: 2, Contract: "Month-to-month", gender: "Female", SeniorCitizen: "1", Partner: "No",
    Dependents: "No", PhoneService: "Yes", MultipleLines: "Yes", InternetService: "Fiber optic",
    OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
    StreamingTV: "Yes", StreamingMovies: "Yes", MonthlyCharges: 99.65, TotalCharges: 199.3,
    PaperlessBilling: "Yes", PaymentMethod: "Electronic check",
  },
  loyal: {
    tenure: 60, Contract: "Two year", gender: "Male", SeniorCitizen: "0", Partner: "Yes",
    Dependents: "Yes", PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL",
    OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes",
    StreamingTV: "No", StreamingMovies: "No", MonthlyCharges: 74.1, TotalCharges: 4446,
    PaperlessBilling: "No", PaymentMethod: "Bank transfer (automatic)",
  },
};

// Keep add-on services consistent with the internet / phone choice
function syncDependentFields() {
  const noInternet = form.InternetService.value === "No";
  form.querySelectorAll("select.inet").forEach((s) => {
    if (noInternet) s.value = "No internet service";
    else if (s.value === "No internet service") s.value = "No";
    s.disabled = noInternet;
  });
  const noPhone = form.PhoneService.value === "No";
  if (noPhone) form.MultipleLines.value = "No phone service";
  else if (form.MultipleLines.value === "No phone service") form.MultipleLines.value = "No";
  form.MultipleLines.disabled = noPhone;
}
form.InternetService.addEventListener("change", syncDependentFields);
form.PhoneService.addEventListener("change", syncDependentFields);

document.querySelectorAll("[data-preset]").forEach((btn) =>
  btn.addEventListener("click", () => {
    const p = PRESETS[btn.dataset.preset];
    Object.entries(p).forEach(([k, v]) => (form[k].value = v));
    syncDependentFields();
    form.requestSubmit();
  })
);

function payload() {
  const data = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    data[el.name] = el.type === "number" ? Number(el.value) : el.value;
  }
  data.SeniorCitizen = Number(data.SeniorCitizen);
  return data;
}

const fmtPct = (x) => `${Math.round(x * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function render(r) {
  const color = `var(--${r.risk_level === "Low" ? "low" : r.risk_level === "Medium" ? "med" : "high"})`;
  const maxImpact = Math.max(...r.top_drivers.map((d) => d.impact), 0.01);
  const drivers = r.top_drivers.length
    ? `<ul class="drivers">${r.top_drivers
        .map(
          (d) => `<li><span>${esc(LABELS[d.feature] || d.feature)}
            <span class="val">= ${esc(d.value)}</span></span>
            <span class="imp">+${(d.impact * 100).toFixed(1)} pts</span>
            <div class="bar"><span style="width:${(d.impact / maxImpact) * 100}%"></span></div></li>`
        )
        .join("")}</ul>`
    : `<p class="none">Nothing in this profile raises the risk above a typical retained customer.</p>`;

  resultEl.classList.remove("empty");
  resultEl.innerHTML = `
    <div class="score"><span class="pct">${fmtPct(r.churn_probability)}</span>
      <span class="badge ${r.risk_level}">${r.risk_level} risk</span></div>
    <div class="meter"><span style="width:${r.churn_probability * 100}%;background:${color}"></span>
      <i class="tick" style="left:${r.threshold * 100}%"></i></div>
    <div class="meter-legend"><span>0%</span><span>decision threshold ${fmtPct(r.threshold)}</span><span>100%</span></div>
    <p class="verdict">${
      r.will_churn
        ? "<strong>Likely to churn.</strong> Worth a retention offer."
        : "<strong>Likely to stay.</strong> No action needed."
    }</p>
    <h3>What's raising the risk</h3>${drivers}`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";
  try {
    const body = JSON.stringify(payload());
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const msg = Array.isArray(err.detail) ? err.detail.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ") : res.statusText;
      throw new Error(msg);
    }
    render(await res.json());
  } catch (err) {
    resultEl.classList.remove("empty");
    resultEl.innerHTML = `<p class="error">Couldn't get a prediction: ${esc(err.message)}</p>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict churn risk";
  }
});

(async () => {
  try {
    const m = await (await fetch("/model-info")).json();
    const t = m.test_metrics;
    const rows = [
      ["Algorithm", m.model_name.replace(/_/g, " ")],
      ["Training rows", m.n_rows.toLocaleString()],
      ["Base churn rate", fmtPct(m.churn_rate)],
      ["ROC-AUC (hold-out)", t.roc_auc.toFixed(3)],
      ["Recall", fmtPct(t.recall)],
      ["Precision", fmtPct(t.precision)],
      ["F1", t.f1.toFixed(3)],
      ["Trained", m.trained_at.slice(0, 10)],
    ];
    document.getElementById("model-info").innerHTML = rows
      .map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`)
      .join("");
  } catch {
    document.getElementById("model-info").innerHTML = "<dt>Model info unavailable</dt>";
  }
})();

syncDependentFields();
