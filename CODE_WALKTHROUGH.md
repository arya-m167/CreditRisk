# Code walkthrough — Explainable Credit Risk Scoring

Every file, what it does, and — the important part — *why it's built that way instead of some
simpler or more obvious way*. Read this once end to end, then use it as reference before an
interview. Anywhere you see "**Why**," that's the sentence to have ready if someone asks you to
defend a decision.

---

## The shape of the system, and why it's shaped this way

```
React (port 5173)  →  FastAPI (port 8000)  →  PostgreSQL / SQLite
                              ↓
                     XGBoost model + SHAP
```

**Why three separate pieces instead of one script?** Because that's not how real systems get
used. A notebook that prints an accuracy score is a research artifact — it has one user (you)
and one session. An API means *any* client can use the model: a web form today, a mobile app or
a batch job later, without touching the model code. A database means predictions outlive the
request that created them — you can ask "what did we predict for applicant #4,821 three weeks
ago" at any point after the fact. Separating these concerns is the actual skill being
demonstrated here; the model itself is almost the easy part.

**Why FastAPI specifically, not Flask or Django?** FastAPI gives you request validation (via
Pydantic) and auto-generated interactive docs for free, just from type hints — meaning less code
to write and less code that can silently accept bad data. For a project meant to look
production-minded, "the API validates its own inputs and documents itself" is a stronger signal
than "the API works if you send it exactly the right JSON."

---

## Backend

### `backend/data/credit_card_default.csv`
The raw dataset — 30,000 Taiwanese credit card accounts, 23 features, 1 target
(`default.payment.next.month`).

**Why this dataset over building your own?** Because the point of this project is to demonstrate
engineering and ML judgment, not data collection. A real, published, peer-reviewed dataset with
a known paper behind it (Yeh & Lien, 2009) gives you a baseline to compare against and a citation
to point to — "I got AUC 0.778, the literature on this exact dataset is in a similar range" is a
much stronger claim than an unverifiable number from a dataset nobody else has touched.

### `backend/app/ml/train.py`
Runs once, offline, to produce the model. Not called by the live app.

**Why separate the training script from the API entirely, instead of training on startup?**
Training takes time and depends on having the CSV, scikit-learn, XGBoost all present — none of
that should be a requirement for the API to boot quickly and reliably. Separating them also
mirrors how real ML teams work: a data scientist trains and versions a model; a backend service
loads whatever the current approved model is. Conflating the two would make the API's startup
time unpredictable and would mean anyone deploying the app needs the full training dataset just
to run a web server.

```python
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
```
**Why fold undocumented codes into "other" instead of dropping those rows?** Only ~400 of 30,000
rows have these undocumented values — dropping them wouldn't hurt much, but it's still throwing
away real, paid-for-by-nobody-twice data for no real gain. Folding them into an existing
"other" category keeps every row usable and doesn't require inventing a new category the model
would rarely see (and thus wouldn't learn anything reliable about).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
```
**Why `stratify=y`?** Only ~22% of applicants default. Without stratifying, a random split could
by chance put 25% defaulters in train and 18% in test — then your test metrics wouldn't
represent the same problem your training data represents. Stratifying guarantees both sets keep
the same ~22% base rate, so the AUC you measure is trustworthy.

**Why measure a logistic regression baseline at all, if XGBoost is the model you're shipping?**
Because "AUC 0.778" is a meaningless number on its own — you need something to compare it to.
The baseline answers the implicit question "was the fancier model worth it?" (yes, +6 points of
AUC) and shows you understand that model selection is a comparison, not a single number picked
in isolation.

```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```
**Why this instead of SMOTE (synthetic oversampling)?** SMOTE generates *fake* rows by
interpolating between real ones to balance the classes. That's a reasonable technique in
general, but it directly conflicts with this project's other goal: every prediction needs to be
explainable via SHAP, and SHAP explanations are only meaningful for real applicants. If the
model were trained partly on synthetic people, you couldn't cleanly claim "here's what mattered
for this real applicant" without the caveat that some of what the model learned came from data
that doesn't correspond to anyone real. `scale_pos_weight` solves the imbalance problem by
changing how much the *loss function* penalizes mistakes on the minority class — no synthetic
rows involved, so every explanation stays traceable to a real person.

```python
max_depth=4, subsample=0.8, colsample_bytree=0.8
```
**Why these specific, fairly conservative values instead of tuning aggressively for the highest
possible AUC?** A shallow, subsampled model is a deliberate choice against overfitting on a
30,000-row dataset — deeper trees given more freedom would likely fit noise in the training set
and look better on paper while generalizing worse. It's also an honest choice: chasing every
last point of AUC on a well-studied public dataset risks fooling yourself with an artifact of
hyperparameter search rather than a genuinely better model. If asked "did you tune
hyperparameters extensively," the honest answer is no — these are reasonable, literature-typical
defaults, and that's a fine thing to say plainly rather than overclaiming tuning that didn't
happen.

```python
def fairness_check(model, X_test, y_test, sex_test):
    ...
    fp = ((p_g == 1) & (y_g == 0)).sum()
```
**Why false positive rate specifically, out of several possible fairness metrics?** In a lending
context, a false positive means a creditworthy person gets denied or charged more — that's the
harm regulators and ethicists focus on most in credit scoring specifically (as opposed to, say,
false negative rate, which would matter more if the harm you cared about was the *lender's*
losses rather than the applicant's treatment). Picking the metric that matches the actual
real-world harm, and being able to say why you picked that one over the alternatives, is the
difference between a checkbox and a real analysis.

**Why save `background.joblib` (a sample of training rows) as a separate artifact?** SHAP's
`TreeExplainer` can work in a couple of modes — one needs a reference/background dataset to
compare each new prediction against. Saving a fixed 200-row sample at training time means every
future explanation is computed against a stable, known reference, rather than the API needing
access to the full training set (which shouldn't need to travel with a deployed API) just to
explain a single prediction.

### `backend/app/database.py`
```python
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./credit_risk.db")
```
**Why default to SQLite locally but use Postgres in Docker, via one environment variable, rather
than picking one database and using it everywhere?** Local development should have zero setup
cost — SQLite is a single file, nothing to install or configure. Deployment should use a real
database that handles concurrent writes properly, which SQLite is not designed for. Reading the
connection string from an environment variable instead of hardcoding it is the standard way to
let the *same code* behave correctly in both situations — you're not choosing between "quick to
develop" and "production-ready," you get both from one codebase.

**Why the `get_db()` generator/dependency pattern instead of just opening a connection at the
top of each function?** It guarantees the database session is always closed, even if the
request throws an exception halfway through — the `finally` block runs regardless. Without this
pattern, an error partway through a request could leak an open connection, which under load
eventually exhausts the connection pool and takes the whole app down.

### `backend/app/models/db_models.py`
```python
features_json = Column(JSON)
```
alongside individual columns like `limit_bal`, `sex`, `age`.

**Why store the data twice — once as individual columns, once as one JSON blob?** They serve
different jobs. The individual columns exist so the database can efficiently filter and
aggregate (`WHERE age > 30`, `AVG(limit_bal)`) — that's what SQL columns are good at. The JSON
blob exists so that months later, `/explain/{id}` can reconstruct *exactly* what the model saw
for that applicant, including fields that don't have their own column, without needing the
original request to still exist anywhere. If you only kept individual columns, you'd need one
column per feature (23 of them) cluttering the schema for fields you'll rarely query directly;
if you only kept the JSON blob, you couldn't write a simple SQL query for portfolio-level
questions. Doing both is a genuine practical tradeoff, not just being lazy.

**Why is there a `model_version` field that the code barely uses yet?** Because in a real
system, models get retrained. If the false-positive rate on new applicants suddenly changes next
quarter, the first question anyone asks is "did we change the model?" Without a version stamp
on every stored prediction, that question is unanswerable after the fact. It's a small field
that costs nothing to include now and would be genuinely painful to add retroactively later —
worth having even though the current single-model version of this project doesn't exercise it
much.

### `backend/app/schemas.py`
**Why define Pydantic schemas at all, instead of just accepting a raw dict/JSON in the endpoint
and pulling fields out manually?** Two reasons. First, validation for free: `SEX: int = Field(...,
ge=1, le=2)` means a malformed request gets rejected automatically, with a clear error, before
any of your logic runs — you're not writing `if sex not in [1, 2]: raise ...` by hand for 23
fields. Second, self-documentation: FastAPI turns these schemas into the interactive `/docs`
page automatically. Anyone (a recruiter, a teammate) can go to `/docs` and see exactly what the
API expects without reading a line of your code.

### `backend/app/ml/inference.py`
```python
_model = joblib.load(...)   # runs once, at import time
```
**Why load the model once at module import time instead of inside the `score_applicant`
function?** Loading a serialized model from disk isn't free — it can take anywhere from tens to
hundreds of milliseconds. If that happened on every request, every single prediction would carry
that overhead, which compounds badly under any real load. Loading once, into a module-level
variable, means the cost is paid exactly once, at server startup, and every request after that
just reuses the already-loaded object in memory.

```python
_explainer = shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")
```
**Why `TreeExplainer` and not the more general `KernelExplainer`?** `TreeExplainer` is written
specifically for tree-based models like XGBoost — it computes *exact* SHAP values by walking the
actual decision paths in the trees, and it's fast. `KernelExplainer` works for any model type
(including ones with no internal structure to exploit, like a black-box neural net) but is
slower and only approximates the true SHAP values via sampling. Since the model here actually is
a tree ensemble, using the tree-specific explainer is strictly better — no reason to pay for a
generic, approximate method when an exact, fast one applies directly.

```python
contributions = sorted(..., key=lambda t: abs(t[2]), reverse=True)[:top_n]
```
**Why sort by absolute value and show only the top 6, instead of all 23 features?** All 23
SHAP values are computed regardless, but a bar chart with 23 rows is not something a person
scans in one glance — the whole point of an explanation is that a human can look at it and
immediately understand the decision. Sorting by magnitude of impact (regardless of direction)
and cutting to a handful surfaces the factors that actually moved the needle and drops the ones
that were essentially noise for this particular applicant.

### `backend/app/routers/predictions.py`
**Why does `/predict` write to the database inside the same function that returns the
prediction, rather than scoring first and saving separately later?** Because an unsaved
prediction is functionally the same as a prediction that never happened, from an audit
standpoint — if the app crashed between "return the score to the user" and "save it," you'd have
made a real credit decision with zero record of it. Saving before returning means the audit
trail is only ever missing a record if the *entire request* failed, in which case the user never
got a score either — consistent either way.

**Why does `/explain/{id}` take a *stored prediction's ID* rather than accepting the applicant's
data again directly?** Two reasons. It proves the explanation corresponds to a decision that was
actually made and logged (you can't ask to "explain" a hypothetical that was never scored), and
it means the frontend doesn't need to hold onto and re-send 23 fields of form data just to ask
for an explanation a few seconds later — it just needs the ID it already got back from
`/predict`.

### `backend/app/main.py`
```python
Base.metadata.create_all(bind=engine)
```
**Why create the database table at app startup instead of requiring a separate manual migration
step?** For a project this size, requiring `alembic upgrade head` or similar before the app can
even run adds friction for anyone trying to clone and run this quickly, for very little benefit
— there's exactly one table and it hasn't changed shape over time. (In a larger, longer-lived
project with a schema that changes across versions, a real migration tool like Alembic would
replace this — worth saying explicitly if asked, so it doesn't look like you don't know the
limitation.)

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
**Why is this needed at all?** Browsers block JavaScript from making requests to a different
origin (different domain *or port*) than the page itself came from, by default, as a security
protection against malicious sites silently calling APIs on your behalf. Your React app
(`localhost:5173`) and your API (`localhost:8000`) count as different origins purely because of
the port difference, even though they're both "yours" and both on `localhost`. CORS middleware
is the API explicitly telling browsers "requests from this origin are allowed."

**Why `allow_origins=["*"]` (anyone) instead of just the frontend's actual URL?** Convenience
during development — you don't need to update the API every time you test from a different
local port or a temporary deploy preview URL. This is explicitly called out in the code as
something to tighten before a real deployment — in production, you'd list only your actual
frontend's domain, so no other website could make authenticated-looking requests to your API on
a user's behalf.

---

## Frontend

### `frontend/index.html`
```html
<div id="root"></div>
<script type="module" src="/src/main.jsx"></script>
```
**Why is this file almost empty?** Because it's not really "the page" — it's the mounting point.
React takes over everything inside `#root` and renders the entire UI from JavaScript. This is
standard for any React single-page app: the HTML file's only job is to load the JS bundle and
give it a place to attach to. There's no server-rendered content here because nothing about this
app benefits from server-side rendering — it's an internal tool behind a login-free demo, not a
public page that needs to be indexed by search engines or load meaningfully before JS runs.

### `frontend/src/main.jsx`
```jsx
createRoot(document.getElementById('root')).render(
  <StrictMode><App /></StrictMode>
)
```
**Why wrap the app in `<StrictMode>`?** It's a development-only tool that intentionally
double-invokes certain functions to surface bugs (like side effects in the wrong place) that
would otherwise only show up unpredictably in production. It adds no runtime cost in the actual
production build — pure upside during development, which is why it's included by default by
every modern React project scaffold.

### `frontend/vite.config.js`
```js
export default defineConfig({ plugins: [react()] })
```
**Why Vite instead of Create React App or a hand-rolled Webpack config?** Vite's dev server
starts near-instantly and updates the browser near-instantly on save (hot module replacement),
because it serves your source files directly over native ES modules during development instead
of bundling everything up front the way older tools do. Create React App is effectively
unmaintained at this point; reaching for Vite is the current standard default for a new React
project and signals you're working with current tooling rather than what was standard five years
ago.

### `frontend/src/api.js`
```js
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
```
**Why read the backend's URL from an environment variable instead of hardcoding
`localhost:8000`?** The exact same reasoning as `DATABASE_URL` on the backend: this file should
work unmodified whether you're running locally or pointing at a real deployed API. Hardcoding
the URL would mean every deployment requires a code change and a rebuild just to point
somewhere else — an environment variable means the same built JS bundle can be configured per
environment.

**Why one shared `request()` helper wrapping `fetch`, instead of calling `fetch` directly in
each of the three API functions?** So error handling (checking `res.ok`, throwing a readable
error) and the base URL only need to be written once. If the API's error format changes, or you
add auth headers later, there's exactly one place to change it instead of three.

### `frontend/src/App.jsx`
**Why does one top-level component hold all the shared state (`prediction`, `explanation`,
`tab`), rather than each child component managing its own?** Because `ExplanationPanel` needs
the prediction that `ScoreForm`'s submission produced, and `PortfolioDashboard` needs to know
when a new prediction happened so it can refresh. When sibling components need to share data,
that data has to live in their common parent — this is the most basic version of React's "lift
state up" pattern, and reaching for a state management library (Redux, Zustand, etc.) here would
be solving a problem this small app doesn't actually have.

```js
const exp = await explainPrediction(pred.id);
```
**Why call `explainPrediction` automatically right after `scoreApplicant`, instead of adding a
separate "explain" button the user has to click?** Because an unexplained risk score is exactly
the thing this whole project argues against — showing a number with no reasoning defeats the
point. Making the explanation appear automatically, without an extra step, is a small UX
decision that reinforces the project's actual thesis.

### `frontend/src/components/ScoreForm.jsx`
**Why pre-fill the form with a real example applicant instead of leaving it blank?** A blank
23-field form is a wall a reviewer has to climb before they see anything work. Pre-filling with
a real row from the dataset means anyone — a recruiter, an interviewer — can open the app and
see a working prediction within one click, with zero required typing. It also guarantees the
demo path always uses realistic values, since the fields were never hand-typed by a user who
might enter something nonsensical.

### `frontend/src/components/ExplanationPanel.jsx`
```js
const pct = (Math.abs(f.shap_value) / maxAbs) * 50;
```
**Why scale each bar relative to the largest value among the shown factors, instead of using a
fixed scale?** A fixed scale (say, "the bar covers the full width of the widest possible SHAP
value ever seen") would make most applicants' charts look nearly flat, since any individual
factor's contribution is usually small relative to the model's full possible range. Scaling
relative to *this applicant's own* top factor means the chart always uses the available space
well and the relative importance between this applicant's own factors is still accurate — which
is the comparison that actually matters for reading the chart.

**Why a diverging bar (left/right from a center line) instead of a plain bar chart?** SHAP
values are signed — a factor can push risk up or pull it down. A plain bar chart (all bars
starting from zero on one side) would only show *magnitude*, losing the direction, which is
half the information a risk officer actually needs ("is this hurting or helping this applicant's
case").

### `frontend/src/components/PortfolioDashboard.jsx`
**Why fetch fresh stats every time `refreshKey` changes instead of updating the numbers locally
in JavaScript after each new prediction?** Because the aggregates (average default rate, counts
per risk band) are computed by the *database*, from *all* historical predictions — not just the
one you just made. Recomputing them correctly on the frontend would mean duplicating logic that
already exists correctly on the backend, and risking the two falling out of sync. Refetching is
slightly less "instant" but is guaranteed correct.

### `frontend/src/index.css`
**Why hand-written CSS with custom properties instead of Tailwind or a component library?** At
this size — a handful of components, one consistent visual language — a utility framework adds
a learning curve and bundle weight without buying much; plain CSS variables (`--bg`, `--panel`,
`--text`, etc.) already give you a single source of truth for the color palette that's trivial
to change project-wide. The dark, monospace-numbers aesthetic was a deliberate choice to make
the tool *look* like something a risk analyst would actually use day to day, rather than a
generic light dashboard template — visual choices tied to the subject matter read as intentional
rather than default.

---

## Docker & deployment

### `backend/Dockerfile`
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
```
**Why copy `requirements.txt` and install dependencies *before* copying the rest of the app
code, instead of copying everything at once?** Docker caches each instruction as a layer, and
reuses a cached layer if nothing that produced it changed. Application code changes constantly;
dependencies change rarely. Ordering it this way means editing a single line of Python and
rebuilding the image doesn't force a full dependency reinstall — only the (usually rare)
`requirements.txt` change does. This is a standard Docker optimization, and getting the order
backwards is one of the most common beginner mistakes.

### `frontend/Dockerfile`
A two-stage build: build the React app with Node in stage one, then copy *only the built static
files* into a fresh, tiny Nginx image for stage two.

**Why two stages instead of just running `npm run dev` inside a container?** The final image
that actually gets deployed and run doesn't need Node.js, npm, or any of your source code — it
only needs to serve static HTML/CSS/JS files, which Nginx does extremely efficiently. Shipping a
smaller image means faster deploys, a smaller attack surface (nothing to exploit if there's no
Node runtime present), and no chance of accidentally shipping source code or `node_modules` to
production.

### `docker-compose.yml`
```yaml
depends_on:
  db:
    condition: service_healthy
```
**Why wait for a health check instead of just starting the backend after the database
container starts?** "Container started" and "database ready to accept connections" are not the
same moment — Postgres takes a brief moment to initialize after its process starts. Without
this, the backend could try to connect during that gap and crash on startup, which is a classic,
confusing "works when I restart it, fails on first boot" bug. The health check makes the startup
order actually correct, not just usually correct.

---

## The one-paragraph pitch, if someone asks "what is this project"

*"I built an explainable credit default risk model — XGBoost trained on a published academic
dataset, served through a FastAPI backend with a Postgres audit log, with a React dashboard on
top. Every prediction comes with a SHAP explanation showing exactly which factors drove the
score, and I ran a fairness check that found the model flags male applicants incorrectly more
often than female applicants — which is the kind of finding a real model-risk review would
catch before deployment. The whole thing is containerized and I made a deliberate call not to
use synthetic oversampling for the class imbalance, specifically so every explanation stays
traceable to a real applicant."*

That's a 45-second answer that covers the ML, the engineering, and the judgment call — which is
exactly what two minutes of interview time is actually testing for.
