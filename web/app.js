const optionsEl = document.querySelector("#options");
const runButton = document.querySelector("#run");
const errorEl = document.querySelector("#error");
const initialOptions = [
  ["billing", "Payment or subscription issues"],
  ["technical", "Bugs or integration problems"],
  ["sales", "Pricing or account questions"],
];

function addOption(name = "", description = "") {
  if (optionsEl.children.length >= 8) return;
  const row = document.createElement("div");
  row.className = "option-row";
  row.innerHTML = `
    <div class="option-letter"></div>
    <input class="name" aria-label="Option name" placeholder="option key" />
    <input class="description" aria-label="Option description" placeholder="semantic description" />
    <button type="button" class="remove" title="Remove">×</button>`;
  row.querySelector(".name").value = name;
  row.querySelector(".description").value = description;
  row.querySelector(".remove").addEventListener("click", () => {
    if (optionsEl.children.length > 2) row.remove();
    relabel();
  });
  optionsEl.appendChild(row);
  relabel();
}

function relabel() {
  [...optionsEl.children].forEach((row, index) => {
    row.querySelector(".option-letter").textContent = `${"ABCDEFGH"[index]}.`;
  });
}

function pct(value) { return `${(value * 100).toFixed(2)}%`; }

function renderResult(payload) {
  const answer = payload.answers.decision;
  const debug = answer._debug;
  document.querySelector("#choice").textContent = answer.choice;
  document.querySelector("#latency").textContent = `${debug.inference_ms.toFixed(1)} ms forward`;
  document.querySelector("#mass").textContent = pct(debug.candidate_mass);
  document.querySelector("#confidence").textContent = pct(answer.confidence);
  document.querySelector("#tokens").textContent = debug.input_tokens;

  const bars = document.querySelector("#bars");
  bars.replaceChildren();
  Object.entries(answer.probabilities).forEach(([name, probability], index) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("div");
    label.className = "bar-label";
    const letter = document.createElement("b");
    letter.textContent = "ABCDEFGH"[index];
    label.append(letter, document.createTextNode(name));
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = pct(probability);
    track.appendChild(fill);
    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = pct(probability);
    row.append(label, track, value);
    bars.appendChild(row);
  });

  document.querySelector("#readout").textContent = JSON.stringify({
    labels: debug.labels,
    candidate_tokens: debug.candidate_tokens,
    candidate_token_ids: debug.candidate_token_ids,
    raw_logits: debug.raw_logits,
  }, null, 2);
  document.querySelector("#top-vocab").textContent = JSON.stringify(debug.top_vocabulary, null, 2);
  document.querySelector("#prompt").textContent = debug.prompt;
  document.querySelector("#response").textContent = JSON.stringify(payload, null, 2);
  document.querySelector("#results").hidden = false;
}

async function run() {
  errorEl.hidden = true;
  const criteria = {};
  for (const row of optionsEl.children) {
    const name = row.querySelector(".name").value.trim();
    const description = row.querySelector(".description").value.trim();
    if (!name || !description) {
      errorEl.textContent = "Every option needs a key and a description.";
      errorEl.hidden = false;
      return;
    }
    if (Object.hasOwn(criteria, name)) {
      errorEl.textContent = `Duplicate option key: ${name}`;
      errorEl.hidden = false;
      return;
    }
    criteria[name] = description;
  }

  runButton.disabled = true;
  runButton.textContent = "Running one forward…";
  try {
    const response = await fetch("/v1/systemone", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        state: document.querySelector("#state").value,
        model: "qwen3.5-2b-oneforward",
        questions: {decision: {
          type: "choice",
          instructions: document.querySelector("#question").value,
          criteria,
        }},
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(payload, null, 2));
    renderResult(payload);
  } catch (error) {
    errorEl.textContent = String(error);
    errorEl.hidden = false;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run one-forward decision";
  }
}

initialOptions.forEach(([name, description]) => addOption(name, description));
document.querySelector("#add-option").addEventListener("click", () => addOption());
runButton.addEventListener("click", run);

fetch("/healthz")
  .then((response) => response.json())
  .then((status) => {
    const el = document.querySelector("#status");
    el.textContent = `Model ready · ${status.model} · ${status.device} · ${status.prompt_mode}`;
    el.classList.add("ready");
  })
  .catch(() => {
    document.querySelector("#status").textContent = "Model is not ready";
  });

// A deterministic, opt-in demo mode is useful for documentation screenshots.
if (new URLSearchParams(window.location.search).has("demo")) run();
