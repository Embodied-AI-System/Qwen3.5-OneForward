"use strict";

const $ = (selector) => document.querySelector(selector);
const optionsElement = $("#options");
const errorElement = $("#error");
const mediaInput = $("#media-input");
const mediaList = $("#media-list");
const mediaDropzone = $("#media-dropzone");
const mediaAttachments = [];
const MAX_IMAGE_BYTES = 16 * 1024 * 1024;
const MAX_VIDEO_BYTES = 64 * 1024 * 1024;
const initialOptions = [
  ["billing", "Payment or subscription issues"],
  ["technical", "Bugs or integration problems"],
  ["sales", "Pricing or account questions"],
];
const batchExample = {
  state: "The checkout integration failed for the third time today. Our launch is tonight.",
  model: "jev-latest",
  questions: {
    department: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: {
        billing: "Payments, invoices, or subscriptions",
        technical: "Bugs, outages, setup, or integrations",
        sales: "Pricing, plans, or procurement",
      },
    },
    urgency: {
      type: "choice",
      instructions: "What is the request priority?",
      criteria: {
        normal: "No deadline or immediate impact",
        urgent: "Deadline, outage, or immediate business impact",
      },
    },
    sentiment: {
      type: "choice",
      instructions: "What is the customer's tone?",
      criteria: {
        neutral: "Calm and factual",
        concerned: "Worried but constructive",
        frustrated: "Clearly dissatisfied or angry",
      },
    },
  },
};

function setMode(mode) {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  $("#single-mode").hidden = mode !== "single";
  $("#batch-mode").hidden = mode !== "batch";
}

function addOption(name = "", description = "") {
  if (optionsElement.children.length >= 8) return;
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
    if (optionsElement.children.length > 2) row.remove();
    relabelOptions();
  });
  optionsElement.appendChild(row);
  relabelOptions();
}

function relabelOptions() {
  [...optionsElement.children].forEach((row, index) => {
    row.querySelector(".option-letter").textContent = `${"ABCDEFGH"[index]}.`;
  });
}

function percent(value) { return `${(value * 100).toFixed(2)}%`; }

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
}

function readDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(reader.result));
    reader.addEventListener("error", () => reject(reader.error || new Error("File read failed")));
    reader.readAsDataURL(file);
  });
}

function renderMedia() {
  mediaList.replaceChildren();
  mediaAttachments.forEach((attachment, index) => {
    const card = document.createElement("article");
    card.className = "media-card";
    const preview = attachment.type === "image" ? document.createElement("img") : document.createElement("video");
    preview.src = attachment.data_url;
    preview.alt = attachment.name;
    if (attachment.type === "video") {
      preview.controls = true;
      preview.muted = true;
      preview.preload = "metadata";
    }
    const meta = document.createElement("div");
    meta.className = "media-meta";
    const name = document.createElement("strong");
    name.textContent = attachment.name;
    const detail = document.createElement("span");
    detail.textContent = `${attachment.type} · ${formatBytes(attachment.size)}`;
    meta.append(name, detail);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "media-remove";
    remove.title = "Remove";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      mediaAttachments.splice(index, 1);
      renderMedia();
    });
    card.append(preview, meta, remove);
    mediaList.appendChild(card);
  });
  mediaDropzone.hidden = mediaAttachments.length > 0;
}

async function addMediaFiles(files) {
  const additions = [...files];
  if (!additions.length) return;
  if (mediaAttachments.length + additions.length > 8) {
    throw new Error("A request can contain at most 8 media files.");
  }
  const videoCount = mediaAttachments.filter((item) => item.type === "video").length
    + additions.filter((file) => file.type.startsWith("video/")).length;
  if (videoCount > 1) throw new Error("A request can contain at most one video.");
  for (const file of additions) {
    const type = file.type.startsWith("image/") ? "image" : file.type.startsWith("video/") ? "video" : null;
    if (!type) throw new Error(`Unsupported file type: ${file.type || file.name}`);
    const limit = type === "image" ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;
    if (file.size > limit) throw new Error(`${file.name} exceeds the ${formatBytes(limit)} limit.`);
    mediaAttachments.push({type, data_url: await readDataUrl(file), name: file.name, size: file.size});
  }
  renderMedia();
}

function buildSingleRequest() {
  const criteria = {};
  for (const row of optionsElement.children) {
    const name = row.querySelector(".name").value.trim();
    const description = row.querySelector(".description").value.trim();
    if (!name || !description) throw new Error("Every option needs a key and a description.");
    if (Object.hasOwn(criteria, name)) throw new Error(`Duplicate option key: ${name}`);
    criteria[name] = description;
  }
  return {
    state: $("#state").value,
    model: "qwen3.5-2b-oneforward",
    media: mediaAttachments.map(({type, data_url, name}) => ({type, data_url, name})),
    questions: {
      decision: {
        type: "choice",
        instructions: $("#question").value,
        criteria,
      },
    },
  };
}

function barRow(name, probability, index) {
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
  fill.style.width = percent(probability);
  track.appendChild(fill);
  const value = document.createElement("div");
  value.className = "bar-value";
  value.textContent = percent(probability);
  row.append(label, track, value);
  return row;
}

function metric(label, value) {
  const element = document.createElement("div");
  const name = document.createElement("span");
  name.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  element.append(name, strong);
  return element;
}

function renderAnswer(questionId, answer) {
  const card = document.createElement("article");
  card.className = "answer-card";
  card.dataset.answerCard = questionId;
  const head = document.createElement("div");
  head.className = "answer-head";
  const title = document.createElement("div");
  const eyebrow = document.createElement("div");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = questionId;
  const choice = document.createElement("h2");
  choice.textContent = answer.choice;
  title.append(eyebrow, choice);
  const forward = document.createElement("div");
  forward.className = "answer-forward";
  forward.textContent = `${answer._debug.inference_ms.toFixed(1)} ms batch forward`;
  head.append(title, forward);
  card.appendChild(head);
  Object.entries(answer.probabilities).forEach(([name, probability], index) => {
    card.appendChild(barRow(name, probability, index));
  });
  const metrics = document.createElement("div");
  metrics.className = "metrics";
  metrics.append(
    metric("candidate mass", percent(answer._debug.candidate_mass)),
    metric("confidence", percent(answer.confidence)),
    metric("input tokens", answer._debug.input_tokens),
    metric("batch index", answer._debug.batch_index),
  );
  card.appendChild(metrics);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Logits, vocabulary, and rendered prompt";
  const debugGrid = document.createElement("div");
  debugGrid.className = "debug-grid";
  const readout = document.createElement("pre");
  readout.textContent = JSON.stringify({
    labels: answer._debug.labels,
    candidate_tokens: answer._debug.candidate_tokens,
    candidate_token_ids: answer._debug.candidate_token_ids,
    raw_logits: answer._debug.raw_logits,
    media: answer._debug.media,
  }, null, 2);
  const vocabulary = document.createElement("pre");
  vocabulary.textContent = JSON.stringify(answer._debug.top_vocabulary, null, 2);
  debugGrid.append(readout, vocabulary);
  const prompt = document.createElement("pre");
  prompt.textContent = answer._debug.prompt;
  details.append(summary, debugGrid, prompt);
  card.appendChild(details);
  return card;
}

function renderResult(payload) {
  const list = $("#answer-list");
  list.replaceChildren();
  Object.entries(payload.answers).forEach(([questionId, answer]) => {
    list.appendChild(renderAnswer(questionId, answer));
  });
  const batch = payload._debug.batch;
  $("#latency").textContent = `${payload._debug.request_ms.toFixed(1)} ms request · ${batch.batch_size} question(s)`;
  $("#batch-stats").textContent = `${batch.batch_size} questions · ${batch.model_forward_count} forward · ${batch.shared_prefix_tokens} shared tok · ${batch.forward_ms.toFixed(1)} ms`;
  $("#response").textContent = JSON.stringify(payload, null, 2);
  $("#results").hidden = false;
}

async function runRequest(payload, button) {
  errorElement.hidden = true;
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Running batch…";
  try {
    const response = await fetch("/v1/systemone", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(body, null, 2));
    renderResult(body);
    await loadHistory();
  } catch (error) {
    errorElement.textContent = String(error);
    errorElement.hidden = false;
    await loadHistory();
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function formatHistoryTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function renderHistory(entries) {
  const list = $("#history-list");
  list.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "Run a test to create the first record.";
    list.appendChild(empty);
    return;
  }
  entries.forEach((entry) => {
    const row = document.createElement("article");
    row.className = "history-row";
    row.dataset.historyId = entry.id;
    const open = document.createElement("button");
    open.type = "button";
    open.className = "history-open";
    const title = document.createElement("strong");
    title.textContent = entry.title;
    const meta = document.createElement("span");
    const forward = entry.metrics.forward_ms;
    meta.textContent = `${formatHistoryTime(entry.created_at)} · ${entry.question_count} question(s)${forward == null ? "" : ` · ${forward.toFixed(1)} ms`}`;
    const status = document.createElement("i");
    status.className = `history-status ${entry.status}`;
    status.textContent = entry.status;
    open.append(title, meta, status);
    open.addEventListener("click", () => openHistory(entry.id));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-delete";
    remove.title = "Delete record";
    remove.setAttribute("aria-label", `Delete ${entry.title}`);
    remove.textContent = "×";
    remove.addEventListener("click", async () => {
      const response = await fetch(`/v1/history/${entry.id}`, {method: "DELETE"});
      if (response.ok) loadHistory();
    });
    row.append(open, remove);
    list.appendChild(row);
  });
}

async function loadHistory() {
  try {
    const response = await fetch("/v1/history?limit=50");
    if (!response.ok) throw new Error(`history ${response.status}`);
    renderHistory((await response.json()).data);
  } catch (error) {
    $("#history-list").textContent = `History unavailable: ${error}`;
  }
}

async function openHistory(entryId) {
  try {
    const response = await fetch(`/v1/history/${entryId}`);
    const entry = await response.json();
    if (!response.ok) throw new Error(entry.detail || `history ${response.status}`);
    document.querySelectorAll(".history-row").forEach((row) => {
      row.classList.toggle("active", row.dataset.historyId === entryId);
    });
    if (entry.response) {
      renderResult(entry.response);
      $("#results").scrollIntoView({behavior: "smooth", block: "start"});
    } else {
      errorElement.textContent = JSON.stringify(entry.error || entry, null, 2);
      errorElement.hidden = false;
    }
  } catch (error) {
    errorElement.textContent = String(error);
    errorElement.hidden = false;
  }
}

initialOptions.forEach(([name, description]) => addOption(name, description));
$("#batch-request").value = JSON.stringify(batchExample, null, 2);
document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
$("#add-option").addEventListener("click", () => addOption());
$("#add-media").addEventListener("click", () => mediaInput.click());
mediaDropzone.addEventListener("click", () => mediaInput.click());
mediaInput.addEventListener("change", async () => {
  try { await addMediaFiles(mediaInput.files); }
  catch (error) { errorElement.textContent = String(error); errorElement.hidden = false; }
  finally { mediaInput.value = ""; }
});
["dragenter", "dragover"].forEach((name) => mediaDropzone.addEventListener(name, (event) => {
  event.preventDefault();
  mediaDropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => mediaDropzone.addEventListener(name, (event) => {
  event.preventDefault();
  mediaDropzone.classList.remove("dragging");
}));
mediaDropzone.addEventListener("drop", async (event) => {
  try { await addMediaFiles(event.dataTransfer.files); }
  catch (error) { errorElement.textContent = String(error); errorElement.hidden = false; }
});
$("#run").addEventListener("click", () => {
  try { runRequest(buildSingleRequest(), $("#run")); }
  catch (error) { errorElement.textContent = String(error); errorElement.hidden = false; }
});
$("#run-batch").addEventListener("click", () => {
  try { runRequest(JSON.parse($("#batch-request").value), $("#run-batch")); }
  catch (error) { errorElement.textContent = `Invalid JSON: ${error}`; errorElement.hidden = false; }
});
$("#load-batch-example").addEventListener("click", () => {
  $("#batch-request").value = JSON.stringify(batchExample, null, 2);
});
$("#history-refresh").addEventListener("click", loadHistory);

loadHistory();
fetch("/healthz")
  .then((response) => response.json())
  .then((status) => {
    const element = $("#status");
    element.textContent = `Model ready · ${status.model} · ${status.device} · ${status.prompt_mode}`;
    element.classList.add("ready");
  })
  .catch(() => { $("#status").textContent = "Model is not ready"; });

if (new URLSearchParams(window.location.search).has("demo")) {
  runRequest(buildSingleRequest(), $("#run"));
}
