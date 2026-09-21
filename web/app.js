const optionsEl = document.querySelector("#options");
const runButton = document.querySelector("#run");
const errorEl = document.querySelector("#error");
const mediaInput = document.querySelector("#media-input");
const mediaList = document.querySelector("#media-list");
const mediaDropzone = document.querySelector("#media-dropzone");
const mediaAttachments = [];
const MAX_IMAGE_BYTES = 16 * 1024 * 1024;
const MAX_VIDEO_BYTES = 64 * 1024 * 1024;
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
  if (mediaAttachments.length + additions.length > 8) throw new Error("A request can contain at most 8 media files.");
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
    media: debug.media,
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
        media: mediaAttachments.map(({type, data_url, name}) => ({type, data_url, name})),
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
document.querySelector("#add-media").addEventListener("click", () => mediaInput.click());
mediaDropzone.addEventListener("click", () => mediaInput.click());
mediaInput.addEventListener("change", async () => {
  try { await addMediaFiles(mediaInput.files); }
  catch (error) { errorEl.textContent = String(error); errorEl.hidden = false; }
  finally { mediaInput.value = ""; }
});
["dragenter", "dragover"].forEach((name) => mediaDropzone.addEventListener(name, (event) => {
  event.preventDefault(); mediaDropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => mediaDropzone.addEventListener(name, (event) => {
  event.preventDefault(); mediaDropzone.classList.remove("dragging");
}));
mediaDropzone.addEventListener("drop", async (event) => {
  try { await addMediaFiles(event.dataTransfer.files); }
  catch (error) { errorEl.textContent = String(error); errorEl.hidden = false; }
});
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
