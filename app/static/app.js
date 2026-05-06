const predictForm = document.getElementById("predictForm");
const imageInput = document.getElementById("imageInput");
const thresholdInput = document.getElementById("thresholdInput");
const wsolInput = document.getElementById("wsolInput");
const camThresholdGroup = document.getElementById("camThresholdGroup");
const camThresholdInput = document.getElementById("camThresholdInput");
const predictButton = document.getElementById("predictButton");

const previewPanel = document.getElementById("previewPanel");
const previewImage = document.getElementById("previewImage");
const fileMeta = document.getElementById("fileMeta");

const resultPanel = document.getElementById("resultPanel");
const resultBadge = document.getElementById("resultBadge");
const resultStats = document.getElementById("resultStats");
const plainImagePanel = document.getElementById("plainImagePanel");
const resultImage = document.getElementById("resultImage");
const wsolPanel = document.getElementById("wsolPanel");
const wsolMessage = document.getElementById("wsolMessage");
const rawJson = document.getElementById("rawJson");
const errorPanel = document.getElementById("errorPanel");

const state = {
  selectedFile: null,
  previewUrl: null,
  lastResponse: null,
};

function formatPercent(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "0.00%";
  }
  return `${(numericValue * 100).toFixed(2)}%`;
}

function setLoading(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.textContent = isLoading ? "Running inference..." : "Predict";
}

function renderError(message) {
  errorPanel.hidden = false;
  errorPanel.textContent = message;
}

function clearError() {
  errorPanel.hidden = true;
  errorPanel.textContent = "";
}

function parseError(data) {
  if (!data || typeof data !== "object") {
    return "Prediction failed.";
  }

  if (typeof data.detail === "string" && data.detail.trim()) {
    return data.detail;
  }

  if (data.error && typeof data.error.message === "string" && data.error.message.trim()) {
    return data.error.message;
  }

  return "Prediction failed.";
}

function setWsolControlVisibility() {
  camThresholdGroup.hidden = !wsolInput.checked;
}

function renderResult(data, wsolRequested) {
  state.lastResponse = data;
  resultPanel.hidden = false;
  rawJson.textContent = JSON.stringify(data, null, 2);

  const label = String(data.label || "unknown").toUpperCase();
  const confidence = formatPercent(data.confidence);

  resultBadge.textContent = `${label} · ${confidence}`;
  resultBadge.classList.remove("safe", "dangerous");
  if (data.label === "safe" || data.label === "dangerous") {
    resultBadge.classList.add(data.label);
  }

  resultStats.innerHTML = `
    <div>Safe probability: <strong>${formatPercent(data.prob_safe)}</strong></div>
    <div>Dangerous probability: <strong>${formatPercent(data.prob_dangerous)}</strong></div>
    <div>Threshold: <strong>${Number(data.threshold).toFixed(2)}</strong></div>
  `;

  plainImagePanel.hidden = wsolRequested;
  wsolPanel.hidden = !wsolRequested;

  if (wsolRequested) {
    const bbox = data.wsol?.bbox || null;

    if (bbox) {
      wsolMessage.textContent =
        `Grad-CAM bbox: x=${bbox.pixel.x_min}, y=${bbox.pixel.y_min}, ` +
        `w=${bbox.pixel.width}, h=${bbox.pixel.height}`;
    } else {
      wsolMessage.textContent =
        "No Grad-CAM region passed the current CAM threshold. Try lowering the CAM threshold.";
    }

    drawImageWithBBox(resultImage, bbox);
  }
}

function drawImageWithBBox(imgEl, bbox) {
  const canvas = document.getElementById("resultCanvas");
  const ctx = canvas.getContext("2d");

  if (!imgEl.complete || imgEl.naturalWidth === 0) {
    imgEl.onload = () => drawImageWithBBox(imgEl, bbox);
    return;
  }

  canvas.width = imgEl.naturalWidth;
  canvas.height = imgEl.naturalHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);

  if (!bbox) {
    return;
  }

  const p = bbox.pixel;
  ctx.lineWidth = Math.max(4, canvas.width / 250);
  ctx.strokeStyle = "#ff3b30";
  ctx.strokeRect(p.x_min, p.y_min, p.width, p.height);
  ctx.fillStyle = "#ff3b30";
  ctx.font = `${Math.max(18, canvas.width / 45)}px "Manrope", sans-serif`;
  ctx.fillText("Grad-CAM bbox", p.x_min, Math.max(24, p.y_min - 8));
}

imageInput.addEventListener("change", () => {
  clearError();

  const file = imageInput.files?.[0];
  if (!file) {
    return;
  }

  state.selectedFile = file;

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
  }

  state.previewUrl = URL.createObjectURL(file);
  previewImage.src = state.previewUrl;
  resultImage.src = state.previewUrl;
  previewPanel.hidden = false;
  resultPanel.hidden = true;

  previewImage.onload = () => {
    fileMeta.textContent = `${file.name} · ${previewImage.naturalWidth}×${previewImage.naturalHeight}px`;
  };
});

wsolInput.addEventListener("change", () => {
  setWsolControlVisibility();
});

predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const file = imageInput.files?.[0];
  if (!file) {
    renderError("Please choose an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("threshold", thresholdInput.value);
  formData.append("include_wsol", wsolInput.checked ? "true" : "false");
  formData.append("cam_threshold", camThresholdInput.value);

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    let data = null;
    try {
      data = await response.json();
    } catch (decodeError) {
      throw new Error("Server returned a non-JSON response.");
    }

    if (!response.ok) {
      throw new Error(parseError(data));
    }

    renderResult(data, wsolInput.checked);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Prediction failed.";
    renderError(message);
  } finally {
    setLoading(false);
  }
});

setWsolControlVisibility();
