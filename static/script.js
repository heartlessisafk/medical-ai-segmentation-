const form = document.getElementById("upload-form");
const fileInput = document.getElementById("mri-file");
const runButton = document.getElementById("run-button");
const resetButton = document.getElementById("reset-button");
const dropzone = document.getElementById("dropzone");
const messageBox = document.getElementById("message-box");

const originalPreview = document.getElementById("original-preview");
const maskPreview = document.getElementById("mask-preview");
const overlayPreview = document.getElementById("overlay-preview");

const originalEmpty = document.getElementById("original-empty");
const maskEmpty = document.getElementById("mask-empty");
const overlayEmpty = document.getElementById("overlay-empty");

const confidenceValue = document.getElementById("confidence-value");
const coverageValue = document.getElementById("coverage-value");
const thresholdValue = document.getElementById("threshold-value");

function setMessage(message, type = "") {
  messageBox.textContent = message;
  messageBox.className = `message-box ${type}`.trim();
}

function showImage(imgElement, emptyElement, src) {
  imgElement.src = src;
  imgElement.hidden = false;
  emptyElement.hidden = true;
}

function resetImage(imgElement, emptyElement) {
  imgElement.src = "";
  imgElement.hidden = true;
  emptyElement.hidden = false;
}

function resetResults() {
  resetImage(originalPreview, originalEmpty);
  resetImage(maskPreview, maskEmpty);
  resetImage(overlayPreview, overlayEmpty);

  confidenceValue.textContent = "--";
  coverageValue.textContent = "--";
  thresholdValue.textContent = "0.50";
  setMessage("");
}

function validateFile(file) {
  if (!file) {
    return "Please select an image file.";
  }

  const allowedTypes = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",
  ];

  if (!allowedTypes.includes(file.type) && !/\.(png|jpg|jpeg|bmp|tif|tiff)$/i.test(file.name)) {
    return "Unsupported file type. Please upload PNG, JPG, BMP, or TIFF.";
  }

  if (file.size > 10 * 1024 * 1024) {
    return "File too large. Maximum allowed size is 10 MB.";
  }

  return "";
}

async function submitPrediction(event) {
  event.preventDefault();

  const file = fileInput.files[0];
  const validationError = validateFile(file);
  if (validationError) {
    setMessage(validationError, "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  runButton.disabled = true;
  runButton.textContent = "Processing...";
  setMessage("Running segmentation model...", "");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Prediction failed.");
    }

    showImage(originalPreview, originalEmpty, data.original_image);
    showImage(maskPreview, maskEmpty, data.mask_image);
    showImage(overlayPreview, overlayEmpty, data.overlay_image);

    confidenceValue.textContent = Number(data.confidence).toFixed(4);
    coverageValue.textContent = `${Number(data.coverage_percent).toFixed(2)}%`;
    thresholdValue.textContent = Number(data.threshold).toFixed(2);

    setMessage("Segmentation completed successfully.", "success");
  } catch (error) {
    setMessage(error.message || "Unexpected error occurred.", "error");
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run Segmentation";
  }
}

form.addEventListener("submit", submitPrediction);

resetButton.addEventListener("click", () => {
  form.reset();
  resetResults();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (files && files.length > 0) {
    fileInput.files = files;
    setMessage(`Selected file: ${files[0].name}`, "");
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    setMessage("");
    return;
  }

  const validationError = validateFile(file);
  if (validationError) {
    setMessage(validationError, "error");
    return;
  }

  setMessage(`Selected file: ${file.name}`, "");
});

(function () {
  const toggle = document.querySelector("[data-theme-toggle]");
  const root = document.documentElement;
  let theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  function renderIcon(currentTheme) {
    if (!toggle) return;
    if (currentTheme === "dark") {
      toggle.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="5" stroke="currentColor" stroke-width="2"></circle>
          <path d="M12 1V3M12 21V23M4.22 4.22L5.64 5.64M18.36 18.36L19.78 19.78M1 12H3M21 12H23M4.22 19.78L5.64 18.36M18.36 5.64L19.78 4.22" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path>
        </svg>
      `;
      toggle.setAttribute("aria-label", "Switch to light mode");
    } else {
      toggle.innerHTML = `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M21 12.79A9 9 0 1 1 11.21 3A7 7 0 0 0 21 12.79Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path>
        </svg>
      `;
      toggle.setAttribute("aria-label", "Switch to dark mode");
    }
  }

  root.setAttribute("data-theme", theme);
  renderIcon(theme);

  if (toggle) {
    toggle.addEventListener("click", () => {
      theme = theme === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", theme);
      renderIcon(theme);
    });
  }
})();