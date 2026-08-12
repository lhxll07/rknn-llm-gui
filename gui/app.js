const $ = (id) => document.getElementById(id);
const state = { config: null, polling: null, lastOutput: "" };

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function showToast(message, type = "") {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast visible ${type}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.className = "toast"; }, 3200);
}

function setStatus(value) {
  const text = $("status-text");
  const status = $("status");
  text.textContent = value;
  status.className = "status " + (value.includes("完成") ? "status-done" : value.includes("失败") ? "status-error" : value.includes("中") || value.includes("停止") ? "status-running" : "status-idle");
}

function appendLog(text) {
  const view = $("log-view");
  view.textContent = text || "";
  view.scrollTop = view.scrollHeight;
}

function updatePreset() {
  const platform = $("platform").value;
  const quantization = $("quantization").value;
  const preset = state.config.presets[platform];
  const quant = state.config.quantizations[quantization];
  const optimized = quant.quantized_dtype.startsWith("w4a16") ? 0 : preset.optimization_level;
  $("max-context").value = preset.max_context;
  $("preset-summary").textContent = `${platform}  ·  ${quantization}  ·  ${preset.num_npu_core} 个 NPU 核心  ·  优化等级 ${optimized}`;
}

function populateConfig(config) {
  state.config = config;
  $("platform").replaceChildren(...Object.keys(config.presets).map((name) => new Option(name, name)));
  $("quantization").replaceChildren(...Object.keys(config.quantizations).map((name) => new Option(name, name)));
  $("vision-model").replaceChildren(...Object.keys(config.vision_models).map((name) => new Option(name, name)));
  $("platform").value = "RK3588";
  $("quantization").value = "W8A8";
  $("device").value = config.default_device;
  if (!config.cuda) $("device").querySelector("option[value=cuda]").textContent = "CUDA（不可用）";
  updatePreset();
  $("footer-platform").textContent = config.platform;
  $("platform-caption").textContent = `本地模型转换 · ${config.platform}`;
}

function toggleVision() {
  const vision = $("conversion-mode").value === "视觉模型";
  $("vision-fields").hidden = !vision;
  $("calibration-field").hidden = vision;
  if (vision) applyVisionModel();
}

function applyVisionModel() {
  const model = state.config.vision_models[$("vision-model").value];
  $("vision-height").value = model.height;
  $("vision-width").value = model.width;
}

function collectForm() {
  return {
    model_path: $("model-path").value,
    calibration_path: $("calibration-path").value,
    output_path: $("output-path").value,
    platform: $("platform").value,
    quantization: $("quantization").value,
    device: $("device").value,
    dtype: $("dtype").value,
    max_context: $("max-context").value,
    conversion_mode: $("conversion-mode").value,
    vision_model: $("vision-model").value,
    vision_onnx_path: $("vision-onnx").value,
    vision_rknn_path: $("vision-rknn").value,
    vision_batch_size: $("vision-batch").value,
    vision_height: $("vision-height").value,
    vision_width: $("vision-width").value,
  };
}

function setFormError(message = "") {
  const target = $("form-error");
  target.textContent = message;
  target.hidden = !message;
}

async function refreshJob() {
  try {
    const job = await request("/api/job");
    setStatus(job.status);
    if (job.log) appendLog(job.log);
    if (job.output && job.output !== state.lastOutput) {
      state.lastOutput = job.output;
      $("output-box").hidden = false;
      $("output-path-display").textContent = job.output;
      showToast("转换完成", "success");
    }
    if (!["转换中", "正在停止"].includes(job.status) && state.polling) {
      window.clearInterval(state.polling);
      state.polling = null;
    }
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function startConversion(event) {
  event.preventDefault();
  setFormError("");
  const button = $("start-conversion");
  button.disabled = true;
  button.classList.add("loading");
  try {
    const result = await request("/api/start", { method: "POST", body: JSON.stringify(collectForm()) });
    appendLog("任务已启动，等待日志……");
    setStatus("转换中");
    state.lastOutput = "";
    $("output-box").hidden = true;
    if (state.polling) window.clearInterval(state.polling);
    state.polling = window.setInterval(refreshJob, 700);
    await refreshJob();
  } catch (error) {
    setFormError(error.message);
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
  }
}

async function stopConversion() {
  try {
    const job = await request("/api/stop", { method: "POST", body: "{}" });
    setStatus(job.status);
    appendLog(job.log);
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function checkEnvironment() {
  const button = $("check-environment");
  button.disabled = true;
  button.classList.add("loading");
  try {
    const result = await request("/api/environment", { method: "POST", body: "{}" });
    $("environment-view").textContent = result.text;
    showToast("环境检查完成", "success");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("loading");
  }
}

async function openFolder(path) {
  if (!path) return;
  try {
    await request("/api/open-folder", { method: "POST", body: JSON.stringify({ path }) });
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function selectDirectory(initialPath = "") {
  const result = await request("/api/select-directory", {
    method: "POST",
    body: JSON.stringify({ initial_path: initialPath }),
  });
  return result.path;
}

async function chooseModelDirectory() {
  const button = $("select-model-folder");
  button.disabled = true;
  button.classList.add("loading-dark");
  try {
    const selected = await selectDirectory($("model-path").value);
    if (selected) $("model-path").value = selected;
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("loading-dark");
  }
}

async function chooseOutputDirectory() {
  const button = $("select-output-folder");
  button.disabled = true;
  button.classList.add("loading-dark");
  try {
    const current = $("output-path").value.trim();
    const filename = current.split(/[\\/]/).pop() || "model.rkllm";
    const selected = await selectDirectory(current);
    if (selected) $("output-path").value = `${selected.replace(/\/$/, "")}/${filename}`;
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.classList.remove("loading-dark");
  }
}

async function init() {
  try {
    populateConfig(await request("/api/config"));
  } catch (error) {
    showToast("无法连接到本地服务", "error");
  }
  $("platform").addEventListener("change", updatePreset);
  $("quantization").addEventListener("change", updatePreset);
  $("conversion-mode").addEventListener("change", toggleVision);
  $("vision-model").addEventListener("change", applyVisionModel);
  $("settings-form").addEventListener("submit", startConversion);
  $("stop-conversion").addEventListener("click", stopConversion);
  $("check-environment").addEventListener("click", checkEnvironment);
  $("refresh-environment").addEventListener("click", checkEnvironment);
  $("copy-log").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("log-view").textContent);
    showToast("日志已复制", "success");
  });
  $("select-model-folder").addEventListener("click", chooseModelDirectory);
  $("select-output-folder").addEventListener("click", chooseOutputDirectory);
  $("open-output").addEventListener("click", () => openFolder($("output-path-display").textContent));
}

init();
