const $ = (id) => document.getElementById(id);
const state = {
  config: null,
  polling: null,
  refreshing: false,
  lastOutput: "",
  lastOutputs: [],
  outputDir: "",
  logOffset: 0,
  logFollow: true,
  pollErrors: 0,
  presetContext: null,
  quantizationTouched: false,
};

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`服务返回了无效响应（HTTP ${response.status}）`);
  }
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

function setStatus(value, statusCode = "") {
  const text = $("status-text");
  const status = $("status");
  text.textContent = value;
  const className = statusCode === "completed" || value.includes("完成")
    ? "status-done"
    : statusCode === "failed" || value.includes("失败")
      ? "status-error"
      : statusCode === "cancelled" || value === "已停止"
        ? "status-stopped"
        : statusCode === "running" || statusCode === "stopping" || value.includes("中")
          ? "status-running"
          : "status-idle";
  status.className = `status ${className}`;
}

function appendLog(text) {
  if (!text) return;
  const view = $("log-view");
  const current = view.textContent;
  view.textContent = current.startsWith("开始转换后") ? text : current ? `${current}\n${text}` : text;
  if (state.logFollow) view.scrollTop = view.scrollHeight;
}

function updateControls(statusCode) {
  const busy = ["running", "stopping"].includes(statusCode);
  $("start-conversion").disabled = busy;
  $("stop-conversion").disabled = !busy;
}

function updatePreset() {
  if (!state.config) return;
  const platform = $("platform").value;
  const quantization = $("quantization").value;
  const preset = state.config.presets[platform];
  const quant = state.config.quantizations[quantization];
  if (!preset || !quant) return;
  const optimized = quant.quantized_dtype.startsWith("w4a16") ? 0 : preset.optimization_level;
  const context = $("max-context");
  if (state.presetContext === null || context.value === "" || Number(context.value) === state.presetContext) {
    context.value = preset.max_context;
  }
  state.presetContext = preset.max_context;
  $("preset-summary").textContent = `${platform}  ·  ${quantization}  ·  ${preset.num_npu_core} 个 NPU 核心  ·  优化等级 ${optimized}`;
}

function updatePlatformPreset() {
  if (!state.config) return;
  const platform = $("platform").value;
  const preset = state.config.presets[platform];
  if (!preset) return;
  if (!state.quantizationTouched) {
    $("quantization").value = preset.default_quantization || $("quantization").value;
  }
  updatePreset();
}

function populateConfig(config) {
  state.config = config;
  $("platform").replaceChildren(...Object.keys(config.presets).map((name) => new Option(name, name)));
  $("quantization").replaceChildren(...Object.keys(config.quantizations).map((name) => new Option(name, name)));
  $("vision-model").replaceChildren(...Object.keys(config.vision_models).map((name) => new Option(name, name)));
  $("platform").value = config.default_preset || Object.keys(config.presets)[0];
  $("quantization").value = config.default_quantization || config.presets[$("platform").value].default_quantization || Object.keys(config.quantizations)[0];
  state.quantizationTouched = false;
  $("device").value = config.default_device;
  const cudaOption = $("device").querySelector("option[value=cuda]");
  cudaOption.disabled = !config.cuda;
  if (!config.cuda) cudaOption.textContent = "CUDA（不可用）";
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
  const model = state.config?.vision_models[$("vision-model").value];
  if (!model) return;
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

function stopPolling() {
  if (state.polling) window.clearInterval(state.polling);
  state.polling = null;
}

function startPolling() {
  state.pollErrors = 0;
  if (!state.polling) state.polling = window.setInterval(() => { void refreshJob(); }, 700);
}

function renderOutputs(outputs, primaryOutput, outputDir) {
  const paths = Array.isArray(outputs) && outputs.length ? outputs : primaryOutput ? [primaryOutput] : [];
  if (!paths.length) return;
  state.lastOutput = primaryOutput || paths[0];
  state.lastOutputs = paths;
  state.outputDir = outputDir || paths[0].replace(/[\\/][^\\/]*$/, "");
  $("output-box").hidden = false;
  $("output-path-display").textContent = paths.join("\n");
}

async function refreshJob({ notify = true } = {}) {
  if (state.refreshing) return null;
  state.refreshing = true;
  try {
    const job = await request(`/api/job?after=${encodeURIComponent(state.logOffset)}`);
    state.pollErrors = 0;
    setStatus(job.status, job.status_code);
    state.logOffset = job.log_offset || 0;
    updateControls(job.status_code);
    if (job.log) appendLog(job.log);
    if (job.outputs?.length || job.output) {
      const outputs = Array.isArray(job.outputs) ? job.outputs : [];
      const isNewOutput = job.output !== state.lastOutput || outputs.join("\n") !== state.lastOutputs.join("\n");
      renderOutputs(outputs, job.output, job.output_dir);
      if (isNewOutput && notify && job.status_code === "completed") showToast("转换完成", "success");
    }
    if (!["running", "stopping"].includes(job.status_code)) stopPolling();
    return job;
  } catch (error) {
    state.pollErrors += 1;
    if (state.pollErrors >= 3) {
      stopPolling();
      setStatus("连接中断", "failed");
      updateControls("idle");
      showToast("连接本地服务失败，已停止刷新", "error");
    } else {
      showToast(error.message, "error");
    }
    return null;
  } finally {
    state.refreshing = false;
  }
}

async function startConversion(event) {
  event.preventDefault();
  setFormError("");
  const button = $("start-conversion");
  button.disabled = true;
  $("stop-conversion").disabled = false;
  button.classList.add("loading");
  try {
    await request("/api/start", { method: "POST", body: JSON.stringify(collectForm()) });
    setStatus("转换中", "running");
    updateControls("running");
    $("log-view").textContent = "";
    state.logOffset = 0;
    state.logFollow = true;
    appendLog("任务已启动，等待日志……");
    state.lastOutput = "";
    state.lastOutputs = [];
    state.outputDir = "";
    $("output-box").hidden = true;
    stopPolling();
    startPolling();
    await refreshJob();
  } catch (error) {
    setFormError(error.message);
    showToast(error.message, "error");
    updateControls("idle");
  } finally {
    button.classList.remove("loading");
  }
}

async function stopConversion() {
  try {
    const job = await request("/api/stop", { method: "POST", body: "{}" });
    setStatus(job.status, job.status_code);
    updateControls(job.status_code);
    if (["running", "stopping"].includes(job.status_code)) startPolling();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function checkEnvironment(button = $("check-environment")) {
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
    const job = await refreshJob({ notify: false });
    if (job && ["running", "stopping"].includes(job.status_code)) startPolling();
  } catch (error) {
    showToast("无法连接到本地服务", "error");
  }
  $("platform").addEventListener("change", updatePlatformPreset);
  $("quantization").addEventListener("change", () => {
    state.quantizationTouched = true;
    updatePreset();
  });
  $("conversion-mode").addEventListener("change", toggleVision);
  $("vision-model").addEventListener("change", applyVisionModel);
  $("settings-form").addEventListener("submit", startConversion);
  $("log-view").addEventListener("scroll", () => {
    const view = $("log-view");
    state.logFollow = view.scrollHeight - view.scrollTop - view.clientHeight < 24;
  });
  $("stop-conversion").addEventListener("click", stopConversion);
  $("check-environment").addEventListener("click", () => checkEnvironment($("check-environment")));
  $("refresh-environment").addEventListener("click", () => checkEnvironment($("refresh-environment")));
  $("copy-log").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("log-view").textContent);
      showToast("日志已复制", "success");
    } catch (error) {
      showToast(`复制失败：${error.message}`, "error");
    }
  });
  $("select-model-folder").addEventListener("click", chooseModelDirectory);
  $("select-output-folder").addEventListener("click", chooseOutputDirectory);
  $("open-output").addEventListener("click", () => openFolder(state.outputDir || state.lastOutput));
  updateControls("idle");
}

init();
