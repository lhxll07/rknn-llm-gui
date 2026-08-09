const presentation = document.querySelector("#presentation");
const scenes = [...document.querySelectorAll(".scene")];
const playButton = document.querySelector("#play-button");
const replayButton = document.querySelector("#replay-button");
const soundButton = document.querySelector("#sound-button");
const timeline = document.querySelector("#timeline");
const progress = document.querySelector("#timeline-progress");
const currentTimeLabel = document.querySelector("#current-time");
const sceneCounter = document.querySelector("#scene-counter");
const recordingFrame = document.querySelector(".recording-frame");
const recording = document.querySelector("#recording");
const voiceover = document.querySelector("#voiceover");

const TOTAL_TIME = 60;
let elapsed = 0;
let isPlaying = false;
let frameRequest = null;
let lastFrame = 0;
let activeScene = -1;
let hasVoiceover = false;

function formatTime(seconds) {
  const value = Math.max(0, Math.min(TOTAL_TIME, Math.floor(seconds)));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function sceneForTime(seconds) {
  const index = scenes.findIndex((scene) => {
    const start = Number(scene.dataset.start);
    const end = Number(scene.dataset.end);
    return seconds >= start && seconds < end;
  });
  return index === -1 ? scenes.length - 1 : index;
}

function activateScene(index) {
  if (activeScene === index) return;
  activeScene = index;
  scenes.forEach((scene, sceneIndex) => {
    scene.classList.toggle("is-active", sceneIndex === index);
  });
  sceneCounter.textContent = `${String(index + 1).padStart(2, "0")} / ${String(scenes.length).padStart(2, "0")}`;

  if (index === 4 && recordingFrame.classList.contains("has-video")) {
    recording.currentTime = 0;
    if (isPlaying) recording.play().catch(() => {});
  } else {
    recording.pause();
  }
}

function render() {
  const ratio = elapsed / TOTAL_TIME;
  progress.style.width = `${ratio * 100}%`;
  currentTimeLabel.textContent = formatTime(elapsed);
  timeline.setAttribute("aria-valuenow", String(Math.floor(elapsed)));
  activateScene(sceneForTime(elapsed));
}

function stopFrameLoop() {
  if (frameRequest !== null) {
    cancelAnimationFrame(frameRequest);
    frameRequest = null;
  }
}

function tick(timestamp) {
  if (!isPlaying) return;
  if (!lastFrame) lastFrame = timestamp;
  elapsed += (timestamp - lastFrame) / 1000;
  lastFrame = timestamp;

  if (elapsed >= TOTAL_TIME) {
    elapsed = TOTAL_TIME;
    isPlaying = false;
    presentation.classList.remove("is-playing");
    stopFrameLoop();
    voiceover.pause();
    recording.pause();
  }

  render();
  if (isPlaying) frameRequest = requestAnimationFrame(tick);
}

function play() {
  if (elapsed >= TOTAL_TIME) elapsed = 0;
  isPlaying = true;
  lastFrame = 0;
  presentation.classList.add("is-playing");
  if (hasVoiceover) voiceover.play().catch(() => {});
  if (activeScene === 4 && recordingFrame.classList.contains("has-video")) {
    recording.play().catch(() => {});
  }
  render();
  frameRequest = requestAnimationFrame(tick);
}

function pause() {
  isPlaying = false;
  presentation.classList.remove("is-playing");
  stopFrameLoop();
  voiceover.pause();
  recording.pause();
}

function restart() {
  pause();
  elapsed = 0;
  voiceover.currentTime = 0;
  recording.currentTime = 0;
  render();
  play();
}

function seek(seconds) {
  elapsed = Math.max(0, Math.min(TOTAL_TIME, seconds));
  if (hasVoiceover) voiceover.currentTime = elapsed;
  if (activeScene === 4 && recordingFrame.classList.contains("has-video")) {
    recording.currentTime = 0;
  }
  render();
}

playButton.addEventListener("click", () => {
  if (isPlaying) pause();
  else play();
});

replayButton.addEventListener("click", restart);

timeline.addEventListener("click", (event) => {
  const rect = timeline.getBoundingClientRect();
  seek(((event.clientX - rect.left) / rect.width) * TOTAL_TIME);
});

timeline.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    seek(elapsed - 1);
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    seek(elapsed + 1);
  }
  if (event.key === "Home") {
    event.preventDefault();
    seek(0);
  }
  if (event.key === "End") {
    event.preventDefault();
    seek(TOTAL_TIME);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, button, a")) return;
  if (event.code === "Space") {
    event.preventDefault();
    if (isPlaying) pause();
    else play();
  }
});

soundButton.addEventListener("click", () => {
  if (!hasVoiceover) return;
  voiceover.muted = !voiceover.muted;
  soundButton.classList.toggle("is-muted", voiceover.muted);
  soundButton.textContent = voiceover.muted ? "旁白已静音" : "旁白";
});

recording.addEventListener("canplay", () => {
  recordingFrame.classList.add("has-video");
  if (isPlaying && activeScene === 4) recording.play().catch(() => {});
});

recording.addEventListener("error", () => {
  recordingFrame.classList.remove("has-video");
});

voiceover.addEventListener("canplay", () => {
  hasVoiceover = true;
  soundButton.classList.remove("is-muted");
});

voiceover.addEventListener("error", () => {
  hasVoiceover = false;
  soundButton.classList.add("is-muted");
});

render();
