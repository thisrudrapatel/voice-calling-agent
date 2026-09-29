// Browser "phone": streams mic audio to the server over a WebSocket and plays
// the agent's voice back as it arrives.
const $ = (id) => document.getElementById(id);
const phone = $("phone");
const RATE = 16000;

let ws, ctx, micStream, micNode, sourceNode;
let playhead = 0;          // AudioContext time where the next agent chunk starts
let playing = [];          // scheduled AudioBufferSourceNodes, for barge-in
let muted = false;
let timer, startedAt;

fetch("/api/status").then((r) => r.json()).then((s) => {
  $("name").textContent = `${s.agent} · ${s.company}`;
  $("avatar").textContent = s.agent[0];
  $("meta").textContent = `LLM: ${s.llm} · STT: ${s.stt} · TTS: ${s.tts} · ${s.chunks} knowledge chunks`;
}).catch(() => {});

function setState(state) {
  phone.classList.remove("listening", "thinking", "speaking");
  if (state) phone.classList.add(state);
  const labels = { listening: "Listening…", thinking: "Thinking…", speaking: "Speaking…" };
  if (labels[state]) $("status").textContent = `${labels[state]}  ${elapsed()}`;
}

function elapsed() {
  if (!startedAt) return "";
  const s = Math.floor((Date.now() - startedAt) / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function addMessage(role, text) {
  const log = $("log");
  const last = log.lastElementChild;
  // Agent replies arrive sentence by sentence; keep them in one bubble.
  if (last && last.classList.contains(role) && role === "agent") {
    last.textContent += " " + text;
  } else {
    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.textContent = text;
    log.appendChild(div);
  }
  log.scrollTop = log.scrollHeight;
}

function playPcm(buf) {
  const pcm = new Int16Array(buf);
  const audio = ctx.createBuffer(1, pcm.length, RATE);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = audio;
  src.connect(ctx.destination);
  playhead = Math.max(playhead, ctx.currentTime + 0.05);
  src.start(playhead);
  playhead += audio.duration;
  playing.push(src);
  src.onended = () => { playing = playing.filter((s) => s !== src); };
}

function stopPlayback() {
  playing.forEach((s) => { try { s.stop(); } catch (_) {} });
  playing = [];
  playhead = 0;
}

async function startCall() {
  $("log").innerHTML = "";
  $("status").textContent = "Connecting…";
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (e) {
    $("status").textContent = "Microphone permission is needed to call.";
    return;
  }
  ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/static/mic-worklet.js");
  sourceNode = ctx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(ctx, "mic-processor");
  sourceNode.connect(micNode);
  // Some browsers only run a worklet that is pulled by the output graph.
  const sink = ctx.createGain();
  sink.gain.value = 0;
  micNode.connect(sink).connect(ctx.destination);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    phone.classList.add("in-call");
    startedAt = Date.now();
    timer = setInterval(() => setState([...phone.classList].find((c) => ["listening", "thinking", "speaking"].includes(c))), 1000);
    micNode.port.onmessage = (e) => {
      if (!muted && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    };
  };
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") return playPcm(e.data);
    const ev = JSON.parse(e.data);
    if (ev.type === "state") setState(ev.state);
    else if (ev.type === "transcript") addMessage(ev.role, ev.text);
    else if (ev.type === "clear") stopPlayback();
    else if (ev.type === "hangup") endCall("Call ended by agent");
  };
  ws.onclose = () => endCall();
  ws.onerror = () => endCall("Connection failed — is the server running?");
}

function endCall(reason) {
  if (!ws && !ctx) return;
  if (ws) {
    ws.onclose = ws.onerror = null;
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "hangup" }));
      ws.close();
    }
  }
  ws = null;
  clearInterval(timer);
  // Let the goodbye finish playing before tearing audio down.
  const tail = ctx ? Math.max(0, playhead - ctx.currentTime) : 0;
  const c = ctx, m = micStream;
  setTimeout(() => {
    if (m) m.getTracks().forEach((t) => t.stop());
    if (c && c.state !== "closed") c.close();
  }, tail * 1000);
  ctx = null; micStream = null; playing = []; playhead = 0;
  phone.classList.remove("in-call");
  setState(null);
  $("status").textContent = `${reason || "Call ended"}${startedAt ? " · " + elapsed() : ""}`;
  startedAt = null;
  muted = false;
  $("mute").textContent = "Mute";
}

$("call").onclick = startCall;
$("hangup").onclick = () => endCall();
$("mute").onclick = () => {
  muted = !muted;
  $("mute").textContent = muted ? "Unmute" : "Mute";
};
