// script.js - FINAL PERFECTION: Clean TTS, Hindi/English/Hinglish, Stop Button

const API_URL = 'https://chatbotvehicle-production.up.railway.app/ask';

const chatContainer = document.getElementById("chat-container");
const questionInput = document.getElementById("question");
const langSelect = document.getElementById("lang");
const ttsCheckbox = document.getElementById("tts-checkbox");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");

let currentUtterance = null;

// Translations
const translations = {
  "en-US": {
    title: "Voice + Text Chatbot (Hindi & English)",
    autoSpeak: "Auto Speak Reply",
    placeholder: "Type your question or speak...",
    send: "Send",
    mic: "Speak",
    play: "Play",
    stop: "Stop",
    playing: "Playing",
    speaking: "Speaking"
  },
  "hi-IN": {
    title: "वॉइस + टेक्स्ट चैटबॉट (हिंदी और अंग्रेजी)",
    autoSpeak: "ऑटो बोलें जवाब",
    placeholder: "अपना सवाल लिखें या बोलें...",
    send: "भेजें",
    mic: "बोलें",
    play: "सुनें",
    stop: "रुकें",
    playing: "चल रहा है",
    speaking: "बोल रहा है"
  }
};

function updateUI() {
  const lang = langSelect.value;
  document.querySelector("h2").textContent = translations[lang].title;
  document.querySelector(".tts-label span").textContent = translations[lang].autoSpeak;
  questionInput.placeholder = translations[lang].placeholder;
  document.querySelector(".send-btn").textContent = translations[lang].send;
  micBtn.textContent = translations[lang].mic;
}
langSelect.addEventListener("change", updateUI);
window.addEventListener("load", updateUI);

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;

  // Clean for display — keep nice bullets
  const displayText = text
    .replace(/\\/g, "")
    .replace(/^\*\s*/gm, "• ")
    .replace(/^\d+\.\s*/gm, "• ")
    .trim();

  const p = document.createElement("p");
  p.innerHTML = displayText.replace(/\n/g, "<br>");
  div.appendChild(p);

  const time = document.createElement("div");
  time.className = "timestamp";
  time.textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  div.appendChild(time);

  if (sender === "bot") {
    const playBtn = document.createElement("span");
    playBtn.className = "tts-btn";
    playBtn.textContent = translations[langSelect.value].play;
    playBtn.onclick = () => speak(text);
    div.appendChild(playBtn);

    const stopBtn = document.createElement("span");
    stopBtn.className = "tts-btn stop-btn";
    stopBtn.textContent = translations[langSelect.value].stop;
    stopBtn.onclick = stopSpeaking;
    div.appendChild(stopBtn);
  }

  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // Auto speak if enabled
  if (sender === "bot" && ttsCheckbox.checked) {
    speak(text);
  }
}

function showLoader() {
  const loader = document.createElement("div");
  loader.id = "loader";
  loader.className = "message bot";
  loader.innerHTML = `<div class="loader"><span></span><span></span><span></span></div> Thinking...`;
  chatContainer.appendChild(loader);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.remove();
}

// PERFECT CLEAN SPEECH — NO BULLETS, NO SYMBOLS, NATURAL FLOW
function speak(rawText) {
  if (!('speechSynthesis' in window) || !rawText?.trim()) return;

  stopSpeaking();

  let cleanText = rawText
    .replace(/\\/g, " ")
    .replace(/^[\*\•\-\–\—\d]+\.?\s*/gm, " ")           // * • - 1. २.
    .replace(/^[\u0966-\u096F]+\.?\s*/gm, " ")          // Hindi digits १. २.
    .replace(/^[a-zA-Z]\)\s*/gm, " ")                   // a) b)
    .replace(/^[\(\[\{]\s*[\da-zA-Z\u0966-\u096F]+\s*[\)\]\}]\s*/gm, " ")
    .replace(/[^\u0900-\u097F\w\s\.\,\!\?\;\:\-\(\)]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleanText) return;

  const utter = new SpeechSynthesisUtterance(cleanText);
  utter.lang = langSelect.value;
  utter.rate = 0.9;
  utter.pitch = 1;
  utter.volume = 1;

  utter.onstart = () => {
    document.querySelectorAll(".message.bot").forEach(msg => {
      const playBtn = msg.querySelector(".tts-btn:not(.stop-btn)");
      const stopBtn = msg.querySelector(".stop-btn");
      if (playBtn) playBtn.textContent = translations[langSelect.value].playing || translations[langSelect.value].speaking;
      if (stopBtn) stopBtn.classList.add("show");
    });
  };

  utter.onend = utter.onerror = () => {
    currentUtterance = null;
    document.querySelectorAll(".tts-btn:not(.stop-btn)").forEach(btn => {
      btn.textContent = translations[langSelect.value].play;
    });
    document.querySelectorAll(".stop-btn").forEach(btn => btn.classList.remove("show"));
  };

  currentUtterance = utter;
  speechSynthesis.speak(utter);
}

function stopSpeaking() {
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
    currentUtterance = null;
  }
  document.querySelectorAll(".tts-btn:not(.stop-btn)").forEach(btn => {
    btn.textContent = translations[langSelect.value].play;
  });
  document.querySelectorAll(".stop-btn").forEach(btn => btn.classList.remove("show"));
}

async function askBot() {
  let question = questionInput.value.trim();
  if (!question) return;

  addMessage(question, "user");
  questionInput.value = "";
  showLoader();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, lang: langSelect.value })
    });

    removeLoader();
    if (!res.ok) throw new Error("Server error");

    const data = await res.json();
    const answer = data?.answer || "No response.";
    addMessage(answer, "bot");

  } catch (err) {
    removeLoader();
    addMessage("Network error. Please try again.", "bot");
  }
}

function startListening() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    alert("Voice input not supported in this browser");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = langSelect.value;
  recognition.continuous = false;
  recognition.interimResults = false;

  micBtn.textContent = "...";
  micBtn.disabled = true;

  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    questionInput.value = transcript;
    askBot();
  };

  recognition.onerror = () => {
    alert("Voice recognition error");
    micBtn.textContent = translations[langSelect.value].mic;
    micBtn.disabled = false;
  };

  recognition.onend = () => {
    micBtn.textContent = translations[langSelect.value].mic;
    micBtn.disabled = false;
  };

  recognition.start();
}

// Event Listeners
sendBtn.addEventListener("click", askBot);
micBtn.addEventListener("click", startListening);
questionInput.addEventListener("keypress", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askBot();
  }
});