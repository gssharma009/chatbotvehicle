// script.js - FINAL: Full Hindi + English Support, Mobile Ready
const API_URL = 'https://chatbotvehicle-production.up.railway.app/ask';

const chatContainer = document.getElementById("chat-container");
const questionInput = document.getElementById("question");
const langSelect = document.getElementById("lang");
const ttsCheckbox = document.getElementById("tts-checkbox");
const micBtn = document.getElementById("mic-btn");

// Dynamic UI translation
const translations = {
  "en-US": {
    title: "Voice + Text Chatbot (Hindi & English)",
    autoSpeak: "Auto Speak Reply",
    placeholder: "Type your question or speak...",
    send: "Send",
    mic: "Speak"
  Speak"
  },
  "hi-IN": {
    title: "वॉइस + टेक्स्ट चैटबॉट (हिंदी और अंग्रेजी)",
    autoSpeak: "ऑटो बोलें जवाब",
    placeholder: "अपना सवाल लिखें या बोलें...",
    send: "भेजें",
    mic: "बोलें"
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

// Run on load + language change
langSelect.addEventListener("change", updateUI);
window.addEventListener("load", updateUI);

function addMessage(text, sender) {
  const div = document.createElement("div");
  div.className = `message ${sender}`;

  const p = document.createElement("p");
  p.innerHTML = text.replace(/\n/g, "<br>");
  div.appendChild(p);

  const time = document.createElement("div");
  time.className = "timestamp";
  time.textContent = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  div.appendChild(time);

  if (sender === "bot") {
    const speaker = document.createElement("span");
    speaker.className = "tts-btn";
    speaker.textContent = "Speak";
    speaker.onclick = () => speak(text);
    div.appendChild(speaker);
  }

  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
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

function speak(text) {
  if ('speechSynthesis' in window) {
    speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = langSelect.value;
    utter.rate = 0.9;
    utter.pitch = 1;
    speechSynthesis.speak(utter);
  }
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
      body: JSON.stringify({
        question,
        lang: langSelect.value  // ← This tells backend which language to reply in
      })
    });

    removeLoader();

    if (!res.ok) {
      addMessage("Server error. Please try again.", "bot");
      return;
    }

    const data = await res.json();
    const answer = data?.results?.answer || data?.answer || "No response.";

    addMessage(answer, "bot");

    if (ttsCheckbox.checked) {
      speak(answer);
    }
  } catch (err) {
    removeLoader();
    addMessage("Network error. Check internet.", "bot");
  }
}

// Voice Input — Works on Android + iPhone
function startListening() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    alert(translations[langSelect.value].mic + " not supported in this browser.");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = langSelect.value;
  recognition.continuous = false;
  recognition.interimResults = false;

  micBtn.textContent = "...";
  micBtn.disabled = true;

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    questionInput.value = transcript;
    askBot();
  };

  recognition.onerror = (event) => {
    console.error(event.error);
    alert("Voice error: " + event.error);
  };

  recognition.onend = () => {
    micBtn.textContent = translations[langSelect.value].mic;
    micBtn.disabled = false;
  };

  recognition.start();
}

// Enter to send
questionInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askBot();
  }
});