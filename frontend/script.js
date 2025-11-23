// script.js - FINAL VERSION (Works with your Render backend)
const API_URL = "https://chatbotvehicle-backend.onrender.com/ask";  // ← Your live URL

const chatContainer = document.getElementById("chat-container");
const questionInput = document.getElementById("question");
const langSelect = document.getElementById("lang");
const ttsCheckbox = document.getElementById("tts-checkbox");
const micBtn = document.getElementById("mic-btn");

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

    // Speak button for bot messages
    if (sender === "bot") {
        const speaker = document.createElement("span");
        speaker.className = "tts-btn";
        speaker.textContent = "Speak";
        speaker.onclick = () => speak(text);
        div.appendChild(speaker);
    }

    chatContainer.appendChild(div);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
}

function showLoader() {
    const loader = document.createElement("div");
    loader.id = "loader";
    loader.className = "message bot";
    loader.innerHTML = `<div class="loader"><span></span><span></span><span></span></div> Thinking...`;
    chatContainer.appendChild(loader);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
}

function removeLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.remove();
}

function speak(text) {
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = langSelect.value;  // hi-IN or en-US
        utter.rate = 0.9;
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
            body: JSON.stringify({ question })
        });

        removeLoader();

        if (!res.ok) {
            addMessage("Server error. Try again.", "bot");
            return;
        }

        const data = await res.json();

        // This handles your current backend format: { "results": { "answer": "..." } }
        let answer = "";
        if (data.results && data.results.answer) {
            answer = data.results.answer;
        } else if (data.answer) {
            answer = data.answer;
        } else {
            answer = "No answer received.";
        }

        addMessage(answer, "bot");

        // Auto-speak if checkbox is on
        if (ttsCheckbox.checked) {
            speak(answer);
        }

    } catch (err) {
        removeLoader();
        console.error(err);
        addMessage("Network error. Check connection.", "bot");
    }
}

// Voice Input (Chrome/Edge)
function startListening() {
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
        alert("Voice not supported. Use Chrome/Edge.");
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = langSelect.value;
    recognition.interimResults = false;

    micBtn.textContent = "Listening...";
    micBtn.style.background = "#d32f2f";

    recognition.start();

    recognition.onresult = (e) => {
        questionInput.value = e.results[0][0].transcript;
        askBot();
    };

    recognition.onerror = recognition.onend = () => {
        micBtn.textContent = "Speak";
        micBtn.style.background = "#ff5722";
    };
}

// Send on Enter (Shift+Enter for new line)
questionInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askBot();
    }
});