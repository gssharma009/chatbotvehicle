document.addEventListener("DOMContentLoaded", () => {
    const API_URL = "https://chatbotvehicle-backend.onrender.com/ask";
    const chatContainer = document.getElementById("chat-container");

    function addMessage(content, sender, tts=false) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender;
        msgDiv.innerHTML = content;

        const timeSpan = document.createElement("div");
        timeSpan.className = "timestamp";
        const now = new Date();
        timeSpan.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        msgDiv.appendChild(timeSpan);

        // Optional TTS button for bot messages
        if (sender === "bot") {
            const ttsBtn = document.createElement("span");
            ttsBtn.className = "tts-btn";
            ttsBtn.textContent = "🔊 Speak";
            ttsBtn.onclick = () => speakText(content);
            msgDiv.appendChild(ttsBtn);
        }

        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        if (sender === "bot" && tts) speakText(content);
    }

    function showLoader() {
        const loader = document.createElement("div");
        loader.className = "message bot";
        loader.id = "loader";
        loader.innerHTML = `<span class="loader"><span></span><span></span><span></span></span>`;
        chatContainer.appendChild(loader);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function removeLoader() {
        const loader = document.getElementById("loader");
        if (loader) loader.remove();
    }

    function speakText(text) {
        const lang = document.getElementById("lang").value;
        const utter = new SpeechSynthesisUtterance(text);
        utter.lang = lang;
        speechSynthesis.speak(utter);
    }

    async function askBot() {
        const qElem = document.getElementById("question");
        const question = qElem.value.trim();
        if (!question) return alert("Please enter a question!");

        addMessage(question, "user");
        qElem.value = "";

        showLoader();

        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });

            removeLoader();

            if (!res.ok) {
                addMessage(`❌ API Error: ${res.status}`, "bot");
                return;
            }

            const data = await res.json();
            const answer = data.answer || "No response from backend";

            const ttsEnabled = document.getElementById("tts-checkbox").checked;
            addMessage(answer, "bot", ttsEnabled);

        } catch (err) {
            removeLoader();
            console.error(err);
            addMessage("❌ Network/Error: " + err, "bot");
        }
    }

    function startListening() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert("Your browser does not support Speech Recognition");
            return;
        }

        const lang = document.getElementById("lang").value;
        const recognition = new SpeechRecognition();
        recognition.lang = lang;
        recognition.interimResults = false;

        recognition.start();

        recognition.onresult = function(event) {
            const text = event.results[0][0].transcript;
            document.getElementById("question").value = text;
            askBot();
        };

        recognition.onerror = function(err) {
            console.error("SpeechRecognition error", err);
            alert("Voice recognition error");
        };
    }

    window.askBot = askBot;
    window.startListening = startListening;
    window.speakText = speakText;
});
