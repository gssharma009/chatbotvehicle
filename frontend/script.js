document.addEventListener("DOMContentLoaded", () => {
    const API_URL = "https://chatbotvehicle-backend.onrender.com/ask";
    const chatWindow = document.getElementById("chat-window");

    function addMessage(content, sender) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message " + sender;
        msgDiv.innerHTML = content;

        const timeSpan = document.createElement("div");
        timeSpan.className = "timestamp";
        const now = new Date();
        timeSpan.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        msgDiv.appendChild(timeSpan);

        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function showLoader() {
        const loader = document.createElement("div");
        loader.className = "message bot";
        loader.id = "loader";
        loader.innerHTML = `<span class="loader"><span></span><span></span><span></span></span>`;
        chatWindow.appendChild(loader);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function removeLoader() {
        const loader = document.getElementById("loader");
        if (loader) loader.remove();
    }

    async function askBot() {
        const qElem = document.getElementById("question");
        const q = qElem.value.trim();
        if (!q) return alert("Please enter a question!");

        addMessage(q, "user");
        qElem.value = "";

        showLoader();

        try {
            const res = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: q })
            });

            if (!res.ok) {
                removeLoader();
                addMessage(`❌ API Error: ${res.status}`, "bot");
                return;
            }

            const data = await res.json();
            removeLoader();

            const answer = data.answer || "No response from backend";
            addMessage(answer, "bot");

            const langSelect = document.getElementById("lang").value;
            const utter = new SpeechSynthesisUtterance(answer);
            utter.lang = langSelect;
            speechSynthesis.speak(utter);

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
});
