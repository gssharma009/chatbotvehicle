const API_URL = "https://YOUR_BACKEND_URL.onrender.com/ask";

// --------------------
// Send Text Query
// --------------------
async function askBot() {
    const q = document.getElementById("question").value.trim();
    if (!q) return alert("Please enter a question!");

    const responseDiv = document.getElementById("response");
    responseDiv.innerHTML = "⏳ Thinking...";

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q })
        });

        const data = await res.json();
        responseDiv.innerHTML = data.answer || "No response from backend";
    } catch (err) {
        responseDiv.innerHTML = "❌ Error: " + err;
    }
}

// --------------------
// Voice Input (STT)
// --------------------
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

    recognition.onerror = function() {
        alert("Voice recognition error");
    };
}
