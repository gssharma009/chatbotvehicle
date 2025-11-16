const BACKEND_URL = "YOUR_RENDER_BACKEND_URL/ask";  // ADD /ask

async function sendText() {
    const text = document.getElementById("textInput").value;

    const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: text })
    });

    const data = await res.json();
    document.getElementById("answer").innerText = data.answer || data.error;
}
