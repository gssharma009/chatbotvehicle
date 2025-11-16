const API_URL = "https://chatbotvehicle-backend.onrender.com/ask";
const chatWindow = document.getElementById("chat-window");

// --------------------
// Add message to chat window
// --------------------
function addMessage(content, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message " + sender;
    msgDiv.innerHTML = content;

    const timeSpan = document.createElement("div");
    timeSpan.className = "timestamp";
    const now = new Date();
    timeSpan.textContent = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    msgDiv.appendChild(timeSpan);

    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// --------------------
// Show loader animation
// --------------------
function showLoader() {
    const loader = document.createElement("div");
    loader.className = "message bot";
    loader.id = "loader";

    loader.innerHTML = `<span class="loader"></span><span class="loader"></span><span class="loader"></span>`;
    chatWindow.appendChild(loader);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// --------------------
// Remove loader
// --------------------
function removeLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.remove();
}

// --------------------
// Send Text Query
// --------------------
async function askBot() {
    const q = document.getElementById("question").value.trim();
    if (!q) return alert("Please enter a question!");

    addMessage(q, "user");
    document.getElementById("question").value = "";

    showLoader();

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q })
        });

        const
