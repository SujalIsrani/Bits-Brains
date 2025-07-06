let keyCount = 0;
let keyTimes = [];
let startTime = null;

document.addEventListener("DOMContentLoaded", () => {
    const box = document.getElementById("typingBox");

    box.addEventListener("keydown", (e) => {
        const time = new Date().getTime();
        keyCount++;
        if (!startTime) startTime = time;
        keyTimes.push(time);
    });
});

function submitMetrics() {
    if (keyTimes.length < 2) return alert("Type more for accurate metrics!");

    const totalTimeSec = (keyTimes[keyTimes.length - 1] - startTime) / 1000;

    // ✅ Typing speed in KEYS PER MINUTE
    const typing_speed = (keyCount / totalTimeSec) * 60;

    // Simulated values (replace with real if available)
    const nqScore = (Math.random() * (1.0 - 0.0) + 0.1).toFixed(4);  // Keeping it in same realistic range
    const sTap = (Math.random() * (200 - 140) + 140).toFixed(2);
    const afTap = (Math.random() * (140 - 80) + 80).toFixed(2);

    const metrics = {
        username: username,
        nqScore: nqScore,
        typing_speed: typing_speed.toFixed(2),  // ✅ Now sending KPM
        sTap: sTap,
        afTap: afTap
    };

    fetch("/submit", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metrics)
    })
    .then(response => response.json())
    .then(result => {
        if (result.status === "success") {
            document.getElementById("results").innerHTML = `
                🧠 NeuroQWERTY Score: <b>${nqScore}</b><br>
                ⌨️ Typing Speed: <b>${typing_speed.toFixed(2)} keys/min</b><br>
                👉 Single Tap (sTap): <b>${sTap}</b><br>
                🧩 Alternate Tap (afTap): <b>${afTap}</b><br><br>
                🔍 <b>Predicted PD Probability:</b> ${result.probability}%<br>
                ${result.prediction === 1 ? "🧩 Likely PD Detected" : "✅ No PD Detected"}
            `;
        } else {
            alert("Prediction failed.");
        }
    });
}
