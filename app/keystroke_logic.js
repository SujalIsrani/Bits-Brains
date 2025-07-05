
let keystrokeData = [];
let keyDownTimestamps = {};
let lastKeyUpTimestamp = null;
let startTime = null;
let endTime = null;
let charCount = 0;

let sTapKey = " ";
let sTapStart = null;
let sTapCount = 0;

let afTapKeys = ["S", "L"];
let afTapStart = null;
let afTapSequence = [];

function initKeystrokeCapture() {
    if (document.getElementById("keystroke-input")) return;

    const textarea = document.createElement("textarea");
    textarea.id = "keystroke-input";
    textarea.placeholder = "Start typing here...";
    textarea.style.width = "100%";
    textarea.style.height = "150px";
    textarea.style.padding = "10px";
    textarea.style.fontSize = "16px";
    textarea.style.borderRadius = "5px";
    textarea.style.border = "1px solid #ccc";
    textarea.style.backgroundColor = "#222";
    textarea.style.color = "#fff";

    document.body.appendChild(textarea);

    textarea.addEventListener("keydown", function (e) {
        const key = e.key.toUpperCase();
        const now = performance.now();

        if (!startTime) startTime = now;
        keyDownTimestamps[key] = now;

        if (key.length === 1) charCount++;

        if (key === sTapKey) {
            if (!sTapStart) sTapStart = now;
            sTapCount++;
        }

        if (afTapKeys.includes(key)) {
            if (!afTapStart) afTapStart = now;
            afTapSequence.push({ key, time: now });
        }
    });

    textarea.addEventListener("keyup", function (e) {
        const key = e.key.toUpperCase();
        const now = performance.now();

        if (keyDownTimestamps[key]) {
            const pressTime = keyDownTimestamps[key];
            const holdTime = now - pressTime;
            const flightTime = lastKeyUpTimestamp ? pressTime - lastKeyUpTimestamp : null;

            keystrokeData.push({
                key: key,
                press_time: pressTime,
                release_time: now,
                hold_time: holdTime,
                flight_time: flightTime
            });

            lastKeyUpTimestamp = now;
            delete keyDownTimestamps[key];
        }
    });
}

function stopTestAndReturnResults() {
    endTime = performance.now();
    const durationSec = Math.max((endTime - startTime) / 1000, 1); // Avoid division by zero
    const typingSpeed = charCount / durationSec;

    // sTap calculation
    let sTapSpeed = 0;
    if (sTapCount > 1 && sTapStart) {
        const sTapDuration = Math.max((endTime - sTapStart) / 1000, 1);
        sTapSpeed = sTapCount / sTapDuration;
    }

    // afTap calculation
    let afTapSpeed = 0;
    if (afTapSequence.length > 1 && afTapStart) {
        let alternations = 0;
        for (let i = 1; i < afTapSequence.length; i++) {
            if (afTapSequence[i].key !== afTapSequence[i - 1].key) alternations++;
        }
        const afTapDuration = Math.max((endTime - afTapStart) / 1000, 1);
        afTapSpeed = alternations / afTapDuration;
    }

    // nqScore calculation
    const holdTimes = keystrokeData.map(k => k.hold_time).filter(Boolean);
    const flightTimes = keystrokeData.map(k => k.flight_time).filter(f => f !== null && !isNaN(f));
    const holdSD = holdTimes.length > 1 ? standardDeviation(holdTimes) : 0;
    const flightSD = flightTimes.length > 1 ? standardDeviation(flightTimes) : 0;
    const nqScore = (holdSD + flightSD) / 2;

    return {
        nqScore: Number(nqScore.toFixed(4)) || 0,
        typing_speed: Number(typingSpeed.toFixed(2)) || 0,
        sTap: Number(sTapSpeed.toFixed(2)) || 0,
        afTap: Number(afTapSpeed.toFixed(2)) || 0,
        raw_data: keystrokeData
    };
}

function standardDeviation(arr) {
    const n = arr.length;
    if (n === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    const variance = arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    return Math.sqrt(variance);
}

initKeystrokeCapture();

