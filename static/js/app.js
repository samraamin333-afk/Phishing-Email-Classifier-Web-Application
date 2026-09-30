/**
 * Client-side script for Phishing Email Classifier Web Application
 * Handles asynchronous scan submissions, preset loading, and interactive gauge animations.
 */

document.addEventListener("DOMContentLoaded", () => {
  const scanForm = document.getElementById("scanForm");
  const subjectInput = document.getElementById("emailSubject");
  const bodyInput = document.getElementById("emailBody");
  const scanBtn = document.getElementById("scanBtn");
  const clearBtn = document.getElementById("clearBtn");
  const resultsContainer = document.getElementById("resultsContainer");
  const placeholderContainer = document.getElementById("placeholderContainer");
  const presetButtons = document.querySelectorAll(".preset-btn");

  // Handle Preset Loading
  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const subject = btn.getAttribute("data-subject") || "";
      const body = btn.getAttribute("data-body") || "";
      
      subjectInput.value = subject;
      bodyInput.value = body;

      // Highlight active button
      presetButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      // Auto trigger scan
      submitScan();
    });
  });

  // Handle Clear
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      subjectInput.value = "";
      bodyInput.value = "";
      presetButtons.forEach(b => b.classList.remove("active"));
      if (resultsContainer) resultsContainer.style.display = "none";
      if (placeholderContainer) placeholderContainer.style.display = "flex";
    });
  }

  // Handle Form Submission
  if (scanForm) {
    scanForm.addEventListener("submit", (e) => {
      e.preventDefault();
      submitScan();
    });
  }

  async function submitScan() {
    const subject = subjectInput.value.trim();
    const body = bodyInput.value.trim();

    if (!subject && !body) {
      alert("Please enter email body or subject to scan.");
      return;
    }

    // Set loading state
    const originalBtnText = scanBtn.innerHTML;
    scanBtn.innerHTML = `<span>Scanning Threat Vectors...</span>`;
    scanBtn.disabled = true;

    try {
      const response = await fetch("/scan", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ subject, body }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      console.error("Scan failed:", err);
      alert("Error scanning email: " + err.message);
    } finally {
      scanBtn.innerHTML = originalBtnText;
      scanBtn.disabled = false;
    }
  }

  function renderResults(res) {
    if (placeholderContainer) placeholderContainer.style.display = "none";
    if (resultsContainer) resultsContainer.style.display = "block";

    const isPhish = res.prediction === "Phishing";
    const verdictBanner = document.getElementById("verdictBanner");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictSubtitle = document.getElementById("verdictSubtitle");
    const verdictBadge = document.getElementById("verdictBadge");

    // Update Verdict Banner
    if (isPhish) {
      verdictBanner.className = "verdict-banner phish";
      verdictTitle.textContent = "PHISHING EMAIL DETECTED";
      verdictSubtitle.textContent = "High probability malicious communication flagged by Naive Bayes.";
      verdictBadge.className = "badge-tag phish";
      verdictBadge.textContent = "THREAT LEVEL: " + res.severity;
    } else {
      verdictBanner.className = "verdict-banner safe";
      verdictTitle.textContent = "AUTHENTIC / LEGITIMATE EMAIL";
      verdictSubtitle.textContent = "Standard business communication. No malicious heuristic cues detected.";
      verdictBadge.className = "badge-tag safe";
      verdictBadge.textContent = "VERIFIED SAFE";
    }

    // Update Threat Score & Meter
    const scoreVal = document.getElementById("threatScoreVal");
    const meterFill = document.getElementById("meterFill");
    const probPhish = document.getElementById("probPhish");
    const probHam = document.getElementById("probHam");

    scoreVal.textContent = res.threat_score + "%";
    meterFill.style.width = res.threat_score + "%";
    meterFill.className = "meter-fill " + (isPhish ? "phish" : "safe");

    probPhish.textContent = res.probabilities.phishing + "%";
    probHam.textContent = res.probabilities.legitimate + "%";

    // Update Heuristics
    const h = res.heuristics || {};
    document.getElementById("hUrlCount").textContent = h.url_count ?? 0;
    
    const ipElem = document.getElementById("hIpUrl");
    ipElem.textContent = h.has_ip_url ? "DETECTED" : "None";
    ipElem.className = "heuristic-val " + (h.has_ip_url ? "alert" : "");

    const kwElem = document.getElementById("hUrgentKw");
    kwElem.textContent = h.urgent_keywords_found ? h.urgent_keywords_found.length : 0;
    kwElem.className = "heuristic-val " + (h.urgent_keywords_found && h.urgent_keywords_found.length > 0 ? "alert" : "");

    document.getElementById("hCapsRatio").textContent = (Math.round((h.caps_ratio || 0) * 100)) + "%";

    // Update Explainability Tokens
    const tokenCloud = document.getElementById("tokenCloud");
    tokenCloud.innerHTML = "";
    const signals = res.explainability ? res.explainability.token_signals : [];

    if (signals && signals.length > 0) {
      signals.forEach(sig => {
        const pill = document.createElement("span");
        pill.className = "token-pill " + (sig.tendency === "phishing" ? "phish" : (sig.tendency === "legitimate" ? "legitimate" : "neutral"));
        pill.textContent = `${sig.word} (${sig.tendency === "phishing" ? "+" : "-"}${sig.impact})`;
        tokenCloud.appendChild(pill);
      });
    } else {
      tokenCloud.innerHTML = `<span style="color:#64748b; font-size:13px;">No dominant keyword signals identified.</span>`;
    }

    // Update Recommendations
    const recsList = document.getElementById("recsList");
    recsList.innerHTML = "";
    if (res.recommendations && res.recommendations.length > 0) {
      res.recommendations.forEach(r => {
        const li = document.createElement("li");
        li.textContent = r;
        recsList.appendChild(li);
      });
    }
  }
});
