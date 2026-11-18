
(() => {
  // Prevent the extension from running more than once.
  if (window.neuroBotExtensionLoaded) return;
  window.neuroBotExtensionLoaded = true;

  // Create the right-side classification panel.
  const panel = document.createElement("div");
  panel.id = "neurobot-classification-panel";

  panel.innerHTML = `
    <div class="nb-header">
      <h3>NeuroBot Classification</h3>
      <span class="nb-status">Active</span>
    </div>

    <div class="nb-content">
      <p class="nb-subtitle">Latest student message</p>
      <div id="nb-student-message">
        Waiting for a student message...
      </div>

      <p class="nb-subtitle">Detected category</p>
      <div id="nb-category">Not detected yet</div>

      <p class="nb-subtitle">Reason</p>
      <div id="nb-reason">Waiting for NeuroBot's response...</div>
    </div>
  `;

  document.body.appendChild(panel);

  // Read the latest student message.
  function getLatestStudentMessage() {
    const messages = document.querySelectorAll(
      '[data-testid="user"]'
    );

    if (!messages.length) return null;

    return messages[messages.length - 1].innerText.trim();
  }

  // Read the classification from NeuroBot's response.
  function getClassification() {
    const responses = document.querySelectorAll(
      '[data-testid="bot"]'
    );

    if (!responses.length) return null;

    const latestResponse =
      responses[responses.length - 1].innerText;

    const categoryMatch = latestResponse.match(
      /Category\s*:\s*(Admission Enquiry|Course Information|Payment Issue|Certificate Issue|Appreciation|Complaint|Other)/i
    );

    const reasonMatch = latestResponse.match(
      /Reason\s*:\s*([\s\S]*?)(?:\n|$)/i
    );

    if (!categoryMatch) return null;

    return {
      category: categoryMatch[1],
      reason: reasonMatch
        ? reasonMatch[1].trim()
        : "Classification provided by NeuroBot."
    };
  }

  // Update the panel.
  function updatePanel() {
    const studentMessage = getLatestStudentMessage();
    const classification = getClassification();

    if (studentMessage) {
      document.getElementById(
        "nb-student-message"
      ).innerText = studentMessage;
    }

    if (classification) {
      document.getElementById(
        "nb-category"
      ).innerText = classification.category;

      document.getElementById(
        "nb-reason"
      ).innerText = classification.reason;
    }
  }

  // Watch for new chat messages.
  const observer = new MutationObserver(() => {
    updatePanel();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    characterData: true
  });

  updatePanel();
})();
