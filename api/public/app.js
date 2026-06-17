document.addEventListener("DOMContentLoaded", () => {
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = submitBtn.querySelector(".btn-text");
    const spinner = submitBtn.querySelector(".loading-spinner");
    const chatMessages = document.getElementById("chat-messages");
    const welcomeCard = document.getElementById("welcome-card");
    const modelSelect = document.getElementById("model-select");
    const statusBadge = document.getElementById("status-badge");
    const statusText = document.getElementById("status-text");

    // Check backend connection status and load DB stats
    async function checkStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            
            if (data.connected) {
                statusBadge.className = "status-badge connected";
                statusText.textContent = `Connected: ${data.indexed_chunks} chunks`;
            } else {
                statusBadge.className = "status-badge disconnected";
                statusText.textContent = "Disconnected (No DB)";
            }
        } catch (e) {
            statusBadge.className = "status-badge disconnected";
            statusText.textContent = "Offline (Server down)";
            console.error("Status check failed:", e);
        }
    }

    // Call status immediately
    checkStatus();

    // Bind Quick Start buttons
    document.querySelectorAll(".qs-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.dataset.query;
            queryInput.value = query;
            queryForm.dispatchEvent(new Event("submit"));
        });
    });

    // Handle Form Submission
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        // Disable input during request
        queryInput.disabled = true;
        submitBtn.disabled = true;
        btnText.classList.add("hidden");
        spinner.classList.remove("hidden");

        // Remove welcome card on first submission
        if (welcomeCard) {
            welcomeCard.remove();
        }

        // 1. Add User Message
        appendMessage("user", query);
        queryInput.value = "";

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            // 2. Fetch answer from API
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    model: modelSelect.value,
                    k: 4
                })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Query failed");
            }

            const data = await response.json();

            // 3. Add Assistant Message
            appendMessage("assistant", data.answer, data.references);

        } catch (err) {
            appendMessage("assistant", `Error executing query: ${err.message}. Please verify your API key and database connection.`);
        } finally {
            // Re-enable input controls
            queryInput.disabled = false;
            submitBtn.disabled = false;
            btnText.classList.remove("hidden");
            spinner.classList.add("hidden");
            queryInput.focus();
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    });

    // Helper: Append a message bubble to the chat workspace
    function appendMessage(sender, text, references = []) {
        const messageDiv = document.createElement("div");
        messageDiv.className = `message ${sender}`;

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "message-bubble";
        bubbleDiv.textContent = text;
        messageDiv.appendChild(bubbleDiv);

        // Render Collapsible References if present (only for assistant messages)
        if (sender === "assistant" && references && references.length > 0) {
            const refContainer = document.createElement("div");
            refContainer.className = "references-container";

            const toggleHeader = document.createElement("div");
            toggleHeader.className = "ref-toggle-header";
            toggleHeader.textContent = `Source References (${references.length})`;
            
            const refList = document.createElement("div");
            refList.className = "references-list";

            references.forEach((ref) => {
                const refCard = document.createElement("div");
                refCard.className = "ref-card";

                const refTitle = document.createElement("div");
                refTitle.className = "ref-card-title";
                refTitle.textContent = ref.title || `Document ${ref.doc_id}`;

                const refMeta = document.createElement("div");
                refMeta.className = "ref-card-meta";

                const sourceSpan = document.createElement("span");
                sourceSpan.textContent = `File: ${ref.source} (Chunk ${ref.chunk_index})`;
                refMeta.appendChild(sourceSpan);

                if (ref.authority) {
                    const authSpan = document.createElement("span");
                    authSpan.textContent = `| Authority: ${ref.authority}`;
                    refMeta.appendChild(authSpan);
                }

                if (ref.collections) {
                    const collSpan = document.createElement("span");
                    collSpan.textContent = `| Category: ${ref.collections}`;
                    refMeta.appendChild(collSpan);
                }

                refCard.appendChild(refTitle);
                refCard.appendChild(refMeta);

                if (ref.link) {
                    const refLink = document.createElement("a");
                    refLink.href = ref.link;
                    refLink.target = "_blank";
                    refLink.className = "ref-card-link";
                    refLink.textContent = "View Official Document ↗";
                    refCard.appendChild(refLink);
                }

                refList.appendChild(refCard);
            });

            // Toggle logic
            toggleHeader.addEventListener("click", () => {
                const isOpen = refList.classList.contains("show");
                if (isOpen) {
                    refList.classList.remove("show");
                    toggleHeader.classList.remove("active");
                } else {
                    refList.classList.add("show");
                    toggleHeader.classList.add("active");
                }
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });

            refContainer.appendChild(toggleHeader);
            refContainer.appendChild(refList);
            bubbleDiv.appendChild(refContainer);
        }

        const metaDiv = document.createElement("div");
        metaDiv.className = "message-meta";
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        metaDiv.textContent = sender === "user" ? `You • ${time}` : `AI • ${time}`;
        messageDiv.appendChild(metaDiv);

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
