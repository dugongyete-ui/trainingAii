---
name: Dzeck WebUI runtime
description: Replit preview behavior and CPU constraints for the Streamlit interface.
---

The WebUI should render its chat shell before loading a large checkpoint; loading
the model during module execution makes the preview appear blank while the
checkpoint is being read. Load the selected model only after the first prompt.

**Why:** The default Transformers foundation is large for a CPU-only workspace,
and Streamlit's websocket must pass through the Replit preview proxy.

**How to apply:** Keep the web workflow on port 5000, launch Streamlit with
CORS and XSRF protection disabled for the private preview proxy, use full
precision on CPU, and keep CPU generation defaults modest.