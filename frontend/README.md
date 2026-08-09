# Frontend

This folder now contains a working vanilla JS frontend for the Data Analyst Agent.

Current stack:
- Plain HTML, CSS, and ES modules
- No frontend build step required
- Consumes the local JSON bridge in `backend/api_server.py`

Run it:
- `python run_api.py`
- Open `http://127.0.0.1:8787`

Files:
- `index.html` - app entry point
- `src/main.js` - application shell and state management
- `src/api.js` - frontend API adapter
- `src/model.js` - investigation normalization and UI mapping
- `src/styles.css` - visual system and responsive layout

What it shows:
- Home and new investigation flows
- Dataset catalog
- Investigation workspace
- Answer, findings, evidence, confidence, and reports
- Guided and collaborative control surfaces
