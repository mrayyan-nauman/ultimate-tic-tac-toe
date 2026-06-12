ns

## Your current structure
```
project/
├── venv/
├── backend/
│   ├── app.py
│   └── ai.py
└── tictactoe/        ← React frontend
```

## Step 1 — Run the backend (Terminal 1)
```bash
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install flask flask-cors      # if not already installed
cd backend
python app.py                     # should start on http://localhost:5000
```
Verify it's up: visit `http://localhost:5000/api/move` in browser (will show method-not-allowed for GET — that's fine, it means Flask is running).

## Step 2 — Run the frontend (Terminal 2, new window)
```bash
cd tictactoe
npm install                       # first time only
npm run dev                       # opens http://localhost:8080
```
Right now the frontend still uses the **local random AI** in `src/lib/gameLogic.ts`. The game will work, but it won't talk to your Flask backend yet.

## Step 3 — Wire frontend to Flask
Edit `tictactoe/src/pages/Game.tsx`:
- Replace the `getAIMove(...)` call inside the AI-turn `useEffect` with:
  ```ts
  const res = await fetch("http://localhost:5000/api/move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boards, boardWinners, activeBoard }),
  });
  const move = await res.json();   // { boardIndex, cellIndex }
  ```
- Make the effect callback `async`, and handle the move the same way the local AI move is handled today.

## Step 4 — Sanity-check CORS
In `backend/app.py` make sure you have:
```python
from flask_cors import CORS
CORS(app)   # or CORS(app, resources={r"/api/*": {"origins": "*"}})
```
Otherwise the browser will block the request.

## Step 5 — Play
With both terminals running, open `http://localhost:8080`, click Start, and your Flask AI will be picking the moves.

---

### Technical notes
- Backend port 5000, frontend port 8080 — keep them separate; do **not** try to serve the React build from Flask for development.
- Request payload contract the frontend will send: `{ boards: (string|null)[9][9], boardWinners: (string|null)[9], activeBoard: number|null }`.
- Expected response from Flask: `{ "boardIndex": number, "cellIndex": number }`.
- If you want, I can also add a loading spinner while waiting for the backend, and a fallback to the local AI if the fetch fails.

Want me to implement Step 3 (swap the local AI for the fetch call) now?
