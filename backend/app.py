from flask import Flask, request, jsonify
from flask_cors import CORS
from ai import get_ai_move

app = Flask(__name__)
CORS(app)  # Allow frontend on localhost:8080 to talk to backend on localhost:5000


@app.route("/api/move", methods=["POST"])
def api_move():
    """
    Expects JSON:
    {
      "boards":       [["X",null,"O", ...], ...],   // 9 boards of 9 cells
      "boardWinners": [null,"X",null,...],            // 9 winners
      "activeBoard":  4                                // int or null
    }

    Returns JSON:
    {
      "boardIndex": 4,
      "cellIndex":  7
    }
    """
    data = request.get_json(force=True)

    boards = data["boards"]
    board_winners = data["boardWinners"]
    active_board = data.get("activeBoard")

    move = get_ai_move(boards, board_winners, active_board)

    if move is None:
        return jsonify({"error": "No valid moves"}), 400

    return jsonify(move)


@app.route("/", methods=["GET"])
def health():
    return "Flask backend is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
