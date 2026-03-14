#!/usr/bin/env python3
"""
Chess game viewer - Generate interactive HTML board from PGN.

Generates a standalone HTML file with an interactive chess board that allows
navigating through a game's moves.

Usage:
    python scripts/view_game.py data/raw/pgnmentor/game.json
    python scripts/view_game.py data/raw/pgnmentor/game.json --output viewer.html
    python scripts/view_game.py data/raw/pgnmentor/game.json --auto-open
"""

import argparse
import io
import json
import webbrowser
from pathlib import Path

import chess.pgn

# Add src to path
#sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_agentic_rag.core.models import Document


def load_game_from_json(json_path: Path) -> dict:
    """Load game data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_moves_from_pgn(pgn_string: str) -> list[dict]:
    """
    Extract moves from PGN string.
    
    Returns:
        List of move dicts with SAN notation and FEN position
    """
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    
    if not game:
        return []
    
    moves = []
    board = game.board()
    
    for i, move in enumerate(game.mainline_moves(), 1):
        san = board.san(move)
        board.push(move)
        
        moves.append({
            "number": (i + 1) // 2,
            "color": "white" if i % 2 == 1 else "black",
            "san": san,
            "fen": board.fen(),
        })
    
    return moves


def reconstruct_pgn(metadata: dict) -> str:
    """Reconstruct PGN from metadata if moves are available."""
    # Try to find moves in metadata or content
    # For now, we'll create a minimal PGN
    white = metadata.get('white', 'Unknown')
    black = metadata.get('black', 'Unknown')
    event = metadata.get('event', 'Unknown')
    site = metadata.get('site', 'Unknown')
    date = metadata.get('date', '????.??.??')
    result = metadata.get('result', '*')
    
    pgn_parts = [
        f'[Event "{event}"]',
        f'[Site "{site}"]',
        f'[Date "{date}"]',
        f'[White "{white}"]',
        f'[Black "{black}"]',
        f'[Result "{result}"]',
        '',
        result  # For now, just the result
    ]
    
    return '\n'.join(pgn_parts)


def generate_html(game_data: dict, moves: list[dict]) -> str:
    """Generate standalone HTML viewer."""
    
    metadata = game_data.get('metadata', {})
    white = metadata.get('white', 'Unknown')
    black = metadata.get('black', 'Unknown')
    white_elo = metadata.get('elo_white', '?')
    black_elo = metadata.get('elo_black', '?')
    event = metadata.get('event', 'Unknown')
    site = metadata.get('site', 'Unknown')
    date = metadata.get('date', '????.??.??')
    opening = metadata.get('opening', 'Unknown')
    eco = metadata.get('eco', '')
    result = metadata.get('result', '*')
    
    # Convert moves to JavaScript array
    moves_js = json.dumps(moves)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{white} vs {black} - Chess Game Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .players {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
            font-size: 1.2em;
        }}
        
        .player {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .player-name {{
            font-weight: bold;
            font-size: 1.3em;
        }}
        
        .player-elo {{
            color: #ffd700;
            font-size: 0.9em;
        }}
        
        .game-info {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            padding: 30px;
        }}
        
        @media (max-width: 900px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .board-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        #chessboard {{
            width: 100%;
            max-width: 600px;
            aspect-ratio: 1;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            grid-template-rows: repeat(8, 1fr);
            border: 3px solid #333;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .square {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3em;
            cursor: default;
            transition: background-color 0.2s;
        }}
        
        .square.light {{
            background-color: #f0d9b5;
        }}
        
        .square.dark {{
            background-color: #b58863;
        }}
        
        .square.highlight {{
            background-color: #baca44 !important;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }}
        
        button {{
            padding: 12px 24px;
            font-size: 1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}
        
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .info-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }}
        
        .info-panel h3 {{
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #666;
        }}
        
        .info-value {{
            color: #333;
        }}
        
        .moves-panel {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .move-list {{
            display: grid;
            grid-template-columns: 40px 1fr 1fr;
            gap: 8px;
            font-family: monospace;
        }}
        
        .move-number {{
            font-weight: bold;
            color: #666;
        }}
        
        .move {{
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        
        .move:hover {{
            background-color: #e9ecef;
        }}
        
        .move.active {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        .current-move {{
            text-align: center;
            font-size: 1.2em;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>♟️ Chess Game Viewer</h1>
            
            <div class="players">
                <div class="player">
                    <div class="player-name">⬜ {white}</div>
                    <div class="player-elo">ELO: {white_elo}</div>
                </div>
                <div class="player">
                    <div class="player-name">⬛ {black}</div>
                    <div class="player-elo">ELO: {black_elo}</div>
                </div>
            </div>
            
            <div class="game-info">
                <span>📍 {event}</span>
                <span>🌍 {site}</span>
                <span>📅 {date}</span>
                <span>🎯 {result}</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="board-container">
                <div id="chessboard"></div>
                
                <div class="current-move" id="currentMove">
                    Initial Position
                </div>
                
                <div class="controls">
                    <button onclick="goToStart()">⏮️ Start</button>
                    <button onclick="previousMove()">◀️ Previous</button>
                    <button onclick="toggleAutoPlay()" id="playBtn">▶️ Play</button>
                    <button onclick="nextMove()">▶️ Next</button>
                    <button onclick="goToEnd()">⏭️ End</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="info-panel">
                    <h3>Game Information</h3>
                    <div class="info-item">
                        <span class="info-label">Opening:</span>
                        <span class="info-value">{opening}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">ECO:</span>
                        <span class="info-value">{eco}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Moves:</span>
                        <span class="info-value" id="totalMoves">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Current:</span>
                        <span class="info-value" id="currentMoveNum">0</span>
                    </div>
                </div>
                
                <div class="moves-panel">
                    <h3>Moves</h3>
                    <div class="move-list" id="moveList"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const moves = {moves_js};
        let currentPosition = -1;
        let autoPlayInterval = null;
        
        const pieceSymbols = {{
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }};
        
        function parseFEN(fen) {{
            const parts = fen.split(' ');
            const position = parts[0];
            const rows = position.split('/');
            const board = [];
            
            for (let row of rows) {{
                let boardRow = [];
                for (let char of row) {{
                    if (isNaN(char)) {{
                        boardRow.push(char);
                    }} else {{
                        for (let i = 0; i < parseInt(char); i++) {{
                            boardRow.push('');
                        }}
                    }}
                }}
                board.push(boardRow);
            }}
            
            return board;
        }}
        
        function renderBoard(fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1') {{
            const board = parseFEN(fen);
            const chessboard = document.getElementById('chessboard');
            chessboard.innerHTML = '';
            
            for (let row = 0; row < 8; row++) {{
                for (let col = 0; col < 8; col++) {{
                    const square = document.createElement('div');
                    square.className = 'square ' + ((row + col) % 2 === 0 ? 'light' : 'dark');
                    
                    const piece = board[row][col];
                    if (piece) {{
                        square.textContent = pieceSymbols[piece] || piece;
                    }}
                    
                    chessboard.appendChild(square);
                }}
            }}
        }}
        
        function updateMoveDisplay() {{
            const currentMoveEl = document.getElementById('currentMove');
            const currentMoveNumEl = document.getElementById('currentMoveNum');
            
            if (currentPosition === -1) {{
                currentMoveEl.textContent = 'Initial Position';
                currentMoveNumEl.textContent = '0';
            }} else {{
                const move = moves[currentPosition];
                const moveNum = Math.floor(currentPosition / 2) + 1;
                const color = currentPosition % 2 === 0 ? '⬜' : '⬛';
                currentMoveEl.textContent = `${{moveNum}}. ${{color}} ${{move.san}}`;
                currentMoveNumEl.textContent = currentPosition + 1;
            }}
            
            // Update move list highlighting
            document.querySelectorAll('.move').forEach((el, idx) => {{
                el.classList.toggle('active', idx === currentPosition);
            }});
        }}
        
        function nextMove() {{
            if (currentPosition < moves.length - 1) {{
                currentPosition++;
                renderBoard(moves[currentPosition].fen);
                updateMoveDisplay();
            }}
        }}
        
        function previousMove() {{
            if (currentPosition >= 0) {{
                currentPosition--;
                const fen = currentPosition === -1 
                    ? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
                    : moves[currentPosition].fen;
                renderBoard(fen);
                updateMoveDisplay();
            }}
        }}
        
        function goToStart() {{
            currentPosition = -1;
            renderBoard();
            updateMoveDisplay();
        }}
        
        function goToEnd() {{
            if (moves.length > 0) {{
                currentPosition = moves.length - 1;
                renderBoard(moves[currentPosition].fen);
                updateMoveDisplay();
            }}
        }}
        
        function goToMove(index) {{
            if (index >= -1 && index < moves.length) {{
                currentPosition = index;
                const fen = index === -1 
                    ? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
                    : moves[index].fen;
                renderBoard(fen);
                updateMoveDisplay();
            }}
        }}
        
        function toggleAutoPlay() {{
            const playBtn = document.getElementById('playBtn');
            
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                playBtn.textContent = '▶️ Play';
            }} else {{
                playBtn.textContent = '⏸️ Pause';
                autoPlayInterval = setInterval(() => {{
                    if (currentPosition < moves.length - 1) {{
                        nextMove();
                    }} else {{
                        toggleAutoPlay();
                    }}
                }}, 1000);
            }}
        }}
        
        function renderMoveList() {{
            const moveList = document.getElementById('moveList');
            moveList.innerHTML = '';
            
            for (let i = 0; i < moves.length; i += 2) {{
                const moveNum = Math.floor(i / 2) + 1;
                
                const numEl = document.createElement('div');
                numEl.className = 'move-number';
                numEl.textContent = moveNum + '.';
                moveList.appendChild(numEl);
                
                const whiteMove = document.createElement('div');
                whiteMove.className = 'move';
                whiteMove.textContent = moves[i].san;
                whiteMove.onclick = () => goToMove(i);
                moveList.appendChild(whiteMove);
                
                if (i + 1 < moves.length) {{
                    const blackMove = document.createElement('div');
                    blackMove.className = 'move';
                    blackMove.textContent = moves[i + 1].san;
                    blackMove.onclick = () => goToMove(i + 1);
                    moveList.appendChild(blackMove);
                }} else {{
                    moveList.appendChild(document.createElement('div'));
                }}
            }}
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') previousMove();
            if (e.key === 'ArrowRight') nextMove();
            if (e.key === ' ') {{
                e.preventDefault();
                toggleAutoPlay();
            }}
        }});
        
        // Initialize
        document.getElementById('totalMoves').textContent = moves.length;
        renderBoard();
        renderMoveList();
        updateMoveDisplay();
    </script>
</body>
</html>'''
    
    return html


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chess game viewer - Generate interactive HTML board",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "game_file",
        type=Path,
        help="Path to game JSON file",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML file path (default: same name as input)",
    )
    
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Automatically open in browser",
    )
    
    args = parser.parse_args()
    
    # Load game
    print(f"Loading game from {args.game_file}...")
    game_data = load_game_from_json(args.game_file)
    
    # Extract moves (we need the actual PGN for this)
    # For PGNMentor files, we need to reconstruct or store full PGN
    print("Extracting moves...")
    
    # For now, create a placeholder - you'll need to update PGNMentorLoader
    # to also store the full PGN string in metadata
    moves = []
    
    # Check if we have PGN in metadata
    if 'pgn' in game_data.get('metadata', {}):
        pgn_string = game_data['metadata']['pgn']
        moves = extract_moves_from_pgn(pgn_string)
    else:
        print("Warning: No PGN data found in JSON. Board will show initial position only.")
        print("Tip: Update PGNMentorLoader to save full PGN in metadata['pgn']")
    
    # Generate HTML
    print("Generating HTML...")
    html = generate_html(game_data, moves)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.game_file.with_suffix('.html')
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML viewer generated: {output_path.absolute()}")
    
    # Auto-open in browser
    if args.auto_open:
        print("Opening in browser...")
        webbrowser.open(f'file://{output_path.absolute()}')
    else:
        print(f"\nTo view: open {output_path.absolute()}")
        print("Or run with --auto-open flag")


if __name__ == "__main__":
    main()