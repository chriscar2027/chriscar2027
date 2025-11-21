```markdown
# Trading Decision Engine — Plug & Play

This repository contains a plug-and-play trading decision engine (refactor of original v1.002) with tests and CI.

## Quickstart (local)

1. Clone:
   git clone https://github.com/chriscar2027/chriscar2027.git
   cd chriscar2027

2. Create and activate a virtualenv (recommended):
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run the bot:
   python bot.py --ticker NVDA

   Optionally specify a simulated headline:
   python bot.py --ticker NVDA --headline "Nvidia CEO arrested for fraud"

5. Run tests:
   pytest -q

## What I changed / why

- Renamed the original script into `bot.py` and restructured into functions for clarity and testability.
- Added deterministic unit test (mocks yfinance and sentiment) so CI doesn't rely on network.
- Added CI workflow that installs deps and runs tests automatically on push/PR.
- Added `requirements.txt` for reproducible installs.

## Notes

- The test suite mocks external network calls (yfinance and VADER) to be deterministic and fast.
- The code gracefully exits if it cannot fetch a ticker's data; CI/test path uses mocked data so tests pass without network.
```
