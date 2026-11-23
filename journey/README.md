# Project Journey page

An interactive one-page walk through the whole project. Scroll down a winding path: each
checkpoint is a phase (0 → 6), and the finish line covers returns and strategies. Click a
checkpoint to open a side panel with tabs for that phase: **Input · Procedure · Example ·
Output**. The finish line has these tabs:

- Results
- Growth chart (interactive)
- What each means
- Beat the market?
- Game theory
- Glossary

## Files
| File | What it is |
|---|---|
| `index.html` | The page. All text, styles and code are in this one file, with no build step |
| `data.js` | Weekly growth-of-₹100 curves, 2018–2025, for strategies A–G, "buy every call" and the S&P 500 |
| `build_data.py` | Regenerates `data.js` from the committed `data/backtest_finbert*_daily.csv` files |

## View it
Open `index.html` in a browser, or serve the folder:
```bash
python -m http.server 8000 --directory journey
# then open http://localhost:8000
```

## Rebuild the chart data
```bash
python journey/build_data.py
```

## Deploy
The folder is static, so any static host works, for example GitHub Pages pointed at `/journey`,
Netlify or Vercel. Fonts load from Google Fonts. When offline, the page falls back to system fonts.
