# Troubleshooting

## pygame window does not open

`render_mode="human"` and `pickomino-play` require a display. They will not work
on headless servers or in environments without a display.

Use `render_mode=None` for training without a window.

## Installation fails

Make sure you are installing in a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pickomino-env
```

## Wrong Python version

Pickomino requires Python 3.10–3.14. Check your version:
```bash
python --version
```

## ValueError: render_mode

Valid values are `None`, `"human"`, and `"rgb_array"`. Any other value raises
a `ValueError`.

## ValueError: number_of_bots

Valid range is 1–6. Any value outside this range raises a `ValueError`.
