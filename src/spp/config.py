from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "src"
CCU_GLOB = str(SRC / "steamcozy-ccu" / "data" / "daily" / "*" / "*.csv")
PRICE_GLOB = str(SRC / "steamcozy-prices" / "data" / "*" / "*.csv")
NAMES = SRC / "steamappidlist" / "data" / "games_appid.json"
DB = ROOT / "data" / "panel.duckdb"
OUT = ROOT / "outputs"
DOCS = ROOT / "docs" / "data"

SOURCES = {
    "steamcozy-ccu": "328642ee5d1e35adf6d2752b765df84a3ede60f8",
    "steamcozy-prices": "ba29457826e02bb28d955e19ca7fcbc4c944f423",
    "steamappidlist": "cd44cea6e5e12021718fccb1a3be269eab3359d2",
}

# Steam Autumn Sale 2026 ran from October 1 to October 8 (store-wide).
SALE_START = date(2026, 10, 1)
SALE_END = date(2026, 10, 8)

MIN_SAMPLES = 30        # samples a game needs for a daily cycle fit
MIN_MEAN_PLAYERS = 200  # games quieter than this are dominated by rounding noise
