#!/usr/bin/env bash
# Fetch the three public sources at the commits this analysis was run on.
#   steamcozy-ccu      Steam concurrent player samples, one CSV per collection run (git scraping)
#   steamcozy-prices   Steam store prices in 8 regions, twice a day (git scraping)
#   steamappidlist     Steam app id to name list
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/src
fetch() {  # repo sha dir [paths...]
  local repo=$1 sha=$2 dir=data/src/$3; shift 3
  if [ ! -d "$dir/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "https://github.com/$repo" "$dir"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  if [ $# -gt 0 ]; then git -C "$dir" sparse-checkout set --no-cone "$@"; fi
  git -C "$dir" checkout -q "$sha"
  echo "$repo @ ${sha:0:7}"
}
fetch scos-lab/steamcozy-ccu    328642ee5d1e35adf6d2752b765df84a3ede60f8 steamcozy-ccu    '/data/daily/' '/README.md'
fetch scos-lab/steamcozy-prices ba29457826e02bb28d955e19ca7fcbc4c944f423 steamcozy-prices '/data/' '/README.md'
fetch jsnli/steamappidlist      cd44cea6e5e12021718fccb1a3be269eab3359d2 steamappidlist   '/data/games_appid.json'
