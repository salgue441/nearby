#!/usr/bin/env sh
# A tour of the HTTP API with curl: a collection of articles with payloads,
# filtered and batch searches, updates, deletes and metrics.
#
# Start a server first, for example:
#   cargo run -p nearby-server --release
#   docker run -p 7700:7700 -e NEARBY_API_KEY=change-me nearby
# then run:
#   NEARBY_URL=http://127.0.0.1:7700 NEARBY_API_KEY=change-me sh examples/http_api.sh

set -eu

URL="${NEARBY_URL:-http://127.0.0.1:7700}"
KEY="${NEARBY_API_KEY:-}"

# Sends a request with an optional JSON body and prints the response.
call() {
    method="$1"; path="$2"; body="${3:-}"
    echo "> $method $path ${body:+$body}"
    set -- -s -X "$method" "$URL$path" -H 'content-type: application/json'
    [ -n "$KEY" ] && set -- "$@" -H "Authorization: Bearer $KEY"
    [ -n "$body" ] && set -- "$@" -d "$body"
    curl "$@"
    printf '\n\n'
}

call GET /health

# Vectors are 3-d topic mixes (science, sports, politics), standing in for
# the embeddings a model would produce. Cosine distance compares direction.
call PUT /collections/articles '{"dim": 3, "metric": "cosine", "index": {"type": "hnsw"}}'

call PUT /collections/articles/points '{"points": [
  {"id": 1, "vector": [0.9, 0.1, 0.0], "payload": {"title": "New exoplanet found", "year": 2024, "lang": "en"}},
  {"id": 2, "vector": [0.8, 0.0, 0.3], "payload": {"title": "Climate bill passes", "year": 2023, "lang": "en"}},
  {"id": 3, "vector": [0.1, 0.9, 0.1], "payload": {"title": "Cup final recap", "year": 2024, "lang": "es"}},
  {"id": 4, "vector": [0.0, 0.2, 0.9], "payload": {"title": "Election results", "year": 2024, "lang": "es"}},
  {"id": 5, "vector": [0.7, 0.6, 0.0], "payload": {"title": "The physics of a free kick", "year": 2022, "lang": "en"}}
]}'

call GET /collections/articles/points/1

# Nearest to "mostly science", with payloads in the results.
call POST /collections/articles/search '{"vector": [1, 0, 0], "k": 3, "with_payload": true}'

# The same, limited to 2024 articles in English.
call POST /collections/articles/search '{"vector": [1, 0, 0], "k": 3, "with_payload": true,
  "filter": {"and": [{"field": "year", "gte": 2024}, {"field": "lang", "eq": "en"}]}}'

# Several queries in one request; results come back in order.
call POST /collections/articles/search/batch '{"searches": [
  {"vector": [0, 1, 0], "k": 2},
  {"vector": [0, 0, 1], "k": 2, "filter": {"field": "lang", "in": ["es"]}}
]}'

# Writing a point replaces its vector and payload; deleting removes it.
call PUT /collections/articles/points '{"points": [{"id": 5, "vector": [0.2, 0.9, 0.0], "payload": {"title": "Free kicks, explained", "year": 2025, "lang": "en"}}]}'
call POST /collections/articles/points/delete '{"ids": [2]}'
call GET /collections/articles

# Request counts, latencies and collection sizes, in the Prometheus format.
echo "> GET /metrics (collection gauges)"
set -- -s "$URL/metrics"
[ -n "$KEY" ] && set -- "$@" -H "Authorization: Bearer $KEY"
curl "$@" | grep '^nearby_collection'
echo

call DELETE /collections/articles
