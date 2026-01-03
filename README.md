# ai-golf-coaches
AI golf coaches built from YouTube video transcripts.

## Quick Start

- Ensure environment variables are set (double-underscore nesting):

```
export YOUTUBE__API_KEY="<your-youtube-data-api-key>"

# Option A: provide full proxy URLs directly (recommended)
export PROXY__HTTP="http://user:pass@proxy-host:port"   # optional
export PROXY__HTTPS="http://user:pass@proxy-host:port"  # optional

# Option B: compose proxies from parts (CLI auto-applies if http/https not set)
export PROXY__HOST="proxy-host"      # e.g. proxy.webshare.io
export PROXY__PORT="<port>"          # e.g. 80 or 3128
export PROXY__USERNAME="<username>"
export PROXY__PASSWORD="<password>"
export PROXY__SCHEME="http"          # default is http
```

- Fill `config/channels.yaml` with channel IDs for each key.

- Install dependencies with Poetry:

```
poetry install
poetry run aig resolve egs
```

## Commands

- Build catalog (long-form videos only):

```
poetry run aig build-catalog elitegolfschools
```

- Fetch transcripts for videos missing them:

```
poetry run aig fetch-transcripts elitegolfschools --limit 5
```

Artifacts are stored under `data/<channel>/catalog.jsonl` and `data/<channel>/transcripts/<video_id>.jsonl`.

Notes:
- The CLI auto-applies `HTTP_PROXY`/`HTTPS_PROXY` for transcript requests if proxy URLs are provided or can be composed from `PROXY__HOST`/`PORT` and optional credentials.
- Avoid aggressive parallelization to prevent burning residential proxies.
