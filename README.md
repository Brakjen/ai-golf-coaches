# ai-golf-coaches
AI golf coaches built from YouTube video transcripts.

## Quick Start

- Ensure environment variables are set (double-underscore nesting):

```
export YOUTUBE__API_KEY="<your-youtube-data-api-key>"
export PROXY__HTTP="http://user:pass@proxy-host:port"   # optional
export PROXY__HTTPS="http://user:pass@proxy-host:port"  # optional
```

- Fill `config/channels.yaml` with channel IDs for each key.

- Install dependencies with Poetry:

```
poetry install
poetry run ai-golf resolve egs
```

## Commands

- Build catalog (long-form videos only):

```
poetry run ai-golf build_catalog elitegolfschools
```

- Fetch transcripts for videos missing them:

```
poetry run ai-golf fetch_transcripts elitegolfschools --limit 5
```

Artifacts are stored under `data/<channel>/catalog.jsonl` and `data/<channel>/transcripts/<video_id>.jsonl`.
