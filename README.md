# Voice Identity Platform

A voice identification and speaker diarization service that helps identify unique speakers across audio recordings.

## Features

- **Speaker Diarization**: Automatically segments audio by speaker ("who spoke when")
- **Voice Embeddings**: Creates unique voice fingerprints using ECAPA-TDNN neural networks
- **Speaker Identification**: Matches new voices against known speakers in your database
- **Real-time Streaming**: WebSocket support for live audio processing
- **Multi-tenant**: Supports multiple organizations with isolated user data

## Quick Start

### Prerequisites

- Python 3.10+
- Supabase account
- HuggingFace account (for PyAnnote model access)

### Installation

1. Clone the repository and install dependencies:

```bash
cd voiceIdentity
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up your Supabase project:
   - Create a new project at [supabase.com](https://supabase.com)
   - Enable the `vector` extension in SQL Editor: `CREATE EXTENSION vector;`
   - Run the migration in `supabase/migrations/001_initial_schema.sql`
   - Create an `audio-files` storage bucket

3. Get a HuggingFace token:
   - Create account at [huggingface.co](https://huggingface.co)
   - Accept the PyAnnote model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Generate access token in Settings

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Run the server:

```bash
python -m app.main
# Or with uvicorn directly:
uvicorn app.main:app --reload
```

## API Usage

### Authentication

All requests require two headers:
- `X-API-Key`: Your organization's API key
- `X-User-Id`: External user identifier

### Process Audio File

```bash
curl -X POST "http://localhost:8000/api/v1/audio/process" \
  -H "X-API-Key: your-api-key" \
  -H "X-User-Id: user123" \
  -F "file=@recording.wav"
```

Response:
```json
{
  "conversation_id": "uuid",
  "audio_url": "https://...",
  "segments": [
    {
      "segment_id": "uuid",
      "speaker_id": "uuid",
      "speaker_name": "John",
      "is_new_speaker": false,
      "start_ms": 0,
      "end_ms": 5000,
      "confidence": 0.92
    }
  ],
  "total_speakers": 2,
  "new_speakers": 1
}
```

### List Speakers

```bash
curl "http://localhost:8000/api/v1/speakers" \
  -H "X-API-Key: your-api-key" \
  -H "X-User-Id: user123"
```

### Name a Speaker

```bash
curl -X PATCH "http://localhost:8000/api/v1/speakers/{speaker_id}" \
  -H "X-API-Key: your-api-key" \
  -H "X-User-Id: user123" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Smith"}'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket(
  'ws://localhost:8000/api/v1/stream?x_api_key=KEY&x_user_id=USER'
);

// Start a conversation
ws.send(JSON.stringify({ action: 'start' }));

// Send audio chunks (binary)
ws.send(audioChunkArrayBuffer);

// Receive identifications
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'identification') {
    console.log('Identified speakers:', data.segments);
  }
};

// End conversation
ws.send(JSON.stringify({ action: 'end' }));
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Client Apps    │────▶│   FastAPI       │
│  (REST/WebSocket)     │   Service       │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ PyAnnote │ │ ECAPA-   │ │ pgvector │
              │ Diarizer │ │ TDNN     │ │ Matcher  │
              └──────────┘ └──────────┘ └──────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │    Supabase     │
                                    │  PostgreSQL +   │
                                    │    Storage      │
                                    └─────────────────┘
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Supabase project URL | Required |
| `SUPABASE_KEY` | Supabase anon key | Required |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | Required |
| `HF_TOKEN` | HuggingFace access token | Required |
| `VOICE_MATCH_THRESHOLD` | Similarity threshold for matching | 0.75 |
| `EMBEDDING_DIMENSION` | Voice embedding size | 192 |

## License

MIT
