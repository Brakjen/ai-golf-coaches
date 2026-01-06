## AI Golf Coach Development

### Data Retrieval and Storage
We will retrieve transcriptions from Youtube videos from select channels:
- Elite Golf Schools (@elitegoolfschools)
- Milo Lines Golf (@milolinesgolf)

We will use the Youtube Data API from Googole to fetch transcripts for all videos
from each channel.
Some limitations:

- We are only interested in the long form videos, not the "Shorts".
- Elite Golf Schools have some Livestream long for videos. These are live Question and Answer sessions where the instructor answers questions from viewers. We also want these videos, but they need to be flagged as "livestream" or something similar.

I want to store the transcripts as JSON lines (.jsonl), where each line is a JSON parseable piece of the transcript.
Each line will be of the form

```json
{
    "chunk": "bla bla bla", "start": "<seconds into video>"
}
```

or similar based on how the Youtube Data API returns the trascript.
All data will be stored in a `data` directory, with subdirectories per channel.

```
./data/
├── elitegolfschools
│   └── trascripts
└── milolinesgolf
    └── transcripts
```

We will also store a `catalog.json` file for each channel that stores metadata about each video available on the channel.
Ergo, the catalogs will represent our most recent knowledge of what is available on the channel.
We may not have transcripts for all videos in the catalog (which we can use to target new transcripts specifically).
So the `data` directory look like this:

```
./data/
├── elitegolfschools
│   ├── catalog.jsonl
│   └── trascripts
└── milolinesgolf
    ├── catalog.jsonl
    └── transcripts
```

Now let's talk a bit about convenience (but not solutioning!).
At some point we will implement a CLI to access various functionality.
We will probably specify which channel we are working with, but I don't want to specify
the entire longform handle (e.g. @elitegolfschools or just elitegolfschools).
I want to maintain a configuration file where we can stre settings per channel.
One setting could be one or more aliases.
For example

```yaml
elitegolfschools:
  handle: @elitegolfschools
  channel_id: <id>
  aliases:
    - egs
    - riley
    - elite

milolinesgolf:
  handle: @milolinesgolf
  channel_id: <id>
  aliases:
    - mlg
    - milo
```


### Coding Style
#### Data Contracts
I want to use data contracts on to make sure our data is stored correctly.
This means using pydantic to define things like API responses, transcript objectsm transcript chunk objects, livestream objects, etc.

#### API Keys and Secrets
Secrets and key variables are defined as environment variables.
They use the "double under" syntax to indicate diffrent classes of secrets.
This way we can load with pydantic and get "dot" access.
A YAML with env vars and descriptions can be found in `config/secrets.yaml`.

#### Proxies
We need to use proxies when fetching transcript files.
I have set up an account with Webshare that gives me rotating residential proxies.
We don't want to burn these proxies, so we need to be mindful of how we implement the data retrieval.
We can't parallelize too much, and we need to back off a bit after a failed request.


#### Documentation
- Docstrings. We ALWAYS add Google style docstrings to all functions and classes.
- We always give accurate type hints to our functions.
- We always give type hints to in line variable definitions when the type is not obvious, for example when a function returns one of our own custom data contract classes.

### Extracting QA segments from livestreams
We will have to extract all questions and answers from all the livestreams.
I think the best way is to use a good openai model to read the transcripts, parse all user questions and coach answers,
and store these as jsonl.
Then we can later build an index on the QA segments, and match our end user questions with the QA segments.

We could additionally match the AI coach answer against the QA segment answers, and showcase those also.
Two different roads to relevant content for our end users.

### Indexing of long form videos
Build index on videos for channels individually.
I don't want to store the index locally, but store it in openai.
This way we keep the repo clean and light, and deployment faster.

The index will be used for semantic matching of question with our transcription chunks.
This way we can present to the user the most relevant videos and timestamps.

### Indexing of livestreams
Similar idea as above.
We will match the question against all other livestream questions,
and present to the user the most similar ones along with the coach response.
I think this will be useful if the matching works well.


### Agent Definitions
#### Simple test framework
- Use an openai model that accepts 128k context tokens
- Don't use any RAG
- We will provide a static context, as defined in the channels yaml file. A util prepares the context.
- We pass the static context to the agent, and let it form a response based on this.
- Agent instructions/tones is defined in its own file under config. We also pass this to the agent.
- Again, no RAG yet. I want to test the static context first.

Lets make a very simple function that takes in a question that gets passed to the openai model, along with the additional context mentioned above.
I don't want special CLI options now.
I don't want testing frameworks. By test I mean a very simple proof of concept that lets me ask an agent that has my special context.
Lets hard code models and parameters for now to keep it simple.
