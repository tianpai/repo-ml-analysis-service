# repo-ml-analysis-service

**repo-ml-analysis-service** is a FastAPI-based Python microservice for lightweight
ML analysis related to GitHub repositories and project data.

It is designed to work independently or as a microservice within
[DailyRepo](https://github.com/tianpai/dailyrepo) pipeline.

Currently, it performs keyword frequency and clustering analysis using ML.
Future functionality will include star history growth and trend analysis for
repositories.

## Features

1. Accepts a list of keywords and returns the top N most frequently used
   keywords using a small [transformer model](<https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)>).

## Current Endpoints

1. POST /analyze-keywords

```bash
curl --location 'http://localhost:8000/analyze-keywords' \
--header 'Content-Type: application/json' \
--data '{
  "topics": [
    "llms",
    "python",
    "rag",
    "ai",
    "cursor",
    "cursor-ai",
    "cursorai",
    "roocode",
    "task-manager",
    "tasks",
    "tasks-list",
    "windsurf",
    "windsurf-ai",
    "agents",
    "ai",
    "genai",
    "llm",
    "llms",
    "openai",
    "tutorials",
    "android",
    "css",
    "desktop",
    "html",
    "neovim",
    "tmux",
    "agents",
    "ai",
    "context-window",
    "framework",
    "llms",
    "linux",
    "game-development",
    ],
    "topN": 5,
    "includeRelated": true,
    "distance_threshold": 0.25,
    "includeClusterSizes": true
}'
```

response:

```json
{
  "topKeywords": ["llms", "ai", "cursor", "tasks", "windsurf"],
  "related": {
    "llms": ["llm"],
    "ai": [],
    "cursor": ["cursor-ai"],
    "tasks": ["tasks-list"],
    "windsurf": ["windsurf-ai"]
  },
  "clusterSizes": {
    "llms": 2,
    "ai": 1,
    "cursor": 2,
    "tasks": 2,
    "windsurf": 2
  }
}
```

OR typescript interface:

```typescript
interface KeywordAnalysisResponse {
  topKeywords: string[];
  related: {
    [key: string]: string[];
  };
  clusterSizes: {
    [key: string]: number;
  };
}
```

## Planned Future Features

1. Star history growth and trend analysis.
2. Repository scoring and ranking based on growth and activity.

## NOTE

1. After the server is deployed, there is a short wait time for the transformer
   model to be downloaded and loaded into memory. This is normal and expected.

## License

MIT License

```

```
