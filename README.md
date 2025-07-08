# repo-ml-analysis-service

**repo-ml-analysis-service** is a FastAPI-based Python microservice for lightweight
ML analysis related to GitHub repositories and project data.

It is designed to work independently or as a microservice within
[DailyRepo](https://github.com/tianpai/dailyrepo) pipeline.

Currently, it performs keyword frequency analysis. Future functionality will
include star history growth and trend analysis for repositories.

## Features

1. Accepts a list of keywords and returns the top N most frequently used keywords.
2. Case-insensitive counting
3. Lightweight and fast for integration in scraping or analysis
   pipelines.
4. Designed to expand with additional ML analysis endpoints in the
   future.

## Current Endpoints

1. POST /analyze-keywords

## Planned Future Features

1. Star history growth and trend analysis.
2. Repository scoring and ranking based on growth and activity.

## License

MIT License
