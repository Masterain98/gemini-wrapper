# Gemini Wrapper

## Overview

Gemini Wrapper is a FastAPI-based wrapper that mimics OpenAI's API structure while integrating Google's Gemini AI models. This allows users to interact with Gemini models using OpenAI-compatible endpoints, making it easier to integrate into existing applications.

## Features

- Supports multiple Gemini models, including experimental ones.

- Provides an OpenAI-compatible `/v1/chat/completions` endpoint.

- Includes an OpenAI-like `/v1/models` endpoint listing available Gemini models.

- Supports both streaming and non-streaming responses.

- Utilizes Google Gemini API for content generation.

- Returns responses in OpenAI-compatible format.

## Installation

1. Fulfill the `.env` file with your Google AI Studio API key and Cloudflare Tunnel token.
2. With Docker enabled, run `docker compose up --build -d` to start the FastAPI server.