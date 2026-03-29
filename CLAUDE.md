# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository is a RAG (Retrieval Augmented Generation) project built on a Scrapbox (Cosense) knowledge base export. The data file `tipsofmlandcs.json` contains 205 interconnected notes on Machine Learning and Customer Success topics, authored by Tatsuya (tandf214355@gmail.com).

## Data File Structure

`tipsofmlandcs.json` is a Scrapbox export with the following schema:

```json
{
  "name": "tipsofmlandcs",
  "displayName": "Machine Learning & Customer Success",
  "exported": 1774786565,
  "users": [{ "id": "...", "name": "...", "displayName": "...", "email": "..." }],
  "pages": [
    {
      "id": "...",
      "title": "Page title",
      "created": 1234567890,
      "updated": 1234567890,
      "views": 42,
      "lines": ["line 1 content", "line 2 content", ...]
    }
  ]
}
```

### Scrapbox Markup in `lines`
- `[page title]` — internal link to another page
- `[url text]` or `[text url]` — external link
- `#tag` — hashtag
- `[* text]` — bold
- `` `code` `` — inline code
- `> text` — blockquote
- Lines starting with a space — indented (hierarchical structure)

## Content Domain

The 205 pages cover:
- **ML/Data Science**: Uplift modeling, causal inference, A/B testing, TabNet, LGBM, CausalImpact
- **Customer Success**: CS frameworks, sales enablement, ABM, community-led growth
- **Business Strategy**: SEDA model, pricing, product management, organizational behavior
- **Data Platform**: Data mesh, data-driven decision making, analytics architecture

Content is primarily in Japanese with English technical terms mixed in.

## Intended Architecture (RAG)

When building the RAG system over this data:
- Parse `pages[].lines` arrays into document chunks
- Use `pages[].title` as document identifier
- Scrapbox internal links (`[page title]`) encode the knowledge graph structure — these can be used to enrich context or build a graph index
- Each page is a coherent topic unit suitable for chunk-level embedding
