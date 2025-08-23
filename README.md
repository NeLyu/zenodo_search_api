# zenodo_search_api

![Static Badge](https://img.shields.io/badge/Powered_by-FastApi-009485?logo=fastapi)
![Static Badge](https://img.shields.io/badge/Powered_by-Zenodo?logo=zenodo)

## API for AI-powered search in Zenodo database

Currently, the first demo version has been released. It is available at http://31.97.180.10/search?q

### About

[Zenodo](https://zenodo.org/) is an open-source repository for scientific data, which is operated by CERN and was created within the OpenAIRE project to support open data.

For this version a small piece of metadata (~14000 records) was harvested via the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH). The records were processed with OpenAI "text-embedding-3-small" model and the same model is used for vectorization of the queries.

So far, API provides two options for the search: 1) AI-powered search 2) simple keyword search.

#### Next steps:
- Harvest the full repository and launch the search on that
- Add more functionality
- Consider an open-source embeddings model