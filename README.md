# zenodo_search_api

![Static Badge](https://img.shields.io/badge/Powered_by-FastApi-009485?logo=fastapi)
![Static Badge](https://img.shields.io/badge/Powered_by-Zenodo-2b7fff?logo=zenodo)

## API for AI-powered search in Zenodo database

Currently, the first demo version has been released. It is available at http://31.97.180.10/search?q

### About

[Zenodo](https://zenodo.org/) is an open-source repository for scientific data, which is operated by CERN and was created within the OpenAIRE project to support open data.

For this version a small piece of metadata (~14000 records) was harvested via the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH). The records were processed with OpenAI "text-embedding-3-small" model and the same model is used for vectorization of the queries.

So far, API provides two options for the search: 1) AI-driven search 2) simple keyword search.

## Usage
**cURL**

AI-driven search
```
curl -X GET "http://31.97.180.10/search?query=epidemiology"
```
Key-word search
```
curl -X GET "http://31.97.180.10/keyword-search?query=epidemiology&limit=10"
```

**Python**

```python
import requests

url = "http://31.97.180.10/search"
params = {
    "query": "epidemiology",
}

response = requests.get(url, params=params)
data = response.json()
```


## Planned Improvements
- Harvest the full repository and enable full-scale search
- Extend API functionality
- Explore integration with open-source embedding models