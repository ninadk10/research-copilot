import feedparser

def arxiv_search(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": entry.links[1].href,  # link[0] is abstract, link[1] is PDF
            "authors": [a.name for a in entry.authors],
            "published": entry.published
        })
    return results
