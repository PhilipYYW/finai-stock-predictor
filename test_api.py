import requests

params = {
    "q": "Microsoft MSFT stock",
    "from": "2026-05-17",
    "to": "2026-06-15",
    "language": "en",
    "sortBy": "publishedAt",
    "pageSize": 5,
    "apiKey": "f3829837585746d3a4a4b5d5b1a4130a",
}

resp = requests.get("https://newsapi.org/v2/everything", params=params)
data = resp.json()

print("Status:", data.get("status"))
print("Total results:", data.get("totalResults"))
print("Message:", data.get("message", ""))

for a in data.get("articles", [])[:3]:
    print("-", a["publishedAt"], a["title"][:80])