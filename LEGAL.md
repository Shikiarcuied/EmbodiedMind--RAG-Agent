# EmbodiedMind — Data Sources and License Declarations

## Data Sources

### 1. Lumina Embodied-AI-Guide
- Repository: https://github.com/TianxingChen/Embodied-AI-Guide
- Access method: GitHub REST API / git clone (no web scraping)
- Usage scope: Non-commercial learning and research

### 2. HuggingFace LeRobot Documentation
- Documentation: https://huggingface.co/docs/lerobot
- Access method: Compliant with huggingface.co/robots.txt, rate-limited access
- Usage scope: Compliant with HuggingFace Terms of Service

### 3. Xbotics Embodied Intelligence Community
- Website: https://xbotics-embodied.site
- Access method: Compliant with robots.txt, rate limit >= 1 second/request
- Usage scope: Non-commercial learning and research

## Disclaimer

This project is for learning and research purposes only. All content copyrights belong to their original authors.
Please obtain explicit authorization from each source platform before commercial use.
If there is any infringement, please contact us for removal.

---

## Compliance Measures

- All HTTP requests carry a User-Agent identifying this bot and providing contact information
- robots.txt rules are fetched and parsed before any crawling begins
- Rate limiting is enforced: >= 1 req/s for web pages, <= 4500 req/hr for GitHub API
- HTTP 429 responses trigger automatic back-off; no forced retries
- Every ingested document carries metadata: source_url, license, crawl_date, content_hash
- All answers include source attribution links
