# EmbodiedMind — Data Sources and License Declarations

## Data Sources

### 1. Lumina Embodied-AI-Guide

- Repository: <https://github.com/TianxingChen/Embodied-AI-Guide>
- Access method: GitHub REST API / git clone (no web scraping)
- Usage scope: Non-commercial learning and research
- Citation:

  ```bibtex
  @misc{embodiedaiguide2025,
    title = {Embodied-AI-Guide},
    author = {Embodied-AI-Guide-Contributors, Lumina-Embodied-AI-Community},
    month = {January},
    year = {2025},
    url = {https://github.com/tianxingchen/Embodied-AI-Guide},
  }
  ```

### 2. HuggingFace LeRobot Documentation

- Documentation: <https://huggingface.co/docs/lerobot>
- Access method: Compliant with huggingface.co/robots.txt, rate-limited access
- Usage scope: Compliant with HuggingFace Terms of Service
- Citation:

  ```bibtex
  @misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
  }
  ```

### 3. Xbotics Embodied Intelligence Community

- Repository: <https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-Guide>
- Access method: GitHub REST API / git clone (no web scraping)
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
