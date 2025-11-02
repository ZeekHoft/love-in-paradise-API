from time import sleep
import newspaper
from newspaper.mthreading import fetch_news
import re

NEWS_NAMES = {
    "pna.gov.ph": "Philippine News Agency",
    "rappler.com": "Rappler",
    "philstar.com": "Philstar",
    "manilatimes.net": "The Manila Times",
    "bworldonline.com": "BusinessWorld",
    "mb.com.ph": "Manila Bulletin",
}


# Parse through each url and display its content
class ArticleScraper:
    timeout = 30
    scrape_threads = 8

    def article_scraper(self, article_links, delay=1.0):
        links_data = {}

        try:

            sleep(delay)  # delay to avoid limit rates

            # add valid URL detector
            for url in article_links:
                try:
                    url_i = newspaper.Article(
                        url="%s" % (url), language="en", timeout=self.timeout
                    )
                    url_i.download()
                    url_i.parse()

                    # Only add if we got valid content
                    if url_i.title and url_i.text:
                        links_data[url] = {
                            "source_name": self.get_source_name(url),
                            "headline": url_i.title,
                            "content": url_i.text,
                        }
                    else:
                        print(f"Skipping {url} - no content extracted")

                except Exception as e:
                    # Log error but continue with other articles
                    print(f"Error scraping {url}: {e}")
                    continue

            print(
                f"Successfully scraped {len(links_data)} out of {len(article_links)} articles"
            )
            return links_data
            # return "\n".join(article_content)
        except Exception as e:
            # Return empty dict instead of string
            print(f"Error in article scraper: {e}")
            return {}

    def scrape_multithreaded(self, article_links: list[str]):
        articles_data = {}
        try:
            articles = [
                newspaper.Article(url=link, timeout=self.timeout)
                for link in article_links
            ]
            results = fetch_news(articles, threads=self.scrape_threads)
            for i in range(len(results)):
                # Only add if we got valid content
                result = results[i]
                url = article_links[i]
                if result.title and result.text:
                    articles_data[url] = {
                        "source_name": self.get_source_name(url),
                        "headline": result.title,
                        "content": result.text,
                    }
                else:
                    print(f"Skipping {url} - no content extracted")
            return articles_data
        except Exception as e:
            # Return empty dict instead of string
            print(f"Error in article scraper: {e}")
            return {}

    def get_source_name(self, url):
        # Get the website or source name based on url
        match = re.search(
            r"^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/\n?]+)", url
        )
        if match:
            domain = match.group(1)
            return NEWS_NAMES.get(domain, "Unknown")
        return "Unkown"


# if __name__ == "__main__":
#     news = ArticleScraper()
#     print(news.article_scraper(list_of_urls))
