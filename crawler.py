import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from unidecode import unidecode
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    separators=["\n", " ", ""]
)

class WebsiteCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited = set()
        self.site_data = []

    def is_internal_link(self, link):
        return urlparse(link).netloc == urlparse(self.base_url).netloc

    def format_text(self, text):
        return unidecode(' '.join(text.split()))

    def exclude_duplicate_content(self, parent_content, child_content):
        # Assuming both parent_content and child_content are lists of text segments
        unique_content = []
        for segment in child_content:
            if segment not in parent_content:
                unique_content.append(segment)
        return unique_content

    def crawl(self, url, parent_content=[]):        
        if url in self.visited:
            return
        response = requests.get(url)
        if response.status_code == 200:
            self.visited.add(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            texts = soup.stripped_strings
            formatted_texts = [self.format_text(text) for text in texts]
            # Exclude duplicate content by comparing with parent content
            if parent_content:
                formatted_texts = self.exclude_duplicate_content(parent_content, formatted_texts)
            
            if not formatted_texts:
                return

            links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
            internal_links = [link for link in links if self.is_internal_link(link)]

            raw_text = ' '.join(formatted_texts)
            texts = text_splitter.split_text(raw_text)
            
            for chunk_id, chunk in enumerate(texts):
                self.site_data.append({
                    'url': url,
                    'chunk-id': chunk_id,
                    'content': chunk,
                    'hyperlinks': internal_links,
                })

            for link in tqdm(internal_links, desc="Crawling"):
                self.crawl(link, formatted_texts)

    def save_data(self):
        with open('site_data.json', 'w') as f:
            json.dump(self.site_data, f, indent=4)

if __name__ == "__main__":
    # starting_url = input("Enter the starting URL for the crawler: ")
    starting_url = "https://www.markhamknifesharpening.ca/"
    crawler = WebsiteCrawler(starting_url)
    crawler.crawl(crawler.base_url)
    crawler.save_data()
