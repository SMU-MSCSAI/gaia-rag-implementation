import json, os, uuid, logging, requests, io, re, time, hashlib
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from models import ProjectData


class DataCollector:
    def __init__(self, base_path):
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words("english"))
        self.content_hashes = set()
        self.title_hashes = set()

    def scrape_content(self, start_url, max_depth=2, max_pages=100, timeout=300):
        start_time = time.time()
        visited_urls, all_content = set(), []

        def process_url(url, depth):
            if (
                depth > max_depth
                or url in visited_urls
                or len(visited_urls) >= max_pages
            ):
                return
            visited_urls.add(url)
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                for elem in soup(
                    [
                        "nav",
                        "footer",
                        "header",
                        "aside",
                        "form",
                        "menu",
                        "advertisement",
                    ]
                ):
                    elem.decompose()
                content = " ".join(
                    [
                        elem.get_text(strip=True)
                        for elem in soup.select(
                            "article, div.content, div.article-content, div.entry-content, main, section, p"
                        )
                    ]
                )
                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash not in self.content_hashes:
                    self.content_hashes.add(content_hash)
                    all_content.append(content)
                    self.logger.info(f"Scraped content from {url}")
                for link in soup.find_all("a", href=True):
                    next_url = urljoin(url, link["href"])
                    if link["href"].lower().endswith(".pdf"):
                        self.process_pdf(next_url, all_content)
                    elif self.is_internal_link(next_url, start_url):
                        process_url(next_url, depth + 1)
            except Exception as e:
                self.logger.warning(f"Failed to scrape {url}: {str(e)}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.submit(process_url, start_url, 0)

        filtered_content = self.filter_and_format_content("\n\n".join(all_content))
        self.logger.info(
            f"Scraped {len(visited_urls)} pages in {time.time() - start_time:.2f} seconds"
        )
        return filtered_content

    def process_pdf(self, url, all_content):
        try:
            response = requests.get(url)
            pdf_text = PdfReader(io.BytesIO(response.content)).pages[0].extract_text()
            all_content.append(f"Content from PDF {url}:\n{pdf_text}")
            self.logger.info(f"Extracted content from PDF: {url}")
        except Exception as e:
            self.logger.warning(f"Failed to process PDF {url}: {str(e)}")

    def filter_and_format_content(self, content):
        paragraphs = content.split("\n\n")
        return "\n\n".join(
            [
                " ".join(p.split())
                for p in paragraphs
                if len(p.split()) > 10
                and not any(
                    word in p.lower()
                    for word in ["cookie", "privacy", "terms of service"]
                )
            ]
        )

    def is_internal_link(self, link, base_url):
        return urlparse(link).netloc in ["", urlparse(base_url).netloc]

    def clean_and_structure_content(self, content):
        sections = re.split(
            r"\n{2,}", re.sub(r"^.*?(?=\n\n)", "", content, flags=re.DOTALL)
        )
        structured_content = []
        for section in sections:
            section = re.sub(
                r"http\S+|www\S+|https\S+|\S+@\S+|\s+", " ", section
            ).strip()
            if len(section.split()) < 20:
                continue
            sentences = sent_tokenize(section)
            title, content = sentences[0], " ".join(sentences[1:])
            title_hash, content_hash = (
                hashlib.md5(title.encode()).hexdigest(),
                hashlib.md5(content.encode()).hexdigest(),
            )
            if title_hash in self.title_hashes or content_hash in self.content_hashes:
                continue
            self.title_hashes.add(title_hash)
            self.content_hashes.add(content_hash)
            cleaned_content = ". ".join(
                [
                    " ".join(
                        [
                            word
                            for word in sent.split()
                            if word.lower() not in self.stop_words
                        ]
                    )
                    for sent in sentences[1:]
                    if len(
                        [
                            word
                            for word in sent.split()
                            if word.lower() not in self.stop_words
                        ]
                    )
                    > 5
                ]
            )
            if cleaned_content:
                structured_content.append(
                    f"Title: {title}\n\nContent: {cleaned_content}"
                )
        return "\n\n".join(structured_content)

    def process(self, json_input: str) -> str:
        try:
            project_data = ProjectData.from_dict(json.loads(json_input))
            project_path = os.path.join(self.base_path, project_data.id)
            os.makedirs(project_path, exist_ok=True)
            self.logger.info(f"Created project with ID {project_data.id}")
            collected_data = self.scrape_content(project_data.domain)
            file_path = os.path.join(project_path, "data.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(collected_data)
            cleaned_file_path = file_path.replace(".txt", "_cleaned_structured.txt")
            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(self.clean_and_structure_content(collected_data))
            project_data.textData = cleaned_file_path
            return project_data.to_json()
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            raise
