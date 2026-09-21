from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .schemas import Document


class DocumentLoader:
    """Download and clean HTML documents."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def download(
        self,
        url: str,
        document_id: str,
    ) -> Path:
        """Download a document unless it already exists locally."""

        output_path = self.raw_dir / f"{document_id}.html"

        if output_path.exists():
            return output_path

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "RAG-Evaluation-Benchmark/1.0"
            },
        )

        response.raise_for_status()

        # Preserve the original response bytes so that
        # UTF-8 characters are not corrupted.
        output_path.write_bytes(response.content)

        return output_path

    def parse_html(
        self,
        html_path: str | Path,
        document_id: str,
        title: str,
        source_url: str,
    ) -> Document:
        """Parse HTML and extract readable text."""

        html_path = Path(html_path)

        html = html_path.read_text(
            encoding="utf-8"
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        # Remove elements that are not part of the
        # useful documentation content.
        for element in soup(
            ["script", "style", "nav", "footer", "header"]
        ):
            element.decompose()

        main_content = soup.find("main")

        if main_content is None:
            main_content = soup

        text = main_content.get_text(
            separator="\n",
            strip=True,
        )

        text = self._clean_text(text)

        return Document(
            document_id=document_id,
            title=title,
            source_url=source_url,
            text=text,
            metadata={
                "source": "Python Documentation",
                "content_type": "text/html",
            },
        )

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """Normalize excessive whitespace."""

        lines = [
            line.strip()
            for line in text.splitlines()
        ]

        lines = [
            line
            for line in lines
            if line
        ]

        return "\n".join(lines)

    def load(
        self,
        document_id: str,
        title: str,
        url: str,
    ) -> Document:
        """Download and parse one document."""

        html_path = self.download(
            url=url,
            document_id=document_id,
        )

        return self.parse_html(
            html_path=html_path,
            document_id=document_id,
            title=title,
            source_url=url,
        )