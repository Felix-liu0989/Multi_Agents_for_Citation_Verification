import os
import time
from io import BytesIO

import arxiv
import fitz
import requests
from bertopic import BERTopic

def get_arxiv_title(arxiv_id: str) -> str:
    # Use the Client for fetching paper details
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)

    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"

    # Format the citation (e.g., APA style)
    title = paper.title
    return title


def get_arxiv_citation(arxiv_id: str) -> str:
    # Use the Client for fetching paper details
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)

    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"

    # Format the citation (e.g., APA style)
    authors = ", ".join(author.name for author in paper.authors)
    title = paper.title
    year = paper.published.year
    return f"{authors} ({year}). {title}. arXiv:{arxiv_id}. https://arxiv.org/abs/{arxiv_id}"


def get_arxiv_abstract(arxiv_id: str) -> str:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)

    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"
    return paper.summary


def get_arxiv_publication_date(arxiv_id: str) -> str:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)

    if not paper:
        return f"No paper found for arXiv ID: {arxiv_id}"
    return str(paper.published)


def download_pdf(arxiv_id: str, save_path: str = "paper.pdf", retries: int = 3) -> bool:
    """
    Download the PDF from arXiv using the arXiv ID with retry logic.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes

            # Download in chunks and write to file
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):  # 1 KB chunks
                    if chunk:
                        f.write(chunk)

            print(f"PDF downloaded successfully: {save_path}")
            return True  # Success
        except (
            requests.exceptions.RequestException,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            print(f"Download failed, attempt {attempt + 1}/{retries}: {e}")
            time.sleep(2)  # Wait a little before retrying
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    print("Failed to download PDF after several attempts.")
    return False


def extract_text_by_page(pdf_path: str) -> list[str]:
    """
    Extract text from each page of the downloaded PDF using PyMuPDF (fitz).
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        pages_text.append(page.get_text())  # Extract text from the page

    return pages_text


def extract_text_by_page_from_pdf(pdf_content: bytes) -> list[str]:
    """
    Extract text from each page of the downloaded PDF using PyMuPDF (fitz).
    """
    doc = fitz.open(stream=BytesIO(pdf_content), filetype="pdf")
    pages_text = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load each page
        pages_text.append(page.get_text())  # Extract text from the page

    return pages_text


def process_arxiv_paper(arxiv_id: str) -> list[str]:
    """
    Full process to download the paper, extract text, and delete the PDF.
    """
    pdf_path = "paper.pdf"  # Temporary file to store the downloaded PDF

    # Step 1: Download the PDF
    if not download_pdf(arxiv_id, pdf_path):
        return []  # Return empty if download fails

    # Step 2: Extract text from the PDF
    pages_text = extract_text_by_page(pdf_path)

    # Step 3: Delete the PDF file after extracting text
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        print(f"PDF file deleted: {pdf_path}")

    return pages_text


def detect_section_start(pages_text: list[str], section_keywords: list[str], min_page_length: int = 100) -> int:
    """
    Detects the page index where references or supplementary sections start.
    Uses formatting and keyword analysis.
    """
    for i, page_text in enumerate(pages_text):
        # Check for keywords near the start of the page (section headers)
        if any(page_text.strip().lower().startswith(keyword) for keyword in section_keywords):
            return i

        # Heuristic: detect sudden shift to citation-heavy text
        lines = page_text.splitlines()
        citation_count = sum(1 for line in lines if "[" in line or "]" in line or "et al" in line)
        if citation_count > 0.3 * len(lines):  # If >30% of lines are citations
            return i

        # Heuristic: detect formatting typical of references
        if len(page_text) < min_page_length:  # Short pages often signal non-main content
            return i

    return len(pages_text)  # If no cutoff section is found, return the entire document


def remove_citations_and_supplements(pages_text: list[str]) -> list[str]:
    """
    Removes references and supplementary sections from the text.
    """
    section_keywords = [
        "references",
        "bibliography",
        "appendix",
        "supplementary material",
        "acknowledgments",
        "supplement",
        "citations",
    ]
    cutoff_index = detect_section_start(pages_text, section_keywords)
    return pages_text[:cutoff_index]


def process_arxiv_paper_with_embeddings(arxiv_id: str, topic_model: BERTopic) -> list[dict]:
    """
    Processes an arXiv paper, splits it into pages, removes references/supplements,
    and returns text and embeddings for each page.
    """
    # Step 1: Download and read the paper
    pdf_path = f"{arxiv_id}.pdf"
    downloaded = download_pdf(arxiv_id, pdf_path)

    if not downloaded:
        return []

    try:
        # Open the PDF and extract text per page
        with fitz.open(pdf_path) as pdf:
            pages_text = [pdf[i].get_text() for i in range(len(pdf))]
    finally:
        # Clean up the PDF after processing
        os.remove(pdf_path)

    # Step 2: Remove citations and supplementary sections
    filtered_pages = remove_citations_and_supplements(pages_text)

    # Step 3: Generate embeddings for the filtered pages
    embedding_model = topic_model.embedding_model
    embeddings = embedding_model.embedding_model.encode(filtered_pages, show_progress_bar=True)

    # Combine pages and embeddings into a structured format
    result = [{"text": text, "embedding": embedding} for text, embedding in zip(filtered_pages, embeddings)]
    return result


# =============================================================================
# from bertopic import BERTopic
#
# # Example Usage
# arxiv_id = "2301.12345"  # Replace with the actual paper ID
# topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")  # Load the pre-trained BERTopic model
#
# result = process_arxiv_paper_with_embeddings(arxiv_id, topic_model)
#
# if result:
#     print("First page text:", result[0]["text"])  # Text of the first page
#     print("First page embedding:", result[0]["embedding"])  # Embedding of the first page
# else:
#     print("No content remains after filtering.")
#
# =============================================================================


def find_most_relevant_pages(
    relevant_pages: list[dict], abstracts: list[str], paper_count_limit: int
) -> dict[str, dict]:
    """
    Identifies all pages for the top {paper_count_limit} papers. (i.e. if there's multiple pages of one paper that are
     deemed relevant, all pages will be included)
    :param relevant_pages:
    :param abstracts: List of paper abstracts
    :param paper_count_limit: Max number of source papers of all pages that are returned
    :return: Dictionary consisting of paper_id: dict pairs
    """
    if paper_count_limit > len(relevant_pages):
        paper_count_limit = len(relevant_pages)

    page_ids = set()
    output = {}
    index = 0
    while len(page_ids) < paper_count_limit:
        paper_id = relevant_pages[index]["paper_id"]
        if paper_id in page_ids:
            output[paper_id]["text"].append(relevant_pages[index]["text"])
        else:
            tmp = {
                "abstract": abstracts[paper_id],
                "text": [relevant_pages[index]["text"]],
            }
            output[paper_id] = tmp
        page_ids.add(paper_id)
        index += 1
    return output


def filter_citations(related_works_section: str, citation_strings: list[str]) -> list[str]:
    matched_citations = set()
    for citation in citation_strings:
        split_citation = citation.split(" ")
        if len(split_citation) > 2:
            first_author = split_citation[1]
            if first_author[-1] == ",":
                first_author = first_author[:-1]
            if f" {first_author}" in related_works_section or f"{first_author} " in related_works_section:
                matched_citations.add(citation)

    return list(matched_citations)
