import fitz  # PyMuPDF for PDF reading
from docx import Document  # For reading DOCX
import re
from collections import Counter

# ----------- Extract text from PDF or DOCX ----------- #
def extract_text(file, file_type):
    """
    Extract raw text from a PDF or DOCX file object.
    
    Parameters:
        file: Uploaded file object
        file_type: 'pdf' or 'docx'
    
    Returns:
        A string containing the extracted text.
    """
    if file_type == 'pdf':
        # Read PDF using PyMuPDF
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return ''.join([page.get_text() for page in doc])
    
    elif file_type == 'docx':
        # Read DOCX using python-docx
        document = Document(file)
        return '\n'.join([para.text for para in document.paragraphs])
    
    return ""

# ----------- Extract top keywords from text ----------- #
def extract_keywords(text, num_keywords=30):
    """
    Extract top keywords from the input text.
    
    Parameters:
        text: Raw resume text
        num_keywords: Number of keywords to return
    
    Returns:
        A list of most common keywords (strings).
    """
    # Basic tokenization
    words = re.findall(r'\b\w+\b', text.lower())

    # Simple stop word list
    stop_words = set([
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "you", "your", "but", "not", "have", "has", "been", "can", "will",
        "had", "they", "their", "what", "when", "how", "who", "why", "where",
        "all", "any", "she", "him", "her", "its", "also", "our", "more", "such",
        "on", "in", "as", "at", "if", "or", "an", "be", "by", "is", "of", "to", "a"
    ])

    # Filter and count
    filtered = [word for word in words if word not in stop_words and len(word) > 2]
    return [kw for kw, _ in Counter(filtered).most_common(num_keywords)]
