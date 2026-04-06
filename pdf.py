import json
import re
import os
import pdfplumber
from typing import List, Dict, Any, Optional


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from every page, preserving all content."""
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "page_num": page_num,
                    "text": text.strip()
                })
    return pages_data


def detect_language(text: str) -> str:
    """Detect if text is primarily Nepali or English."""
    nepali_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    if nepali_chars > english_chars:
        return "nepali"
    return "english"


def merge_pages(pages_data: List[Dict[str, Any]]) -> str:
    """Merge all pages into one continuous text, joining split lines."""
    raw_text = "\n".join(page["text"] for page in pages_data)
    
    # Fix common PDF extraction issues:
    # 1. Join hyphenated words split across lines: "some-\nthing" -> "something"
    raw_text = re.sub(r'-\n(\w)', r'\1', raw_text)
    
    # 2. Join lines that are clearly continuations (not starting with section markers)
    lines = raw_text.split('\n')
    merged_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            merged_lines.append('')
            continue
        
        # If previous line exists and doesn't end with sentence-ending punctuation,
        # and current line doesn't start with a section/chapter marker, join them
        if merged_lines and merged_lines[-1]:
            prev = merged_lines[-1]
            is_section_start = re.match(
                r'^(Section\s+\d+|दफा\s+|Chapter-|Part-|भाग\s+|परिच्छेद)',
                stripped
            )
            prev_ends_sentence = prev.rstrip().endswith(('.', ':', ';', ')', '?', '!'))
            
            if not is_section_start and not prev_ends_sentence:
                # Join with previous line (remove newline)
                merged_lines[-1] = prev.rstrip() + ' ' + stripped
                continue
        
        merged_lines.append(stripped)
    
    return '\n'.join(merged_lines)


def split_by_sections(text: str, language: str) -> List[Dict[str, Any]]:
    """Split text into sections based on language-specific markers."""
    if language == "english":
        # Split at "Section N:" or "Chapter-N" or "Part-N"
        pattern = r'(?=\n?(Section\s+\d+:|Chapter-\d+|Part-\d+))'
    else:
        # Split at "दफा N:" or "परिच्छेद" or "भाग"
        pattern = r'(?=\n?(दफा\s+|परिच्छेद|भाग))'
    
    parts = re.split(pattern, text)
    
    sections = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        
        # Check if this is a section/chapter/part header
        if language == "english":
            header_match = re.match(
                r'(Section\s+\d+:|Chapter-\d+|Part-\d+)[^\n]*', part
            )
        else:
            header_match = re.match(
                r'(दफा\s+\S+:|परिच्छेद[^\n]*|भाग[^\n]*)', part
            )
        
        if header_match:
            title = header_match.group(0).strip()
            content = part.strip()
            
            # For section markers, the next part might be the content
            # (split pattern consumed the marker, content follows)
            sections.append({
                "title": title,
                "content": content,
            })
        elif len(part) > 20:
            # Standalone content without a clear marker (intro text, etc.)
            sections.append({
                "title": "Preamble",
                "content": part,
            })
        
        i += 1
    
    return sections


def get_section_page(page_positions: List[int], char_start: int, char_end: int) -> str:
    """Determine page range for a character range."""
    if not page_positions:
        return "1"
    
    start_page = 1
    end_page = 1
    
    for i, (pos, page_num) in enumerate(page_positions):
        if char_start >= pos:
            start_page = page_num
        if char_end >= pos:
            end_page = page_num
    
    return str(start_page) if start_page == end_page else f"{start_page}-{end_page}"


def build_page_position_map(pages_data: List[Dict[str, Any]]) -> List[tuple]:
    """Build a map of character positions to page numbers."""
    positions = []
    char_pos = 0
    for page in pages_data:
        positions.append((char_pos, page["page_num"]))
        char_pos += len(page["text"]) + 1  # +1 for newline
    return positions


def create_chunks(
    pages_data: List[Dict[str, Any]],
    doc_name: str,
    language: str,
    max_chunk_size: int = 1500,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Create chunks from PDF pages, preserving every word.
    Splits by sections, then by size if needed with overlap.
    """
    full_text = merge_pages(pages_data)
    page_map = build_page_position_map(pages_data)
    sections = split_by_sections(full_text, language)
    
    chunks = []
    chunk_id = 0
    char_offset = 0
    
    for section in sections:
        title = section["title"]
        content = section["content"]
        
        if not content or len(content.strip()) < 10:
            char_offset += len(content) + 1
            continue
        
        # Determine page number
        section_start = full_text.find(content[:50], max(0, char_offset - 100))
        section_end = section_start + len(content) if section_start >= 0 else char_offset + len(content)
        page_str = get_section_page(page_map, section_start, section_end)
        
        if len(content) <= max_chunk_size:
            # Section fits in one chunk
            chunk_id += 1
            chunks.append({
                "chunk_id": chunk_id,
                "text": content,
                "metadata": {
                    "doc": doc_name,
                    "page": page_str,
                    "section": title,
                    "language": language
                }
            })
        else:
            # Split large section into overlapping chunks
            start = 0
            while start < len(content):
                end = min(start + max_chunk_size, len(content))
                
                # Try to break at sentence boundary
                if end < len(content):
                    # Look for sentence end (. or । for Nepali)
                    for break_char in ['।', '.', '\n']:
                        last_break = content.rfind(break_char, start + max_chunk_size // 2, end)
                        if last_break != -1:
                            end = last_break + 1
                            break
                
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunk_id += 1
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": {
                            "doc": doc_name,
                            "page": page_str,
                            "section": title,
                            "language": language
                        }
                    })
                
                # Advance with overlap, ensure progress
                next_start = end - overlap
                if next_start <= start:
                    next_start = start + max_chunk_size - overlap
                start = next_start
        
        char_offset += len(content) + 1
    
    return chunks


def process_document(
    pdf_path: str,
    doc_name: str,
    language: str,
    output_path: str = None,
    max_chunk_size: int = 1500
) -> List[Dict[str, Any]]:
    """Process a single PDF document into chunks."""
    pages_data = extract_text_from_pdf(pdf_path)
    if not pages_data:
        print(f"  WARNING: No text extracted from {pdf_path}")
        return []
    
    print(f"  Extracted {len(pages_data)} pages")
    chunks = create_chunks(pages_data, doc_name, language, max_chunk_size)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    return chunks


def get_chunks_for_vector_store(pdf_path: str, doc_name: str, language: str) -> list:
    """Convenience function to get chunks for vector store."""
    return process_document(
        pdf_path=pdf_path,
        doc_name=doc_name,
        language=language,
        output_path=None,
        max_chunk_size=1500
    )


if __name__ == "__main__":
    pdf_files = [
        {
            "path": "Muluki civil code.pdf",
            "name": "Muluki Civil Code - English",
            "language": "english"
        },
        {
            "path": "मुलुकी देवानी संहिता ऐन.pdf",
            "name": "Muluki Civil Code - Nepali",
            "language": "nepali"
        },
    ]
    
    all_chunks = []
    
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file["path"]):
            print(f"Warning: {pdf_file['path']} not found, skipping...")
            continue
        
        print(f"\nProcessing: {pdf_file['name']} ({pdf_file['language']})...")
        chunks = process_document(
            pdf_path=pdf_file["path"],
            doc_name=pdf_file["name"],
            language=pdf_file["language"],
            output_path=None,
            max_chunk_size=1500
        )
        all_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks")
    
    if all_chunks:
        with open("code_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        eng_count = sum(1 for c in all_chunks if c["metadata"]["language"] == "english")
        nep_count = sum(1 for c in all_chunks if c["metadata"]["language"] == "nepali")
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        print(f"  English: {eng_count}")
        print(f"  Nepali: {nep_count}")
        print(f"Saved to code_chunks.json")
    else:
        print("No chunks created - check if PDF files exist")
