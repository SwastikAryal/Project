import json
import math
import os
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def detect_language(text: str) -> str:
    """Detect if text is primarily Nepali or English."""
    nepali_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    if nepali_chars > english_chars:
        return "nepali"
    return "english"


@dataclass
class SearchResult:
    rank: int
    id: str
    text: str
    section: str
    page: str
    score: float
    language: str


class ManualEmbeddingModel:
    """Wrapper for sentence-transformers with caching."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._initialized = True
        self._cache = {}
        logger.info(f"Loaded multilingual embedding model (dim={self.dimension})")

    def encode(self, text: str) -> List[float]:
        key = text.lower().strip()
        if key in self._cache:
            return self._cache[key]
        embedding = self.model.encode(text).tolist()
        if len(self._cache) < 1000:
            self._cache[key] = embedding
        return embedding

    def clear_cache(self):
        self._cache = {}


class ManualVectorRetriever:
    """Brute-force k-NN with Cosine Similarity. No vector DB."""

    def __init__(self, vectors_path: str = "manual_vectors.json"):
        self.vectors_path = vectors_path
        self.documents: List[Dict] = []
        self.dimension: int = 0
        self.embedding_model = ManualEmbeddingModel()
        self._load()

    def _load(self):
        if not os.path.exists(self.vectors_path):
            raise FileNotFoundError(
                f"Vector store not found: {self.vectors_path}\nRun: python build_vector_store.py"
            )

        with open(self.vectors_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.dimension = data["metadata"]["dimension"]
        eng = sum(1 for d in self.documents if d.get("language") == "english")
        nep = sum(1 for d in self.documents if d.get("language") == "nepali")
        logger.info(f"Loaded {len(self.documents)} docs ({eng} en, {nep} ne)")

    def _cosine_similarity(self, query_vec, doc_vec, doc_norm):
        dot = sum(q * d for q, d in zip(query_vec, doc_vec))
        q_norm = math.sqrt(sum(q * q for q in query_vec))
        if q_norm == 0 or doc_norm == 0:
            return 0.0
        return dot / (q_norm * doc_norm)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.2,
        prefer_language: str = None
    ) -> List[SearchResult]:
        """
        Search with optional language preference.
        If prefer_language is set, boost scores for matching language.
        """
        query_embedding = self.embedding_model.encode(query)
        scored = []

        for doc in self.documents:
            score = self._cosine_similarity(
                query_embedding, doc["embedding"], doc["norm"]
            )

            # Boost score for matching language
            if prefer_language and doc.get("language") == prefer_language:
                score *= 1.15

            if score >= threshold:
                scored.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "section": doc["section"],
                    "page": doc["page"],
                    "language": doc.get("language", "unknown"),
                    "score": score
                })

        scored.sort(key=lambda x: x["score"], reverse=True)

        results = []
        for rank, item in enumerate(scored[:top_k], 1):
            results.append(SearchResult(
                rank=rank,
                id=item["id"],
                text=item["text"],
                section=item["section"],
                page=item["page"],
                score=round(item["score"], 4),
                language=item["language"]
            ))
        return results


class ConversationMemory:
    """Session-based conversation history."""

    def __init__(self, max_turns: int = 10):
        self.turns: List[Dict] = []
        self.max_turns = max_turns

    def add(self, query: str, results: List[SearchResult], answer: str):
        self.turns.append({
            "query": query,
            "retrieved_sections": [r.section for r in results],
            "answer": answer
        })
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context(self) -> str:
        if not self.turns:
            return ""
        context = "Previous conversation:\n"
        for turn in self.turns[-3:]:
            context += f"Q: {turn['query'][:100]}\nA: {turn['answer'][:150]}\n\n"
        return context

    def clear(self):
        self.turns = []


class StrictLegalRAG:
    """
    Bilingual Legal RAG System.
    - Answers in the same language as the question
    - No hallucination: only from retrieved context
    - Manual vector search
    """

    NOT_AVAILABLE_EN = "The answer is not available in the provided data."
    NOT_AVAILABLE_NP = "तपाईंको प्रश्नको जवाफ उपलब्ध छैन।"

    # Legal topic keywords for answerability check
    LEGAL_TOPICS = {
        "marriage": {
            "en": ["marriage", "matrimonial", "husband", "wife", "spouse", "wedding", "conjugal"],
            "np": ["विवाह", "विवाहको", "पति", "पत्नी", "दाम्पत्य", "सहवास", "वर", "वधु", "पति पत्नी"]
        },
        "divorce": {
            "en": ["divorce", "separation", "dissolve", "terminate"],
            "np": ["विच्छेद", "सम्बन्ध विच्छेद", "सम्बन्ध विच्छेदको"]
        },
        "property": {
            "en": ["property", "partition", "inheritance", "succession", "land", "ownership", "possession"],
            "np": ["सम्पत्ति", "जग्गा", "जमिन", "अंशबण्डा", "सम्पदा", "अचल", "स्वामित्व"]
        },
        "adoption": {
            "en": ["adoption", "adopt", "adopted", "adoptive"],
            "np": ["धमनपुर", "धमनपुरी", "धमयपुर", "धमयपुरी", "दत्तक"]
        },
        "guardian": {
            "en": ["guardian", "custody", "minor", "child", "parent", "curatorship", "care"],
            "np": ["संरक्षक", "संरक्षकत्व", "नाबालक", "बालबालिका", "बालक", "अभिभावक", "हेरचाह", "स्याहार", "सम्भार", "हेर्चहा", "बाच्चा", "बच्चा", "बाल"]
        },
        "succession": {
            "en": ["succession", "heir", "inherit", "will", "testament"],
            "np": ["वंशानुगत", "उत्तराधिकारी", "बाँडफाँड", "इच्छापत्र"]
        },
        "contract": {
            "en": ["contract", "agreement", "obligation", "party"],
            "np": ["सम्झौता", "करार", "दायित्व", "सम्झौतापत्र"]
        },
        "court": {
            "en": ["court", "lawsuit", "petition", "judge", "legal"],
            "np": ["अदालत", "मुद्दा", "याचिका", "न्यायाधीश", "कानूनी"]
        }
    }

    def __init__(self, vectors_path: str = "manual_vectors.json"):
        self.retriever = ManualVectorRetriever(vectors_path)
        self.memory = ConversationMemory()
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_key = GROQ_API_KEY

        if not self.groq_key:
            logger.warning("GROQ_API_KEY not set. Using local fallback.")

    def _format_context(self, results: List[SearchResult]) -> str:
        """Format retrieved docs for prompt with full text."""
        contexts = []
        for r in results:
            label = "ENGLISH" if r.language == "english" else "नेपाली"
            text = r.text[:1000] + "..." if len(r.text) > 1000 else r.text
            contexts.append(f"[{label} | {r.section} | Page {r.page}]:\n{text}")
        return "\n\n---\n\n".join(contexts)

    def _is_answerable(self, query: str, context: str, lang: str, top_score: float = 0.0) -> bool:
        """Check if query can be answered from context. Bilingual."""
        if top_score >= 0.5:
            return True

        query_lower = query.lower()
        context_lower = context.lower()

        for topic, keywords in self.LEGAL_TOPICS.items():
            all_keywords = keywords["en"] + keywords["np"]
            for kw in all_keywords:
                if kw in query_lower:
                    if any(k in context_lower for k in all_keywords):
                        return True
                    break

        if lang == "nepali":
            query_chars = set(query_lower)
            context_chars = set(context_lower)
            common = query_chars & context_chars
            nepali_common = {c for c in common if '\u0900' <= c <= '\u097F'}
            if len(nepali_common) >= 5:
                return True

        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        overlap = query_words & context_words
        min_overlap = 1 if len(query_words) <= 3 else 2
        return len(overlap) >= min_overlap

    def _build_prompts(self, query: str, context: str, lang: str, history: str):
        """Build language-specific system and user prompts for detailed answers."""
        if lang == "nepali":
            system_prompt = (
                "You are a legal expert for the Muluki Civil Code of Nepal (मुलुकी देवानी संहिता).\n"
                "तपाईंलाई अंग्रेजी र नेपाली दुवै भाषामा कानूनी पाठ दिइएको छ। "
                "अंग्रेजी पाठ स्पष्ट र सही छ, नेपाली पाठमा OCR त्रुटि हुन सक्छ।\n\n"
                "तपाईंको जवाफ नेपाली भाषामा दिनुहोस् र निम्न नियमहरू पालना गर्नुहोस्:\n"
                "1. जवाफ विस्तृत, संरचित र बुझ्न सजिलो हुनुपर्छ\n"
                "2. प्रत्येक मुख्य बिन्दुलाई अलग अलग बुलेट वा नम्बर गर्नुहोस्\n"
                "3. प्रत्येक बिन्दुमा सम्बन्धित दफा नम्बर र त्यसको विवरण उद्धृत गर्नुहोस्\n"
                "4. कानूनी प्रावधानको अर्थ सरल नेपालीमा व्याख्या गर्नुहोस्\n"
                "5. यदि प्रश्नको जवाफ पाठमा छैन भने मात्र भन्नुहोस्: \"तपाईंको प्रश्नको जवाफ उपलब्ध छैन।\"\n"
                "6. बाहिरी ज्ञान प्रयोग नगर्नुहोस्"
            )
            not_found = self.NOT_AVAILABLE_NP
            user_prompt = (
                f"{history}तलको कानूनी पाठहरूमा आधारित भएर मात्र जवाफ दिनुहोस्:\n\n"
                f"{context}\n\n"
                f"प्रश्न: {query}\n\n"
                f"विस्तृत र संरचित जवाफ दिनुहोस्। प्रत्येक बिन्दुमा दफा नम्बर र त्यसको विवरण उल्लेख गर्नुहोस्।"
            )
        else:
            system_prompt = (
                "You are a legal expert for the Muluki Civil Code of Nepal.\n"
                "Your answer must be in English and follow these rules:\n"
                "1. Provide a detailed, well-structured answer that is easy to understand\n"
                "2. Use numbered points for each main topic\n"
                "3. For each point, cite the specific Section number AND explain what that section says\n"
                "4. Explain the legal provisions in plain, accessible language\n"
                "5. If the answer is not in the text, respond EXACTLY: "
                "\"The answer is not available in the provided data.\"\n"
                "6. Never use outside knowledge"
            )
            not_found = self.NOT_AVAILABLE_EN
            user_prompt = (
                f"{history}Based ONLY on the following legal text:\n\n"
                f"{context}\n\n"
                f"Question: {query}\n\n"
                f"Provide a detailed, structured answer. For each point, cite the Section number "
                f"and explain what that section says about the topic."
            )

        return system_prompt, user_prompt, not_found

    def _query_groq(self, system_prompt: str, user_prompt: str) -> str:
        """Call Groq API for detailed generation."""
        if not self.groq_key:
            return self._local_answer(user_prompt)

        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1500
        }

        try:
            response = requests.post(
                self.groq_url, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._local_answer(user_prompt)

    def _local_answer(self, prompt: str) -> str:
        """Fallback: extract answer from context."""
        match = re.search(
            r'(?:Based ONLY|आधार).*?(?:legal text|कानूनी पाठ).*?:\n\n(.*?)\n\n(?:Question|प्रश्न)',
            prompt, re.DOTALL
        )
        if not match:
            return self.NOT_AVAILABLE_EN

        context = match.group(1)
        sections = context.split("\n\n")
        if not sections:
            return self.NOT_AVAILABLE_EN

        return f"Based on the legal text:\n\n{sections[0][:1000]}"

    def _validate_citations(self, answer: str, results: List[SearchResult], lang: str) -> bool:
        """Check if answer citations match retrieved context. Bilingual."""
        if "not available" in answer.lower() or "उपलब्ध छैन" in answer:
            return True

        def normalize_numbers(text: str) -> set:
            """Convert all numbers to Arabic for comparison."""
            # Devanagari to Arabic mapping
            dev_to_arb = str.maketrans('०१२३४५६७८९', '0123456789')
            normalized = text.translate(dev_to_arb)
            return set(re.findall(r'\d+', normalized))

        cited_nums = normalize_numbers(answer)
        context_nums = set()
        for r in results:
            context_nums.update(normalize_numbers(r.section))

        if cited_nums:
            return bool(cited_nums & context_nums)

        return len(answer) > 50

    # Nepali to English topic mapping for cross-language search
    NP_TO_EN_QUERY = {
        'बच्चा': 'child care custody',
        'बाच्चा': 'child care custody',
        'बालबालिका': 'child minor custody guardian',
        'बालक': 'child minor',
        'नाबालक': 'minor child guardian',
        'विवाह': 'marriage',
        'विच्छेद': 'divorce separation',
        'सम्पत्ति': 'property',
        'धमनपुर': 'adoption',
        'धमनपुरी': 'adoption',
        'संरक्षक': 'guardian curatorship',
        'हेरचाह': 'care custody guardian',
        'हेर्चहा': 'care custody guardian',
        'अंशबण्डा': 'partition',
    }

    def answer(self, query: str) -> Dict:
        """Main entry: retrieve and generate bilingual answer."""
        lang = detect_language(query)

        # 1. Search in matching language
        results = self.retriever.search(
            query, top_k=5, threshold=0.15, prefer_language=lang
        )

        # 2. For Nepali queries, also search English for clean context
        if lang == "nepali":
            # Build English search query from Nepali keywords
            en_keywords = []
            query_lower = query.lower()
            for np_word, en_word in self.NP_TO_EN_QUERY.items():
                if np_word in query_lower:
                    en_keywords.append(en_word)

            if en_keywords:
                en_query = " ".join(set(en_keywords))
                en_results = self.retriever.search(
                    en_query, top_k=3, threshold=0.15, prefer_language="english"
                )
                # Add unique English results
                seen_ids = {r.id for r in results}
                for r in en_results:
                    if r.id not in seen_ids:
                        results.append(r)
                        seen_ids.add(r.id)
                # Keep top 7 total (5 Nepali + up to 2 English)
                results.sort(key=lambda x: x.score, reverse=True)
                results = results[:7]
                for i, r in enumerate(results):
                    r.rank = i + 1

        if not results:
            not_found = self.NOT_AVAILABLE_NP if lang == "nepali" else self.NOT_AVAILABLE_EN
            return {
                "answer": not_found,
                "sources": [],
                "language": lang,
                "method": "no_results"
            }

        # 2. Format context
        context = self._format_context(results)

        # 3. Check answerability
        top_score = results[0].score if results else 0.0
        if not self._is_answerable(query, context, lang, top_score):
            not_found = self.NOT_AVAILABLE_NP if lang == "nepali" else self.NOT_AVAILABLE_EN
            return {
                "answer": not_found,
                "sources": [
                    {"section": r.section, "page": r.page, "score": r.score, "language": r.language}
                    for r in results[:3]
                ],
                "language": lang,
                "method": "low_relevance"
            }

        # 4. Build prompts
        history = self.memory.get_context()
        system_prompt, user_prompt, not_found = self._build_prompts(query, context, lang, history)

        # 5. Generate
        answer = self._query_groq(system_prompt, user_prompt)

        # 6. Validate citations and clean up mixed responses
        if not self._validate_citations(answer, results, lang):
            answer = not_found
        else:
            # Remove trailing "not available" if answer already has content
            if lang == "nepali":
                if "तपाईंको प्रश्नको जवाफ उपलब्ध छैन।" in answer:
                    parts = answer.split("तपाईंको प्रश्नको जवाफ उपलब्ध छैन।")
                    actual_answer = parts[0].strip()
                    if len(actual_answer) > 50:
                        answer = actual_answer
            else:
                if "not available in the provided data" in answer.lower():
                    parts = re.split(r'(?i)the answer is not available in the provided data\.?', answer)
                    actual_answer = parts[0].strip()
                    if len(actual_answer) > 50:
                        answer = actual_answer

        # 7. Store in memory
        self.memory.add(query, results, answer)

        return {
            "answer": answer,
            "sources": [
                {
                    "rank": r.rank,
                    "section": r.section,
                    "page": r.page,
                    "score": r.score,
                    "language": r.language,
                    "preview": r.text[:200] + "..."
                }
                for r in results
            ],
            "retrieved_count": len(results),
            "language": lang,
            "method": "manual_cosine_similarity"
        }

    def reset(self):
        self.memory.clear()
        return {"status": "reset"}


def main():
    """Interactive CLI."""
    print("Legal RAG - Bilingual Manual Vector Search")
    print("=" * 50)
    print("Nepali and English supported")
    print("=" * 50)

    rag = StrictLegalRAG()

    while True:
        query = input("\nQuestion (or 'quit'): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break
        if query == 'reset':
            rag.reset()
            print("Reset")
            continue
        if not query:
            continue

        print("Searching...")
        result = rag.answer(query)

        print(f"\n[{result['language'].upper()}] Answer:")
        print(result["answer"])

        print(f"\nSources ({result['retrieved_count']} retrieved):")
        for s in result["sources"][:3]:
            print(f"  #{s['rank']} {s['section']} (p.{s['page']}) [{s.get('language','')}] score:{s['score']}")


if __name__ == "__main__":
    main()
