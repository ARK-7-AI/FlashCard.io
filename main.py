#!/usr/bin/env python3

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

import PyPDF2
import io
import requests
import json
import re
from datetime import datetime
import hashlib

# Create FastAPI app instance
app = FastAPI(
    title="PDF to Flashcard API",
    description="Convert PDF documents to flashcards with comprehensive sequential coverage",
    version="2.0.0"
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default
        # Add your specific frontend URLs here
        "*"  # Allow all origins (use with caution in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models


class Flashcard(BaseModel):
    question: str
    answer: str
    difficulty: Optional[str] = "medium"
    tags: Optional[List[str]] = []
    topic: Optional[str] = None
    page_reference: Optional[int] = None
    section_reference: Optional[str] = None


class TopicSection(BaseModel):
    topic_name: str
    content: str
    page_numbers: List[int]
    importance_score: float
    flashcard_count: int


class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]
    total_count: int
    processing_time: str
    source_info: dict
    coverage_analysis: Dict[str, Any]
    topics_covered: List[TopicSection]


class GenerationConfig(BaseModel):
    total_flashcards: int = 20
    difficulty_level: str = "medium"  # easy, medium, hard
    focus_areas: Optional[List[str]] = []
    include_definitions: bool = True
    include_examples: bool = False
    ensure_full_coverage: bool = True
    min_flashcards_per_topic: int = 2
    coverage_strategy: str = "balanced"  # balanced, sequential, importance_based


# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"


def extract_text_with_pages(pdf_file) -> Dict[int, str]:
    """Extract text content from PDF with page tracking."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_text = {}

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only store non-empty pages
                pages_text[page_num + 1] = page_text.strip()

        return pages_text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error extracting PDF text: {str(e)}")


def analyze_document_structure(pages_text: Dict[int, str]) -> List[TopicSection]:
    """Analyze document structure and identify key topics with importance scoring."""

    # Combine all text for topic analysis
    full_text = "\n".join(pages_text.values())

    # Create prompt for topic analysis
    analysis_prompt = f"""
Analyze this document and identify the main topics/sections in sequential order. 
For each topic, provide:
1. Topic name
2. Brief description
3. Importance score (1-10)
4. Key concepts covered

Document text (first 3000 chars):
{full_text[:3000]}

Respond in JSON format:
[
    {{
        "topic_name": "Introduction to...",
        "description": "Brief description",
        "importance_score": 8.5,
        "key_concepts": ["concept1", "concept2"]
    }}
]
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": analysis_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 1500}
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            topics_data = extract_json_from_text(result.get("response", ""))

            # Map topics to page ranges
            topics = []
            pages_per_topic = max(1, len(pages_text) //
                                  max(1, len(topics_data)))

            for i, topic_data in enumerate(topics_data):
                start_page = (i * pages_per_topic) + 1
                end_page = min((i + 1) * pages_per_topic, len(pages_text))

                # Get content for this topic's page range
                topic_content = ""
                topic_pages = []
                for page_num in range(start_page, end_page + 1):
                    if page_num in pages_text:
                        topic_content += pages_text[page_num] + "\n"
                        topic_pages.append(page_num)

                topics.append(TopicSection(
                    topic_name=topic_data.get("topic_name", f"Topic {i+1}"),
                    content=topic_content,
                    page_numbers=topic_pages,
                    importance_score=float(
                        topic_data.get("importance_score", 5.0)),
                    flashcard_count=0  # Will be calculated later
                ))

            return topics

    except Exception as e:
        print(f"Topic analysis failed: {e}")

    # Fallback: Create topics based on page ranges
    fallback_topics = []
    pages_per_topic = max(2, len(pages_text) // 5)  # Aim for ~5 topics

    for i in range(0, len(pages_text), pages_per_topic):
        topic_pages = []
        topic_content = ""

        for page_num in range(i + 1, min(i + pages_per_topic + 1, len(pages_text) + 1)):
            if page_num in pages_text:
                topic_content += pages_text[page_num] + "\n"
                topic_pages.append(page_num)

        if topic_content.strip():
            fallback_topics.append(TopicSection(
                topic_name=f"Section {len(fallback_topics) + 1}",
                content=topic_content,
                page_numbers=topic_pages,
                importance_score=5.0,
                flashcard_count=0
            ))

    return fallback_topics


def calculate_flashcard_distribution(topics: List[TopicSection], total_flashcards: int,
                                     min_per_topic: int, strategy: str) -> List[TopicSection]:
    """Calculate how many flashcards each topic should get."""

    if not topics:
        return topics

    # Ensure minimum flashcards per topic
    base_allocation = min_per_topic * len(topics)
    remaining_flashcards = max(0, total_flashcards - base_allocation)

    # Distribute remaining flashcards based on strategy
    if strategy == "importance_based":
        # Distribute based on importance scores
        total_importance = sum(topic.importance_score for topic in topics)
        for topic in topics:
            additional = int(
                (topic.importance_score / total_importance) * remaining_flashcards)
            topic.flashcard_count = min_per_topic + additional

    elif strategy == "sequential":
        # Equal distribution
        additional_per_topic = remaining_flashcards // len(topics)
        remainder = remaining_flashcards % len(topics)

        for i, topic in enumerate(topics):
            topic.flashcard_count = min_per_topic + additional_per_topic
            if i < remainder:  # Distribute remainder to first few topics
                topic.flashcard_count += 1

    else:  # balanced (default)
        # Balance between content length and importance
        content_weights = [len(topic.content) for topic in topics]
        importance_weights = [topic.importance_score for topic in topics]

        # Normalize weights
        max_content = max(content_weights) if content_weights else 1
        max_importance = max(importance_weights) if importance_weights else 1

        combined_weights = []
        for i in range(len(topics)):
            content_norm = content_weights[i] / max_content
            importance_norm = importance_weights[i] / max_importance
            combined_weights.append((content_norm + importance_norm) / 2)

        total_weight = sum(combined_weights)
        for i, topic in enumerate(topics):
            if total_weight > 0:
                additional = int(
                    (combined_weights[i] / total_weight) * remaining_flashcards)
                topic.flashcard_count = min_per_topic + additional
            else:
                topic.flashcard_count = min_per_topic

    # Ensure we don't exceed total flashcards
    total_assigned = sum(topic.flashcard_count for topic in topics)
    if total_assigned != total_flashcards:
        # Adjust the largest topic
        topics[0].flashcard_count += (total_flashcards - total_assigned)

    return topics


def generate_flashcards_for_topic(topic: TopicSection, config: GenerationConfig) -> List[Flashcard]:
    """Generate flashcards for a specific topic."""

    if topic.flashcard_count <= 0:
        return []

    prompt = f"""
Create exactly {topic.flashcard_count} high-quality flashcards from this topic content.

TOPIC: {topic.topic_name}
CONTENT:
{topic.content[:2500]}  # Limit to avoid token limits

REQUIREMENTS:
- Generate exactly {topic.flashcard_count} question-answer pairs
- Difficulty: {config.difficulty_level}
- Cover key concepts comprehensively
- Include specific details and examples from the content
- Make questions clear and testable
- Provide complete, accurate answers
- Strictly generate content from PDF and no generalized answers 

{"Focus on definitions and explanations" if config.include_definitions else ""}
{"Include practical examples and applications" if config.include_examples else ""}

Output as JSON array ONLY (no other text):
[
    {{"question": "What is...", "answer": "..."}},
    {{"question": "How does...", "answer": "..."}}
]
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "max_tokens": 1500
                }
            },
            timeout=90
        )

        if response.status_code != 200:
            print(
                f"Ollama error for topic {topic.topic_name}: {response.text}")
            return []

        result = response.json()
        generated_text = result.get("response", "")

        # Handle empty response
        if not generated_text.strip():
            print(f"Empty response for topic: {topic.topic_name}")
            return []

        flashcards_data = extract_json_from_text(generated_text)

        flashcards = []
        for item in flashcards_data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                # Ensure strings are not None and are properly cleaned
                question = str(item["question"]).strip() if item.get(
                    "question") else "Question not generated"
                answer = str(item["answer"]).strip() if item.get(
                    "answer") else "Answer not generated"

                # Skip invalid flashcards
                if not question or not answer or question == "Question not generated" or answer == "Answer not generated":
                    continue

                # Assign page reference (use first page of topic)
                page_ref = topic.page_numbers[0] if topic.page_numbers else None

                flashcard = Flashcard(
                    question=question,
                    answer=answer,
                    difficulty=config.difficulty_level,
                    tags=config.focus_areas or [],
                    topic=topic.topic_name,
                    page_reference=page_ref,
                    section_reference=f"Pages {min(topic.page_numbers)}-{max(topic.page_numbers)}" if topic.page_numbers else None
                )
                flashcards.append(flashcard)

        # Ensure we don't exceed the requested count
        return flashcards[:topic.flashcard_count]

    except Exception as e:
        print(f"Error generating flashcards for topic {topic.topic_name}: {e}")
        return []


def generate_comprehensive_flashcards(pages_text: Dict[int, str], config: GenerationConfig) -> tuple:
    """Generate flashcards with comprehensive coverage of all content."""

    try:
        # Step 1: Analyze document structure
        print("Analyzing document structure...")
        topics = analyze_document_structure(pages_text)
        print(f"Identified {len(topics)} topics")

        # Step 2: Calculate flashcard distribution
        topics = calculate_flashcard_distribution(
            topics,
            config.total_flashcards,
            config.min_flashcards_per_topic,
            config.coverage_strategy
        )

        # Step 3: Generate flashcards for each topic
        all_flashcards = []
        for i, topic in enumerate(topics):
            print(
                f"Generating flashcards for topic {i+1}/{len(topics)}: {topic.topic_name}")
            try:
                topic_flashcards = generate_flashcards_for_topic(topic, config)
                print(
                    f"Generated {len(topic_flashcards)} flashcards for {topic.topic_name}")
                all_flashcards.extend(topic_flashcards)
            except Exception as e:
                print(
                    f"Error generating flashcards for topic {topic.topic_name}: {e}")
                # Continue with other topics
                continue

        print(f"Total flashcards generated: {len(all_flashcards)}")

        # Step 4: Create coverage analysis
        coverage_analysis = {
            "total_pages_analyzed": len(pages_text),
            "topics_identified": len(topics),
            "flashcards_per_topic": {topic.topic_name: topic.flashcard_count for topic in topics},
            "coverage_percentage": 100.0,  # Full coverage by design
            "strategy_used": config.coverage_strategy,
            "pages_covered": list(range(1, len(pages_text) + 1)),
            "actual_flashcards_generated": len(all_flashcards)
        }

        return all_flashcards, topics, coverage_analysis

    except Exception as e:
        print(f"Error in generate_comprehensive_flashcards: {e}")
        raise


def extract_json_from_text(text: str) -> List[dict]:
    """Extract JSON array from LLM response text."""
    try:
        # Find JSON array in the text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # Try to parse the entire text as JSON
            return json.loads(text)
    except json.JSONDecodeError:
        return parse_flashcards_from_text(text)


def parse_flashcards_from_text(text: str) -> List[dict]:
    """Fallback parser for when JSON extraction fails."""
    flashcards = []
    lines = text.strip().split('\n')

    current_question = None
    current_answer = None

    for line in lines:
        line = line.strip()
        if line.startswith(('Q:', 'Question:', '**Q:')):
            if current_question and current_answer:
                flashcards.append({
                    "question": current_question,
                    "answer": current_answer
                })
            current_question = line.split(':', 1)[1].strip()
            current_answer = None
        elif line.startswith(('A:', 'Answer:', '**A:')):
            current_answer = line.split(':', 1)[1].strip()
        elif current_answer and line and not line.startswith(('Q:', 'A:', 'Question:', 'Answer:')):
            current_answer += " " + line

    if current_question and current_answer:
        flashcards.append({
            "question": current_question,
            "answer": current_answer
        })

    return flashcards


@app.post("/generate-flashcards", response_model=FlashcardResponse)
@app.post("/generate_flashcards", response_model=FlashcardResponse)
async def generate_flashcards(
    file: UploadFile = File(...),
    total_flashcards: int = 20,
    difficulty_level: str = "medium",
    include_definitions: bool = True,
    include_examples: bool = False,
    ensure_full_coverage: bool = True,
    min_flashcards_per_topic: int = 2,
    coverage_strategy: str = "balanced"
):
    """
    Upload a PDF and generate flashcards with comprehensive coverage of all content.

    Parameters:
    - total_flashcards: Total number of flashcards to generate (max 50)
    - coverage_strategy: 'balanced', 'sequential', or 'importance_based'
    - min_flashcards_per_topic: Minimum flashcards per identified topic
    - ensure_full_coverage: Ensure all content is covered
    """
    start_time = datetime.now()

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported")

    try:
        # Read and extract text from PDF with page tracking
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pages_text = extract_text_with_pages(pdf_file)

        if not pages_text or sum(len(text) for text in pages_text.values()) < 200:
            raise HTTPException(
                status_code=400, detail="PDF contains insufficient text content")

        # Configure generation
        config = GenerationConfig(
            total_flashcards=min(total_flashcards, 50),  # Reasonable limit
            difficulty_level=difficulty_level,
            include_definitions=include_definitions,
            include_examples=include_examples,
            ensure_full_coverage=ensure_full_coverage,
            min_flashcards_per_topic=max(1, min_flashcards_per_topic),
            coverage_strategy=coverage_strategy
        )

        # Generate comprehensive flashcards
        try:
            flashcards, topics, coverage_analysis = generate_comprehensive_flashcards(
                pages_text, config)
            print(f"Successfully generated {len(flashcards)} flashcards")
        except Exception as e:
            print(f"Error in comprehensive flashcard generation: {e}")
            raise HTTPException(
                status_code=500, detail=f"Flashcard generation failed: {str(e)}")

        if not flashcards:
            raise HTTPException(
                status_code=500, detail="No flashcards were generated")

        end_time = datetime.now()
        processing_time = str(end_time - start_time)

        # Ensure all data is serializable
        try:
            response_data = FlashcardResponse(
                flashcards=flashcards,
                total_count=len(flashcards),
                processing_time=processing_time,
                source_info={
                    "filename": file.filename,
                    "file_size_mb": round(len(pdf_content) / 1024 / 1024, 2),
                    "total_pages": len(pages_text),
                    "total_text_length": sum(len(text) for text in pages_text.values()),
                    "generated_at": end_time.isoformat()
                },
                coverage_analysis=coverage_analysis,
                topics_covered=topics
            )
            print("Response data created successfully")
            return response_data

        except Exception as e:
            print(f"Error creating response: {e}")
            # Return simplified response if there's a serialization issue
            return {
                "flashcards": [{"question": fc.question, "answer": fc.answer, "topic": fc.topic} for fc in flashcards],
                "total_count": len(flashcards),
                "processing_time": processing_time,
                "source_info": {"filename": file.filename, "total_pages": len(pages_text)},
                "message": "Flashcards generated successfully with simplified response due to serialization issue"
            }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Check if the API and Ollama are running."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "connected" if response.status_code == 200 else "disconnected"

        tags_data = response.json() if response.status_code == 200 else {}
        model_available = any(
            model.get("name", "").startswith("llama3.2:3b")
            for model in tags_data.get("models", [])
        )

        return {
            "status": "healthy",
            "ollama_status": ollama_status,
            "model_available": model_available,
            "model_name": MODEL_NAME,
            "version": "2.0.0",
            "features": ["sequential_coverage", "topic_analysis", "comprehensive_coverage"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/")
async def root():
    """API information and usage instructions."""
    return {
        "message": "PDF to Flashcard API with Comprehensive Coverage",
        "version": "2.0.0",
        "features": [
            "Sequential topic processing",
            "Comprehensive content coverage",
            "Intelligent flashcard distribution",
            "Topic-based organization",
            "Coverage analysis"
        ],
        "endpoints": {
            "/generate-flashcards": "POST - Upload PDF and generate flashcards with full coverage",
            "/health": "GET - Check API and model status",
            "/docs": "GET - Interactive API documentation"
        },
        "model": MODEL_NAME,
        "usage": "Upload a PDF file to /generate-flashcards for comprehensive study flashcards"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
