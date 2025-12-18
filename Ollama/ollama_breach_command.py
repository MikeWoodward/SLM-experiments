"""Command-line app for analyzing data breach articles using Ollama models."""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import ollama
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


@dataclass
class BreachAnalysis:
    """Data class to store breach analysis information."""

    url: str
    text: Optional[str] = None
    analysis: Optional[str] = None
    structured_analysis: Optional["BreachAnalysisResponse"] = None


class BreachAnalysisResponse(BaseModel):
    """Pydantic model for structured breach analysis responses."""

    discusses_data_breach: str = Field(
        description="Does the article discuss a data breach - answer only Yes or No"
    )
    breached_entity: str = Field(
        description="Which entity was breached?"
    )
    records_breached: str = Field(
        description="How many records were breached?"
    )
    breach_occurrence_date: str = Field(
        description=(
            "What date did the breach occur - answer using dd-MMM-YYYY format, "
            "if the date is not mentioned, answer Unknown, if the date is "
            "approximate, answer with a range of dates"
        )
    )
    breach_discovery_date: str = Field(
        description="When was the breach discovered, be as accurate as you can"
    )
    breach_cause_known: str = Field(
        description="Is the cause of the breach known - answer Yes or No only"
    )
    breach_cause: str = Field(
        description="If the cause of the breach is known state it"
    )
    third_parties_involved: str = Field(
        description="Were there any third parties involved - answer only Yes or No"
    )
    third_party_names: str = Field(
        description="If there were third parties involved, list their names"
    )


def _display_progress(
    progress,
    action: str,
    item_name: str,
) -> str:
    """Helper to format progress messages."""
    if not hasattr(progress, "status"):
        return ""
    
    status_msg = f"{action} {item_name}: {progress.status}"
    if (
        hasattr(progress, "completed")
        and hasattr(progress, "total")
        and progress.total is not None
        and progress.total > 0
    ):
        percent = (progress.completed / progress.total) * 100
        status_msg = f"{action} {item_name}: {progress.status} ({percent:.1f}%)"
    return status_msg


def check_and_download_model(
    model_name: str = "mistral",
) -> tuple[bool, float]:
    """
    Check if the specified model exists in Ollama, download if not.

    Args:
        model_name: Name of the model to check/download

    Returns:
        Tuple of (True if model is available, time taken in seconds)
    """
    step_start = time.time()
    try:
        models = ollama.list()
        model_names = [model.model for model in models.models]

        if model_name in model_names or any(
            m.startswith(f"{model_name}:") for m in model_names
        ):
            elapsed = time.time() - step_start
            print(f"Model '{model_name}' is already available.")
            return True, elapsed

        print(f"Model '{model_name}' not found. Downloading...")
        for progress in ollama.pull(model=model_name, stream=True):
            if msg := _display_progress(progress, "Downloading", model_name):
                print(f"\r{msg}", end="", flush=True)

        print()
        elapsed = time.time() - step_start
        print(f"Model '{model_name}' downloaded successfully.")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - step_start
        print(f"Error checking/downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def create_custom_model(
    base_model: str,
    custom_model_name: str = "breach-analyzer",
) -> tuple[bool, float]:
    """
    Create a custom Ollama model with a system prompt.

    Args:
        base_model: Name of the base model to use
        custom_model_name: Name for the custom model

    Returns:
        Tuple of (True if model was created successfully, time taken in seconds)
    """
    step_start = time.time()
    try:
        system_prompt = get_prompt_template()
        print(f"Creating custom model '{custom_model_name}' from '{base_model}'...")
        
        for progress in ollama.create(
            model=custom_model_name,
            from_=base_model,
            system=system_prompt,
            stream=True,
        ):
            if msg := _display_progress(progress, "Creating", custom_model_name):
                print(f"\r{msg}", end="", flush=True)

        print()
        elapsed = time.time() - step_start
        print(f"Custom model '{custom_model_name}' created successfully.")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - step_start
        print(f"Error creating custom model: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def download_and_extract_text(url: str) -> tuple[Optional[str], float]:
    """
    Download HTML from URL and extract text content.

    Args:
        url: URL to download

    Returns:
        Tuple of (extracted text content or None if download fails,
                 time taken in seconds)
    """
    step_start = time.time()
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=" ", strip=True)

        # Clean up multiple whitespaces
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        elapsed = time.time() - step_start
        return text, elapsed

    except requests.exceptions.RequestException as e:
        elapsed = time.time() - step_start
        print(f"Failed to download {url}: {e}")
        return None, elapsed
    except Exception as e:
        elapsed = time.time() - step_start
        print(f"Error processing {url}: {e}")
        import traceback
        traceback.print_exc()
        return None, elapsed


# Questions list for breach analysis
QUESTIONS = [
    "does the article discuss a data breach - answer only Yes or No",
    "which entity was breached?",
    "how many records were breached?",
    (
        "what date did the breach occur - answer using dd-MMM-YYYY format, "
        "if the date is not mentioned, answer Unknown, if the date is "
        "approximate, answer with a range of dates?"
    ),
    "When was the breach discovered, be as accurate as you can?",
    "is the cause of the breach known - answer Yes or No only",
    "If the cause of the breach is known state it",
    "were there any third parties involved - answer only Yes or No",
    "if there were third parties involved, list their names",
]


def get_prompt_template() -> str:
    """Get the system prompt template for the custom model."""
    return (
        "You are a data analyst analyzing text articles for information "
        "on data breaches. Analyze the provided text and answer all the "
        "questions in the specified format."
    )


def analyze_breach(
    breach: BreachAnalysis,
    model_name: str = "breach-analyzer",
) -> float:
    """
    Analyze a breach article using Ollama.

    Args:
        breach: BreachAnalysis object with text to analyze
        model_name: Name of the Ollama model to use

    Returns:
        Time taken in seconds
    """
    if breach.text is None:
        return 0.0

    step_start = time.time()
    try:
        prompt = (
            f"Here is the text to analyze: ###START### {breach.text} ###END###"
        )

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            format=BreachAnalysisResponse.model_json_schema(),
        )

        raw_response = (
            response.response
            if hasattr(response, "response")
            else response.message.get("content", "")
            if hasattr(response, "message")
            else str(response)
        )
        breach.analysis = raw_response

        # Parse and validate JSON response
        try:
            breach.structured_analysis = BreachAnalysisResponse.model_validate_json(
                raw_response
            )
        except Exception as e:
            print(
                f"Warning: Could not parse/validate JSON for {breach.url}: {e}"
            )
            breach.structured_analysis = None

        elapsed = time.time() - step_start
        return elapsed

    except Exception as e:
        elapsed = time.time() - step_start
        print(f"Error analyzing {breach.url}: {e}")
        import traceback
        traceback.print_exc()
        breach.analysis = f"Error: {e}"
        breach.structured_analysis = None
        return elapsed


def format_analysis_with_questions(
    breach: BreachAnalysis,
) -> None:
    """
    Display analysis results with questions paired with answers.

    Args:
        breach: BreachAnalysis object with analysis data
    """
    if breach.structured_analysis:
        structured = breach.structured_analysis
        field_mapping = [
            "discusses_data_breach",
            "breached_entity",
            "records_breached",
            "breach_occurrence_date",
            "breach_discovery_date",
            "breach_cause_known",
            "breach_cause",
            "third_parties_involved",
            "third_party_names",
        ]
        for i, (question, field_name) in enumerate(zip(QUESTIONS, field_mapping), 1):
            answer = getattr(structured, field_name)
            print(f"{i}. {question}")
            print(f"   Answer: {answer}")
            print()
    elif breach.analysis:
        # Fallback to parsing raw text if structured analysis not available
        analysis_text = breach.analysis
        # Try to parse numbered responses (e.g., "1. Yes\n2. Company Name\n...")
        lines = analysis_text.strip().split("\n")

        # Check if response is in numbered format
        numbered_answers = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to match patterns like "1. answer", "1 answer", "1- answer", etc.
            for i in range(len(QUESTIONS)):
                question_num = i + 1
                # Check various patterns - be more flexible with matching
                # Pattern: number followed by period, dash, colon, or space
                pattern = re.compile(
                    rf"^{question_num}[\.\-\:\s]+(.+)$",
                    re.IGNORECASE
                )
                match = pattern.match(line)
                if match:
                    answer = match.group(1).strip()
                    if answer and question_num not in numbered_answers:
                        numbered_answers[question_num] = answer
                        break

        # Display questions with answers
        if numbered_answers and len(numbered_answers) >= len(QUESTIONS) * 0.5:
            # Good match - show Q&A pairs
            for i, question in enumerate(QUESTIONS, 1):
                answer = numbered_answers.get(i, "Not answered")
                print(f"{i}. {question}")
                print(f"   Answer: {answer}")
                print()
        else:
            # Show questions first, then full response
            print("Questions:")
            for i, question in enumerate(QUESTIONS, 1):
                print(f"{i}. {question}")
            print()
            print("LLM Response:")
            print(analysis_text)


def select_model() -> str:
    """
    Ask user to select a model from the available list.

    Returns:
        Selected model name
    """
    available_models = ["llama3.2", "mistral", "phi4"]
    
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = input("\nSelect model (enter number): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                selected = available_models[choice_num - 1]
                print(f"\nSelected model: {selected}")
                return selected
            else:
                print(
                    f"Please enter a number between 1 and {len(available_models)}"
                )
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def _display_breach_results(breach: BreachAnalysis) -> None:
    """Display breach analysis results with text preview."""
    print("\n" + "-" * 70)
    print(f"Results for: {breach.url}")
    print("-" * 70)
    format_analysis_with_questions(breach)
    print("\nOriginal Text (first 500 chars):")
    if breach.text:
        text_preview = (
            breach.text[:500] + "..."
            if len(breach.text) > 500
            else breach.text
        )
        print(text_preview)
    print()


def wait_for_key() -> None:
    """Wait for user to press a key to start."""
    print("\nPress Enter to start the analysis...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze data breach articles using Ollama models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama3.2", "mistral", "phi4"],
        help="LLM model to use (llama3.2, mistral, or phi4)",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="Skip the 'press key to start' prompt",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Breach Analysis Tool")
    print("=" * 70)
    
    print("\nQuestions to be answered:")
    for i, question in enumerate(QUESTIONS, 1):
        print(f"  {i}. {question}")
    
    # Get model selection (from args or interactive)
    if args.model:
        llm_model = args.model
        print(f"\nUsing model: {llm_model}")
    else:
        llm_model = select_model()
    
    # Wait for user to press key (unless skipped)
    if not args.skip_prompt:
        wait_for_key()
    
    start_time = time.time()

    # Initialize breach analysis objects
    urls = [
        (
            "https://mashable.com/article/conduent-data-breach-10-5-million-"
            "social-security-numbers"
        ),
        "https://www.bbc.com/news/articles/cdrn28l6m27o",
        "https://www.twingate.com/blog/tips/Roku-data-breach",
        (
            "https://topclassactions.com/lawsuit-settlements/lawsuit-news/"
            "vitas-hospice-services-data-breach-exposed-patient-employee-"
            "information-class-action-says/"
        ),
        (
            "https://fox17.com/news/local/bluecross-members-in-tennessee-"
            "urged-to-act-after-data-breach-exposes-personal-info-conduent-"
            "class-action-lawsuits-personal-information-social-security-"
            "numbers"
        ),
        "https://www.bbc.com/sport/tennis/articles/c2kp74gxpe7o",
    ]

    breach_analyses = [BreachAnalysis(url=url) for url in urls]

    # Check and download model
    print("\n" + "=" * 70)
    print("Step 1: Model Check/Download")
    print("=" * 70)
    model_available, model_time = check_and_download_model(llm_model)
    if not model_available:
        print(
            f"\nFailed to ensure {llm_model} model is available. Aborting."
        )
        sys.exit(1)

    # Create custom model with system prompt
    print("\n" + "=" * 70)
    print("Step 2: Creating Custom Model")
    print("=" * 70)
    custom_model_name = "breach-analyzer"
    model_created, create_time = create_custom_model(
        base_model=llm_model,
        custom_model_name=custom_model_name,
    )
    if not model_created:
        print(
            f"\nFailed to create custom model '{custom_model_name}'. Aborting."
        )
        sys.exit(1)

    # Download and extract text from URLs
    print("\n" + "=" * 70)
    print("Step 3: Downloading Articles")
    print("=" * 70)
    download_times = []
    for i, breach in enumerate(breach_analyses, 1):
        print(
            f"\nDownloading article {i}/{len(breach_analyses)}: {breach.url}"
        )
        breach.text, download_time = download_and_extract_text(breach.url)
        download_times.append((breach.url, download_time))
        if breach.text:
            print(f"✓ Downloaded: {breach.url} ({download_time:.2f}s)")
        else:
            print(f"✗ Failed: {breach.url} ({download_time:.2f}s)")

    # Analyze breaches using Ollama
    print("\n" + "=" * 70)
    print("Step 4: Analyzing Breaches")
    print("=" * 70)
    successful_downloads = [
        b for b in breach_analyses if b.text is not None
    ]

    if not successful_downloads:
        print("\nNo articles were successfully downloaded.")
        sys.exit(1)

    analysis_times = []
    print("\nAnalysis Results (as they complete):\n")
    for i, breach in enumerate(successful_downloads, 1):
        print(
            f"\nAnalyzing breach {i}/{len(successful_downloads)}: "
            f"{breach.url}"
        )
        analysis_time = analyze_breach(breach, custom_model_name)
        analysis_times.append((breach.url, analysis_time))
        print(f"✓ Analyzed: {breach.url} ({analysis_time:.2f}s)")

        # Display results immediately after each analysis
        if breach.analysis:
            _display_breach_results(breach)

    # Display timing summary
    print("\n" + "=" * 70)
    print("Timing Summary")
    print("=" * 70)
    total_time = time.time() - start_time

    print("\nStep Timings:")
    print(f"  - Model Check/Download: {model_time:.2f}s")
    print(f"  - Custom Model Creation: {create_time:.2f}s")
    print(
        f"  - Total Download Time: "
        f"{sum(t for _, t in download_times):.2f}s"
    )
    print(
        f"  - Total Analysis Time: "
        f"{sum(t for _, t in analysis_times):.2f}s"
    )
    print(f"  - Total Processing Time: {total_time:.2f}s")

    print("\nPer-Article Download Times:")
    for url, dt in download_times:
        print(f"  - {url[:50]}...: {dt:.2f}s")
    print("\nPer-Article Analysis Times:")
    for url, at in analysis_times:
        print(f"  - {url[:50]}...: {at:.2f}s")

    # Display results summary
    print("\n" + "=" * 70)
    print("Analysis Results Summary")
    print("=" * 70)

    for breach in breach_analyses:
        if breach.analysis:
            _display_breach_results(breach)

