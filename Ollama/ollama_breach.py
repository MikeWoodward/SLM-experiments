"""Streamlit app for analyzing data breach articles using Ollama's Mistral model."""

import re
import time
from dataclasses import dataclass
from typing import Optional

import ollama
import requests
import streamlit as st
from bs4 import BeautifulSoup


@dataclass
class BreachAnalysis:
    """Data class to store breach analysis information."""

    url: str
    text: Optional[str] = None
    analysis: Optional[str] = None


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

        # Check for exact match or mistral:latest
        if model_name in model_names or any(
            m.startswith(f"{model_name}:") for m in model_names
        ):
            elapsed = time.time() - step_start
            st.info(f"Model '{model_name}' is already available.")
            return True, elapsed

        # Model not found, download it
        st.info(f"Model '{model_name}' not found. Downloading...")
        with st.status(f"Downloading {model_name}...", expanded=True) as status:
            for progress in ollama.pull(model=model_name, stream=True):
                if hasattr(progress, "status"):
                    status.update(
                        label=f"Downloading {model_name}: {progress.status}"
                    )
                    if hasattr(progress, "completed") and hasattr(progress, "total"):
                        if progress.total > 0:
                            percent = (progress.completed / progress.total) * 100
                            status.update(
                                label=(
                                    f"Downloading {model_name}: "
                                    f"{progress.status} "
                                    f"({percent:.1f}%)"
                                )
                            )

        elapsed = time.time() - step_start
        st.success(f"Model '{model_name}' downloaded successfully.")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - step_start
        st.error(f"Error checking/downloading model: {e}")
        st.exception(e)
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
        st.warning(f"Failed to download {url}: {e}")
        return None, elapsed
    except Exception as e:
        elapsed = time.time() - step_start
        st.warning(f"Error processing {url}: {e}")
        st.exception(e)
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
    """
    Get the prompt template without the article text placeholder.

    Returns:
        Prompt template string without the last sentence
    """
    questions_text = " ".join(
        f"{i + 1} {q}, " for i, q in enumerate(QUESTIONS)
    ).rstrip(", ")
    return (
        "You are a data analyst analyzing text articles for information "
        "on data breaches. You are to answer the following questions: "
        f"{questions_text}. "
        "Give your answer in the form of a list."
    )


def format_prompt(article_text: str) -> str:
    """
    Format the prompt for breach analysis.

    Args:
        article_text: The article text to analyze

    Returns:
        Formatted prompt string
    """
    return (
        f"{get_prompt_template()} "
        f"Here is the text to analyze: ###START### {article_text} ###END###"
    )


def analyze_breach(
    breach: BreachAnalysis,
    model_name: str = "mistral",
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
        prompt = format_prompt(breach.text)

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
        )

        if response and hasattr(response, "message"):
            breach.analysis = response.message.get("content", "")
        else:
            breach.analysis = str(response)

        elapsed = time.time() - step_start
        return elapsed

    except Exception as e:
        elapsed = time.time() - step_start
        st.error(f"Error analyzing {breach.url}: {e}")
        st.exception(e)
        breach.analysis = f"Error: {e}"
        return elapsed


def format_analysis_with_questions(analysis_text: str) -> None:
    """
    Display analysis results with questions paired with answers.

    Args:
        analysis_text: The raw analysis text from the LLM
    """
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
            st.write(f"**{i}. {question}**")
            st.write(f"   *Answer:* {answer}")
            st.write("")
    else:
        # Show questions first, then full response
        st.write("**Questions:**")
        for i, question in enumerate(QUESTIONS, 1):
            st.write(f"{i}. {question}")
        st.write("")
        st.write("**LLM Response:**")
        st.write(analysis_text)


def about_page() -> None:
    """Display the About page."""
    st.title("About")
    st.write(
        "This app shows a straightforward use case for Ollama's "
        "NLP-oriented models. The case is the analysis of press reports "
        "for data breaches."
    )


def analysis_page() -> None:
    """Display the Analysis page with breach analysis functionality."""
    st.title("Breach Analysis")

    st.subheader("Questions to be answered:")
    for i, question in enumerate(QUESTIONS, 1):
        st.write(f"{i}. {question}")

    st.subheader("Select LLM Model")
    llm_model = st.selectbox(
        "Select LLM to use",
        options=["llama3.2", "mistral", "phi4"],
        index=None,
        placeholder="Choose a model...",
    )

    if llm_model:
        if st.button("Go", type="primary"):
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
            st.subheader("Step 1: Model Check/Download")
            model_available, model_time = check_and_download_model(llm_model)
            if not model_available:
                st.error(
                    f"Failed to ensure {llm_model} model is available. Aborting."
                )
                return

            # Download and extract text from URLs
            st.subheader("Step 2: Downloading Articles")
            download_times = []
            with st.status("Downloading articles...", expanded=True) as status:
                for i, breach in enumerate(breach_analyses, 1):
                    status.update(
                        label=(
                            f"Downloading article {i}/{len(breach_analyses)}: "
                            f"{breach.url}"
                        )
                    )
                    breach.text, download_time = download_and_extract_text(
                        breach.url
                    )
                    download_times.append((breach.url, download_time))
                    if breach.text:
                        st.success(
                            f"✓ Downloaded: {breach.url} ({download_time:.2f}s)"
                        )
                    else:
                        st.warning(
                            f"✗ Failed: {breach.url} ({download_time:.2f}s)"
                        )

            # Analyze breaches using Ollama
            st.subheader("Step 3: Analyzing Breaches")
            successful_downloads = [
                b for b in breach_analyses if b.text is not None
            ]

            if not successful_downloads:
                st.error("No articles were successfully downloaded.")
                return

            analysis_times = []
            st.write("**Analysis Results (as they complete):**")
            with st.status("Analyzing breaches...", expanded=True) as status:
                for i, breach in enumerate(successful_downloads, 1):
                    status.update(
                        label=(
                            f"Analyzing breach {i}/{len(successful_downloads)}: "
                            f"{breach.url}"
                        )
                    )
                    analysis_time = analyze_breach(breach, llm_model)
                    analysis_times.append((breach.url, analysis_time))
                    st.info(
                        f"✓ Analyzed: {breach.url} ({analysis_time:.2f}s)"
                    )

                    # Display results immediately after each analysis
                    if breach.analysis:
                        with st.expander(
                            f"Results for: {breach.url}",
                            expanded=True
                        ):
                            format_analysis_with_questions(breach.analysis)
                            st.write("---")
                            st.write("**Original Text (first 500 chars):**")
                            if breach.text:
                                text_preview = (
                                    breach.text[:500] + "..."
                                    if len(breach.text) > 500
                                    else breach.text
                                )
                                st.text(text_preview)

            # Display timing summary
            st.subheader("Timing Summary")
            total_time = time.time() - start_time

            timing_col1, timing_col2 = st.columns(2)

            with timing_col1:
                st.write("**Step Timings:**")
                st.write(f"- Model Check/Download: {model_time:.2f}s")
                st.write(
                    f"- Total Download Time: "
                    f"{sum(t for _, t in download_times):.2f}s"
                )
                st.write(
                    f"- Total Analysis Time: "
                    f"{sum(t for _, t in analysis_times):.2f}s"
                )
                st.write(f"- **Total Processing Time: {total_time:.2f}s**")

            with timing_col2:
                st.write("**Per-Article Download Times:**")
                for url, dt in download_times:
                    st.write(f"- {url[:50]}...: {dt:.2f}s")
                st.write("**Per-Article Analysis Times:**")
                for url, at in analysis_times:
                    st.write(f"- {url[:50]}...: {at:.2f}s")

            # Display results
            st.subheader("Analysis Results")

            for breach in breach_analyses:
                if breach.analysis:
                    with st.expander(
                        f"Results for: {breach.url}", expanded=False
                    ):
                        format_analysis_with_questions(breach.analysis)
                        st.write("---")
                        st.write("**Original Text (first 500 chars):**")
                        if breach.text:
                            text_preview = (
                                breach.text[:500] + "..."
                                if len(breach.text) > 500
                                else breach.text
                            )
                            st.text(text_preview)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Ollama Breach Analysis",
        page_icon="🔒",
        layout="wide",
    )

    # Create pages
    about = st.Page(about_page, title="About", icon="ℹ️")
    analysis = st.Page(analysis_page, title="Analysis", icon="🔍")

    # Create navigation
    pg = st.navigation([about, analysis])
    pg.run()
