# llm_utils.py
# -------------
# This module provides a model-agnostic interface for calling different LLMs (OpenAI, Gemini, Llama/RAG).
# The main entry point is analyze_policy(prompt, model), which dispatches to the correct backend.
# To add a new model, implement a new analyze_with_<model>() function and update analyze_policy.

import os
import google.generativeai as genai
import openai  # For OpenAI GPT models
import logging

logger = logging.getLogger(__name__)


def analyze_with_gemini(prompt):
    """
    Calls Gemini API with the given prompt and returns the response text.
    Requires GEMINI_API_KEY in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            )
        )
        return response.text if response and response.text else None
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return None


def analyze_with_openai(prompt):
    """
    Calls OpenAI API (e.g., GPT-3.5/4) with the given prompt and returns the response text.
    Requires OPENAI_API_KEY in the environment.
    Compatible with openai>=1.0.0.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return None
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None


def analyze_with_llama_rag(prompt):
    """
    Placeholder for future Llama/RAG implementation.
    Extend this function to support your own Llama or RAG pipeline.
    """
    # TODO: Implement Llama/RAG logic here
    logger.warning("Llama/RAG model not implemented yet.")
    return None


def analyze_policy(prompt, model="openai"):
    """
    Model-agnostic interface for policy analysis.
    Supported models:
      - "openai": Uses OpenAI GPT models (default)
      - "gemini": Uses Google Gemini API
      - "llama":  Placeholder for Llama/RAG
    To add a new model, add a new elif branch and function.
    """
    if model == "gemini":
        return analyze_with_gemini(prompt)
    elif model == "llama":
        return analyze_with_llama_rag(prompt)
    elif model == "openai":
        return analyze_with_openai(prompt)
    else:
        raise ValueError(f"Unknown model: {model}") 
