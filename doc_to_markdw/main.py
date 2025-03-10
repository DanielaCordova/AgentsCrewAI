import os
import tempfile
import uuid
import docling
import litellm
from crewai import Agent, Task, Crew, Process
from llama_index.readers.docling import DoclingReader

class OllamaLLM:
    """Custom LLM Wrapper for CrewAI using LiteLLM"""
    def __init__(self, model_name="ollama/llama3", api_base="http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base

    def __call__(self, prompt: str) -> str:
        """ Calls Ollama through LiteLLM and returns the response """
        response = litellm.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base
        )
        return response["choices"][0]["message"]["content"]

# Create Ollama model instance
ollama_llm = OllamaLLM()

def extract_text_from_pdf(pdf_path: str) -> str:
    """ Extracts text from a PDF using DoclingReader """
    reader = DoclingReader()
    documents = reader.load_data(pdf_path)
    return "\n".join([doc.text for doc in documents])

# Define PDF processing agents
pdf_extractor = Agent(
    role="PDF Extractor",
    goal="Extract text and structure from PDF files using Docling.",
    backstory="You specialize in parsing PDFs and retrieving structured content using Docling.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

markdown_converter = Agent(
    role="Markdown Converter",
    goal="Convert extracted text into a well-structured Markdown document with properly formatted tables and headers, ensuring correct spacing and proper table formatting without enclosing everything in code blocks.",
    backstory="You are skilled at formatting documents into clean Markdown with enhanced readability, ensuring proper alignment of table elements.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# Define tasks
def extract_pdf_task(pdf_path):
    return Task(
        description=f"Extract structured text from the PDF file: {pdf_path} using Docling.",
        agent=pdf_extractor,
        expected_output="Well-structured raw text extracted from the PDF, preserving tables and formatting properly.",
        config={"pdf_path": pdf_path}  # Use `config` instead of `context`
    )

def convert_to_markdown_task(output_path="output.md"):
    return Task(
        description="Convert the extracted structured text into a properly formatted Markdown document with structured tables and properly aligned headers. Ensure correct cell separation and alignment in tables, without wrapping everything in code blocks.",
        agent=markdown_converter,
        expected_output=f"Structured Markdown file saved at {output_path}, ensuring proper table formatting.",
        config={"output_file": output_path}
    )

# File path simulation (Replace with actual file input)
pdf_file_path = "invoice.pdf"

# Create CrewAI workflow
crew = Crew(
    agents=[pdf_extractor, markdown_converter],
    tasks=[extract_pdf_task(pdf_file_path), convert_to_markdown_task()],
    verbose=True,
    process=Process.sequential
)

# Execute workflow
output = crew.kickoff()
print(output)
