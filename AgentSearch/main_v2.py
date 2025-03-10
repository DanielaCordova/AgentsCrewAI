import os
import tempfile
import uuid
import docling
import litellm
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process

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

def search_google_cloud_docs(query: str) -> list:
    """ Searches Google Cloud documentation for relevant information. """
    search_url = f"https://www.google.com/search?q=site:cloud.google.com+{query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.find_all('a'):
        link = g.get("href")
        if link and "cloud.google.com" in link:
            results.append(link)
    
    return results[:5]

def fetch_page_content(url: str) -> str:
    """ Fetches and extracts meaningful content from a webpage. """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return "Failed to retrieve content."
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        extracted_text = "\n".join([p.get_text() for p in paragraphs])
        return extracted_text[:2000]  # Limit the amount of text retrieved
    except Exception as e:
        return f"Error fetching content: {e}"

# Define Google Cloud Documentation Researcher agent
cloud_docs_researcher = Agent(
    role="Google Cloud Documentation Researcher",
    goal="Find relevant answers to queries using only Google Cloud documentation and extract relevant content.",
    backstory="You specialize in searching Google Cloud documentation and retrieving accurate information, providing both links and textual evidence.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# Define task
def search_google_cloud_task(query: str):
    search_results = search_google_cloud_docs(query)
    extracted_contents = {url: fetch_page_content(url) for url in search_results}
    
    return Task(
        description=f"Search Google Cloud documentation for information related to: {query}",
        agent=cloud_docs_researcher,
        expected_output="A list of relevant Google Cloud documentation links with summarized content.",
        config={"query": query, "results": extracted_contents}
    )

# Example search query
search_query = "How to use Cloud Functions with Pub/Sub?"

# Create CrewAI workflow
crew = Crew(
    agents=[cloud_docs_researcher],
    tasks=[search_google_cloud_task(search_query)],
    verbose=True,
    process=Process.sequential
)

# Execute workflow
output = crew.kickoff()
print(output)
