import os
import tempfile
import uuid
import docling
import litellm
from crewai import Agent, Task, Crew, Process
import googlesearch

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

def search_google_cloud_docs(query: str) -> str:
    """ Searches Google Cloud documentation for relevant information. """
    search_results = googlesearch.search(f"site:cloud.google.com {query}", num_results=5)
    return "\n".join(search_results)

# Define Google Cloud Documentation Researcher agent
cloud_docs_researcher = Agent(
    role="Google Cloud Documentation Researcher",
    goal="Find relevant answers to queries using only Google Cloud documentation.",
    backstory="You specialize in searching Google Cloud documentation and retrieving accurate information.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# Define task
def search_google_cloud_task(query: str):
    return Task(
        description=f"Search Google Cloud documentation for information related to: {query}",
        agent=cloud_docs_researcher,
        expected_output="A list of relevant Google Cloud documentation links with summarized content.",
        config={"query": query}
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
