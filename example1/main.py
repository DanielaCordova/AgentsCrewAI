import litellm
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

# Define the email
email = "nigerian prince sending some gold"

# Define classifier agent
classifier = Agent(
    role="email classifier",
    goal="Classify emails as 'important', 'casual', or 'spam'.",
    backstory="You help users manage their inbox efficiently.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm  # Correctly use the custom LLM instance
)

# Define responder agent
responder = Agent(
    role="email responder",
    goal="Write a concise response based on the email classification.",
    backstory="You help users respond to emails efficiently.",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm  # Correctly use the custom LLM instance
)

# Define tasks
classify_email = Task(
    description=f"Classify the email: '{email}'",
    agent=classifier,
    expected_output="One of these three options: 'important', 'casual', or 'spam'"
)

respond_to_email = Task(
    description=f"Respond to the email: '{email}' based on classification.",
    agent=responder,
    expected_output="A concise response to the email."
)

# Create CrewAI workflow
crew = Crew(
    agents=[classifier, responder],
    tasks=[classify_email, respond_to_email],
    verbose=True,
    process=Process.sequential
)

# Execute
output = crew.kickoff()
print(output)
