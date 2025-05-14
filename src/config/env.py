import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def setup_api_keys() -> None:
    try:
        pinecone_key = get_env_var("PINECONE_API_KEY")
        deepseek_key = get_env_var("DEEPSEEK_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        llm_provider = get_env_var("LLM_PROVIDER")
        
        os.environ["PINECONE_API_KEY"] = pinecone_key
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        os.environ["OPENAI_API_KEY"] = openai_key  # type: ignore
        os.environ["LLM_PROVIDER"] = llm_provider
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        exit(1)
