import os
from dotenv import load_dotenv
from LLMs import Gemini
from LLMs import GroqAPI
from repo import RepoProcessor

# Load tokens
load_dotenv("token.env")

#Configuration
OWNER = "anarchie347"
REPO = "detecting-patterns-in-rhetoric"
PATH = "human_bullshit/mission_files"
OUTPUT_DIR_GEMINI = "gemini_results"
OUTPUT_DIR_GROQ = "groq_results"
LIMIT = 5

#getting tokens from token.env
github_token = os.getenv("GITHUB_TOKEN")
gemini_token = os.getenv("GEMINI_TOKEN")
groq_token = os.getenv("GROQ_TOKEN")

gemini_llm = Gemini(api_key=gemini_token)
groq_llm = GroqAPI(api_key=groq_token)


processor_gemini = RepoProcessor(
    owner=OWNER,
    repo=REPO,
    github_token=github_token,
    llm_backend=gemini_llm,
    output_dir=OUTPUT_DIR_GEMINI,
    limit=LIMIT
)

processor_groq = RepoProcessor(
    owner=OWNER,
    repo=REPO,
    github_token=github_token,
    llm_backend=groq_llm,
    output_dir=OUTPUT_DIR_GROQ,
    limit=LIMIT
)

processor_gemini.process_folder(PATH)
processor_groq.process_folder(PATH)