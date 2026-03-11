from abc import ABC, abstractmethod
from google import genai
from google.api_core import exceptions
from groq import Groq, GroqError


#Abstract Class
class LLM(ABC):

    @abstractmethod
    def generate(self, text: str) -> str:
        pass


#Gemini implementation
class Gemini(LLM):
    def __init__(self, api_key, model_name="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, text: str) -> str:
        prompt = (f"Can you rewrite this text keeping the same style of writing,"
                  f" only provide the re-written text in your response: {text}")

        try:
            result = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return result.text.replace("\n", " ").strip()
        except exceptions.ResourceExhausted:
            print("Gemini quota reached.")
            return None
        except exceptions.InvalidArgument:
            print("Invalid")
            return None
        except Exception as e:
            print(f"Gemini unexpected error: {e}")
            return None

#Groq implementation
class GroqAPI(LLM):
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, text: str) -> str:
        prompt = (f"Can you rewrite this text keeping the same style of writing,"
                  f" only provide the re-written text in your response: {text}")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )
            return chat_completion.choices[0].message.content.replace("\n", " ").strip()

        except GroqError as e:
            if "rate_limit_exceeded" in str(e).lower():
                print("Groq quota reached.")
            else:
                print(f"Groq unexpected error: {e}")
            return None
