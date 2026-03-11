import os
import requests
from LLMs import LLM

#Gets data from github and saves locally
class RepoProcessor:
    def __init__(self, owner, repo, github_token, llm_backend: LLM, output_dir, limit=5):
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.llm_backend = llm_backend
        self.output_dir = output_dir
        self.limit = limit
        self.files_processed = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_folder(self, path):
        if self.files_processed >= self.limit:
            return

        try:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/contents/{path}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            items = response.json()
        except requests.exceptions.HTTPError as err:
            print(f"Error accessing github folder ({path}): {err}")
            return
        except requests.exceptions.Timeout:
            print("Request timed out")
            return
        except Exception as e:
            print(f"Error accessing {path}: {e}")
            return

        for item in items:
            if self.files_processed >= self.limit:
                break

            if item['type'] == 'dir':
                self.process_folder(item['path'])

            elif item['type'] == 'file' and item['name'].endswith('.csv') and item['name']!="info.csv":
                self.process_file(item)

    def process_file(self, item):
        print(f"Processing {item['name']}")
        try:
            file_response = requests.get(item['download_url'], headers=self.headers, timeout=10)
            file_response.raise_for_status()
            csv_text = file_response.text
            print(csv_text)
            print("---------------------------------------------------------------")
            generated_text = self.llm_backend.generate(csv_text)
            print(generated_text)

            if generated_text:
                self.save_result(item['name'], generated_text)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {item['name']}: {e}")
        except Exception as e:
            print(f"Unexpected error with {item['name']} - {e}")


    def save_result(self, original_name, text):
        try:
            local_filename = os.path.join(self.output_dir, f"bullshit_{original_name}")
            with open(local_filename, "w") as f:
                f.write(text)
            self.files_processed += 1
            print(f"File saved: {local_filename} ({self.files_processed}/{self.limit})")
        except IOError as e:
            print(f"Disk error: {e}")