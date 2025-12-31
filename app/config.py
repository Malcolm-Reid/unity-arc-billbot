from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from Railway (or .env locally)
load_dotenv()

class Settings(BaseModel):
    # API keys (these live in Railway, NOT in GitHub)
    congress_api_key: str | None = os.getenv("CONGRESS_API_KEY")
    openstates_api_key: str | None = os.getenv("OPENSTATES_API_KEY")

    # Where bill data files live
    data_path_fed: str = os.getenv(
        "DATA_PATH_FED",
        "app/data/bills_federal.jsonl"
    )
    data_path_state: str = os.getenv(
        "DATA_PATH_STATE",
        "app/data/bills_state.jsonl"
    )

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

# Create one shared settings object
settings = Settings()
