"""
Global configuration loader and runtime settings management.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Automatically locate and load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the Task 3 pipeline."""
    # Blockchain
    sepolia_rpc_url: str = os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org")
    sepolia_private_key: str = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()
    sepolia_contract_address: str = os.getenv("SEPOLIA_CONTRACT_ADDRESS", "").strip()
    simulated_blockchain: str = os.getenv("SIMULATED_BLOCKCHAIN", "auto").lower()

    # Search & Discovery
    search_engine: str = os.getenv("SEARCH_ENGINE", "auto").lower()
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "").strip()
    playwright_headless: bool = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"

    # Concurrency & Early Stopping
    concurrency_limit: int = int(os.getenv("CONCURRENCY_LIMIT", "8"))
    early_stopping: bool = os.getenv("EARLY_STOPPING", "true").lower() == "true"

    # Thresholds
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))
    margin_threshold: float = float(os.getenv("MARGIN_THRESHOLD", "0.10"))
    early_stop_similarity: float = 0.85

    @property
    def force_simulated_blockchain(self) -> bool:
        if self.simulated_blockchain == "true":
            return True
        if self.simulated_blockchain == "false":
            return False
        # 'auto': fallback to simulated if private key is not configured
        return not bool(self.sepolia_private_key and len(self.sepolia_private_key) >= 64)


def get_config() -> PipelineConfig:
    """Returns singleton pipeline configuration instance."""
    return PipelineConfig()
