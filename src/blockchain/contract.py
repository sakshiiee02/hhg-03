"""
Web3 smart contract client for FaceVerificationRegistry on Ethereum Sepolia.
Provides seamless fallback to an in-memory simulated EVM engine to guarantee 100% demo reliability.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from eth_account import Account
from web3 import Web3
from web3.exceptions import Web3Exception

from src.blockchain.hash import to_bytes32, to_hex32

logger = logging.getLogger(__name__)


@dataclass
class AttestationReceipt:
    """Structured receipt for an on-chain attestation transaction."""
    tx_hash: str
    block_number: int
    submitter: str
    timestamp: int
    content_hash: str
    record_hash: str
    network: str
    is_simulated: bool
    explorer_url: Optional[str] = None


class BaseBlockchainClient:
    """Abstract interface for blockchain interaction."""

    @property
    def network(self) -> str:
        raise NotImplementedError

    def attest(self, content_hash: Union[str, bytes], record_hash: Union[str, bytes]) -> AttestationReceipt:
        raise NotImplementedError

    def get_record(self, record_hash: Union[str, bytes]) -> Dict[str, Any]:
        raise NotImplementedError


class MockBlockchainClient(BaseBlockchainClient):
    """
    In-memory simulated EVM registry client.
    Guarantees zero-failure live demo recordings when testnet RPCs or faucets are unavailable.
    """

    def __init__(self, mock_address: str = "0x71C8A18B632e2F60A3b7f168051EBE1C0526B500"):
        self.contract_address = mock_address
        self.records: Dict[bytes, Dict[str, Any]] = {}
        self.current_block = 6543210
        self.default_sender = "0xDemoSubmitterAddress9876543210abcdef123456"

    @property
    def network(self) -> str:
        return "Ethereum Sepolia (Simulated EVM Mode)"

    def attest(self, content_hash: Union[str, bytes], record_hash: Union[str, bytes]) -> AttestationReceipt:
        c_bytes = to_bytes32(content_hash)
        r_bytes = to_bytes32(record_hash)
        now_ts = int(time.time())
        self.current_block += 1

        self.records[r_bytes] = {
            "contentHash": c_bytes,
            "recordHash": r_bytes,
            "timestamp": now_ts,
            "submitter": self.default_sender,
        }

        # Deterministic simulated tx hash
        tx_digest = Web3.keccak(c_bytes + r_bytes + now_ts.to_bytes(8, "big")).hex()
        tx_hash = "0x" + tx_digest if not tx_digest.startswith("0x") else tx_digest

        return AttestationReceipt(
            tx_hash=tx_hash,
            block_number=self.current_block,
            submitter=self.default_sender,
            timestamp=now_ts,
            content_hash=to_hex32(c_bytes),
            record_hash=to_hex32(r_bytes),
            network="Ethereum Sepolia (Simulated EVM Mode)",
            is_simulated=True,
            explorer_url=f"https://sepolia.etherscan.io/tx/{tx_hash}",
        )

    def get_record(self, record_hash: Union[str, bytes]) -> Dict[str, Any]:
        r_bytes = to_bytes32(record_hash)
        if r_bytes not in self.records:
            raise KeyError(f"Record not found on-chain: {to_hex32(r_bytes)}")
        rec = self.records[r_bytes]
        return {
            "contentHash": to_hex32(rec["contentHash"]),
            "recordHash": to_hex32(rec["recordHash"]),
            "timestamp": rec["timestamp"],
            "submitter": rec["submitter"],
        }


class SepoliaBlockchainClient(BaseBlockchainClient):
    """Production Web3 client targeting Ethereum Sepolia testnet."""

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        contract_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 15}))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum Sepolia RPC at {rpc_url}")

        clean_pk = private_key.strip()
        if not clean_pk.startswith("0x"):
            clean_pk = "0x" + clean_pk
        self.account = Account.from_key(clean_pk)
        self.submitter_address = self.account.address

        self.contract_address = contract_address
        self.contract = None
        self.abi = None

        self._load_artifact()

    @property
    def network(self) -> str:
        return "Ethereum Sepolia"

    def _load_artifact(self) -> None:
        build_path = Path(__file__).resolve().parent.parent.parent / "contracts" / "build" / "FaceVerificationRegistry.json"
        if not build_path.exists():
            raise FileNotFoundError(f"Contract artifact not found at {build_path}. Run build step.")

        with open(build_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.abi = data["abi"]
            self.bytecode = data["bytecode"]

        if self.contract_address and Web3.is_address(self.contract_address):
            checksum_addr = Web3.to_checksum_address(self.contract_address)
            self.contract = self.w3.eth.contract(address=checksum_addr, abi=self.abi)

    def attest(self, content_hash: Union[str, bytes], record_hash: Union[str, bytes]) -> AttestationReceipt:
        if not self.contract:
            raise ValueError("Contract is not deployed or contract_address is not configured.")

        c_bytes = to_bytes32(content_hash)
        r_bytes = to_bytes32(record_hash)

        nonce = self.w3.eth.get_transaction_count(self.submitter_address, "pending")
        gas_price = self.w3.eth.gas_price

        tx = self.contract.functions.attest(c_bytes, r_bytes).build_transaction({
            "from": self.submitter_address,
            "nonce": nonce,
            "gasPrice": int(gas_price * 1.15),  # 15% buffer
        })

        # Estimate gas
        try:
            estimated_gas = self.w3.eth.estimate_gas(tx)
            tx["gas"] = int(estimated_gas * 1.2)
        except Exception:
            tx["gas"] = 150000

        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=self.account.key)
        tx_hash_bytes = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash = self.w3.to_hex(tx_hash_bytes)

        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=60)
        block = self.w3.eth.get_block(receipt.blockNumber)

        return AttestationReceipt(
            tx_hash=tx_hash,
            block_number=receipt.blockNumber,
            submitter=self.submitter_address,
            timestamp=block.timestamp,
            content_hash=to_hex32(c_bytes),
            record_hash=to_hex32(r_bytes),
            network="Ethereum Sepolia",
            is_simulated=False,
            explorer_url=f"https://sepolia.etherscan.io/tx/{tx_hash}",
        )

    def get_record(self, record_hash: Union[str, bytes]) -> Dict[str, Any]:
        if not self.contract:
            raise ValueError("Contract is not initialized.")
        r_bytes = to_bytes32(record_hash)
        result = self.contract.functions.getRecord(r_bytes).call()
        # Returns struct (contentHash, recordHash, timestamp, submitter)
        return {
            "contentHash": to_hex32(result[0]),
            "recordHash": to_hex32(result[1]),
            "timestamp": result[2],
            "submitter": result[3],
        }


def get_blockchain_client(
    rpc_url: Optional[str] = None,
    private_key: Optional[str] = None,
    contract_address: Optional[str] = None,
    force_simulated: bool = False,
) -> BaseBlockchainClient:
    """
    Factory that instantiates SepoliaBlockchainClient if valid credentials are provided,
    otherwise returns MockBlockchainClient with informative logging.
    """
    if force_simulated:
        logger.info("Operating in forced Simulated EVM mode.")
        return MockBlockchainClient()

    if not rpc_url or not private_key or len(private_key.strip()) < 64:
        logger.info("Sepolia credentials missing or incomplete. Activating Simulated EVM mode.")
        return MockBlockchainClient()

    try:
        client = SepoliaBlockchainClient(rpc_url=rpc_url, private_key=private_key, contract_address=contract_address)
        logger.info("Connected to live Ethereum Sepolia testnet.")
        return client
    except Exception as e:
        logger.warning(f"Could not connect to live Sepolia ({e}). Falling back to Simulated EVM mode.")
        return MockBlockchainClient()
