"""
One-click deployment script for FaceVerificationRegistry.sol to Ethereum Sepolia.
Updates .env with the deployed contract address upon successful confirmation.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

load_dotenv()


def deploy() -> None:
    rpc_url = os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org")
    private_key = os.getenv("SEPOLIA_PRIVATE_KEY", "").strip()

    if not private_key:
        print("[ERROR] SEPOLIA_PRIVATE_KEY is not configured in .env.")
        print("Please fund an Ethereum account with Sepolia testnet ETH and set SEPOLIA_PRIVATE_KEY.")
        sys.exit(1)

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"[ERROR] Cannot connect to Ethereum RPC endpoint: {rpc_url}")
        sys.exit(1)

    account = Account.from_key(private_key)
    sender = account.address
    balance_wei = w3.eth.get_balance(sender)
    balance_eth = w3.from_wei(balance_wei, "ether")

    print(f"Deployer address : {sender}")
    print(f"Network balance  : {balance_eth:.4f} Sepolia ETH")

    if balance_wei == 0:
        print("[ERROR] Account balance is 0. Please claim Sepolia ETH from a faucet.")
        sys.exit(1)

    build_path = Path(__file__).resolve().parent.parent.parent / "contracts" / "build" / "FaceVerificationRegistry.json"
    if not build_path.exists():
        print(f"[ERROR] Compiled contract not found at {build_path}")
        sys.exit(1)

    with open(build_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    abi = data["abi"]
    bytecode = data["bytecode"]

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(sender, "pending")
    gas_price = w3.eth.gas_price

    print("Submitting deployment transaction to Sepolia...")
    tx = contract.constructor().build_transaction({
        "from": sender,
        "nonce": nonce,
        "gasPrice": int(gas_price * 1.2),
    })

    try:
        tx["gas"] = int(w3.eth.estimate_gas(tx) * 1.2)
    except Exception:
        tx["gas"] = 800000

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Transaction hash : {w3.to_hex(tx_hash)}")
    print("Waiting for block confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    contract_address = receipt.contractAddress

    print("=" * 60)
    print("[SUCCESS] Contract successfully deployed!")
    print(f"Contract Address : {contract_address}")
    print(f"Etherscan URL    : https://sepolia.etherscan.io/address/{contract_address}")
    print("=" * 60)

    # Automatically write to .env if possible
    env_path = Path(".env")
    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        if "SEPOLIA_CONTRACT_ADDRESS=" in content:
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                if line.startswith("SEPOLIA_CONTRACT_ADDRESS="):
                    new_lines.append(f"SEPOLIA_CONTRACT_ADDRESS={contract_address}")
                else:
                    new_lines.append(line)
            env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            print(f"Updated .env with SEPOLIA_CONTRACT_ADDRESS={contract_address}")


if __name__ == "__main__":
    deploy()
