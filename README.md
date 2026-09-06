# 🔐 Face Identification & Blockchain Verification

An end-to-end pipeline that discovers web/social-media content, verifies a face using **OpenCV SFace**, and records the verified content's **SHA-256 hash** on a blockchain for tamper-evident verification.

### Pipeline

```text
Face Image
   ↓
Web / Social Media Discovery
   ↓
SFace Face Verification
   ↓
Matching Content
   ↓
SHA-256 Hash
   ↓
Hardhat Blockchain
   ↓
On-Chain Verification
```

## 🛠️ Tech Stack

* **Python**
* **OpenCV SFace** — face detection & verification
* **Streamlit** — application interface
* **SHA-256** — data fingerprinting
* **Solidity** — smart contract
* **Hardhat** — local blockchain
* **Web3.py** — blockchain interaction

## 📁 Project Structure

```text
face_blockchain_system/
├── app.py
├── pipeline.py
├── core/
├── contracts/
├── tests/
├── sample_data/
├── requirements.txt
└── README.md
```

## 🚀 Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the local blockchain

From the project directory:

```bash
npx hardhat node
```

Keep this terminal running.

### 3. Start the application

Open a **new terminal**, activate your virtual environment if required, and run:

```bash
streamlit run app.py
```

## 🔄 How It Works

1. Upload a face image.
2. The system performs external web/social-media discovery.
3. Candidate content is collected from the search results.
4. Faces in the candidates are compared with the input using SFace.
5. The matching content is fingerprinted using SHA-256.
6. The fingerprint is recorded on the local Hardhat blockchain.
7. The system retrieves the on-chain fingerprint and compares it with the newly generated hash.
8. The result confirms whether the data is **verified or has been modified**.

## ⛓️ Blockchain Verification

The blockchain stores the **SHA-256 fingerprint**, not the original image.

```text
Content → SHA-256 → Blockchain

Content → SHA-256 → Compare with Blockchain
                         ↓
                  ✅ Match / ❌ Mismatch
```

This provides a tamper-evident record of the discovered data.

## ⚠️ Limitations

* Face recognition can produce false positives or false negatives.
* Web/social-media search results depend on external availability and restrictions.
* Blockchain verification confirms data integrity, but does not by itself prove the truth or ownership of the content.
* The current blockchain setup uses a **local Hardhat network** for demonstration.


