# Intellectual Property Protection

**Doc-Type:** IP Verification Guide · Version 1.0 · AI Whisperers

This directory contains cryptographic proof of prior art for the Ternary VAE project, anchored to the Bitcoin blockchain via OpenTimestamps.

---

## Files

| File | Purpose |
|------|---------|
| `MANIFEST.md` | Core scientific claims with timestamps |
| `MANIFEST.md.ots` | Bitcoin blockchain proof for manifest |
| `HASHES.json` | SHA-256 hashes of all critical files |
| `HASHES.json.ots` | Bitcoin blockchain proof for hashes |
| `HASHES.txt` | Human-readable hash summary |
| `generate_hashes.py` | Script to regenerate hash manifest |

---

## Verification

### 1. Verify Bitcoin Timestamp (Web)

1. Go to https://opentimestamps.org
2. Drop `HASHES.json.ots`
3. Drop `HASHES.json`
4. Site confirms: **"Bitcoin block 933927 attests data existed as of Jan 26, 2026"**

### 2. Verify Bitcoin Timestamp (CLI)

```bash
# Install OpenTimestamps client
pipx install opentimestamps-client

# Verify (requires Bitcoin node, or use web interface)
ots verify docs/ip-protection/HASHES.json.ots

# View attestation info
ots info docs/ip-protection/HASHES.json.ots
```

### 3. Verify File Hashes

```bash
# Verify a single file
sha256sum src/models/ternary_vae.py
# Compare with value in HASHES.json

# Regenerate full manifest
python docs/ip-protection/generate_hashes.py --project-root ../..
```

---

## Bitcoin Attestation Details

| Field | Value |
|-------|-------|
| **Block Height** | [933927](https://blockstream.info/block/000000000000000000012b4426db629fb960dff6f791a2fadda1e8dabefd1700) |
| **Block Time** | 2026-01-26 23:56:57 UTC-3 |
| **Merkle Root** | `6bee4a04c625474536e63289b1b7b59db037520138c5e14b6b112b2a4a01ca22` |
| **Calendar Servers** | btc.calendar.catallaxy.com, alice/bob.btc.calendar.opentimestamps.org |

---

## Protected Claims

1. **P-adic Hyperbolic VAE** - 3-adic structure in Poincare ball
2. **TrainableCodonEncoder** - LOO rho=0.61 on DDG prediction
3. **13-adic Viral Discovery** - DENV-4 evolutionary geometry (R²=0.96)
4. **Dual-Metric Framework** - Shannon + hyperbolic conservation
5. **Contact Prediction** - Codon embeddings predict 3D contacts (AUC=0.67)

---

## Regenerating Hashes

After significant changes, regenerate the hash manifest:

```bash
cd /path/to/ternary-vaes-bioinformatics
python docs/ip-protection/generate_hashes.py \
    --output docs/ip-protection/HASHES.json \
    --summary docs/ip-protection/HASHES.txt

# Create new blockchain timestamp
ots stamp docs/ip-protection/HASHES.json

# Commit and push
git add docs/ip-protection/
git commit -m "chore(ip): update hash manifest"
git push
```

---

*This directory establishes cryptographic proof of prior art for all innovations in this repository.*
