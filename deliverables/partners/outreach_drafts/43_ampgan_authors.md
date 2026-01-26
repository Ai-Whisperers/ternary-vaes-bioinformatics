# Email Draft: AMPGAN v2 Authors

**Status:** NEEDS EMAIL RESEARCH

**Email:** (via ACS correspondence - check paper for corresponding author)

---

## Recipient Research

| Field | Details |
|-------|---------|
| **Name** | AMPGAN v2 Authors |
| **Position** | Various institutions |
| **Key Paper** | AMPGAN v2 - GAN for AMP generation (ACS 2021) |
| **Recent Focus** | Generative adversarial networks for peptides |
| **Code** | Check if GitHub available |

---

## EMAIL

**To:** (corresponding author from paper)

**Subject:** GAN vs. VAE for AMP generation - different failure modes?

---

Hi,

Your AMPGAN v2 work is one of the cleaner GAN applications to peptide generation. We've been doing similar work with VAEs (hyperbolic latent space) and wondered about failure mode differences.

GANs tend to have mode collapse issues; VAEs tend to blur. For AMP generation, do you see AMPGAN sometimes getting stuck generating variants of the same few sequences? We see our VAE occasionally producing "average" peptides that are mediocre at everything.

Curious whether the failure modes are actually complementary - if so, a simple ensemble (generate with both, filter for agreement) might outperform either alone.

Is your code/model available for comparison experiments?

Best,
Ivan Weiss Van Der Pol
Jonathan Verdun
Kyrian Weiss Van Der Pol
AI Whisperers | CONACYT Paraguay
github.com/Ai-Whisperers/ternary-vaes-bioinformatics

---

## Notes

- **Word count:** 122
- **Why this works:** Technical peer discussion, specific question about their experience, proposes actionable collaboration
- **Tone:** Developer-to-developer

---

*Last Updated: 2026-01-26*
