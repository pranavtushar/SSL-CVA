# Child-Centric Voice Anonymization in Single and Multi-Speaker Speech via Domain-Adapted SSL Models

Supplementary audio samples for the paper (**INTERSPEECH 2025**).

## Authors

- Pranav Tushar (Singapore Institute of Technology)
- Xiao Xiao Miao (Duke Kunshan University)
- Rong Tong (Singapore Institute of Technology)

Contact: pranav.tushar@singaporetech.edu.sg, xiaoxiao.miao@dukekunshan.edu.cn, tong.rong@singaporetech.edu.sg

## What is this?

Voice anonymization aims to protect speaker identity while preserving linguistic content and speech usability. However, most anonymization systems are developed on adult speech, leading to degraded performance when applied to child speech. This paper investigates child-centric anonymization by adapting a self-supervised learning (SSL) based anonymization pipeline to the child speech domain. The system is adapted using child speech from the MyST corpus and evaluated under both single-speaker and two-speaker mixture conditions. Experimental results show that child-domain adaptation improves intelligibility and perceptual quality while maintaining strong privacy protection. Extending the approach to multi-speaker further demonstrates that combining target speaker extraction with child-adapted anonymization provides privacy protection while preserving conversational structure. These findings highlight the importance of child-specific adaptation for practical speech anonymization systems.

## How to listen

- **Recommended**: open `index.html` (it has audio players and the full layout).
- On GitHub: click any `wav` link below; GitHub will open the audio file page where you can play/download it.

### Local preview (works in Docker too)

Run a simple static server from the `docs/` folder:

```bash
cd /app/SSL-CVA/docs   # inside Docker
# or: cd /mnt/hdd/pranav/SSL-CVA/docs   # on host
python3 -m http.server 8000
```

Then open `http://localhost:8000` (you should see the web player and the tables).

## Single-speaker child anonymization

Methods: **Original**, **B2** (McAdams baseline), **SSL-Base** (adult-trained), **SSL-FT** (child fine-tuned).

| Sample | Original | B2 | SSL-Base | SSL-FT |
|---|---:|---:|---:|---:|
| MyST (1) | [wav](final/1/myst/myst_002116_2014-02-27_09-29-01_LS_1.1_004/original.wav) | [wav](final/1/myst/myst_002116_2014-02-27_09-29-01_LS_1.1_004/B2.wav) | [wav](final/1/myst/myst_002116_2014-02-27_09-29-01_LS_1.1_004/ssl_base.wav) | [wav](final/1/myst/myst_002116_2014-02-27_09-29-01_LS_1.1_004/ssl_ft.wav) |
| MyST (2) | [wav](final/1/myst/f_myst_002119_2014-03-05_09-46-10_LS_1.3_003/original.wav) | [wav](final/1/myst/f_myst_002119_2014-03-05_09-46-10_LS_1.3_003/B2.wav) | [wav](final/1/myst/f_myst_002119_2014-03-05_09-46-10_LS_1.3_003/ssl_base.wav) | [wav](final/1/myst/f_myst_002119_2014-03-05_09-46-10_LS_1.3_003/ssl_ft.wav) |
| SpeechOcean (1) | [wav](final/1/speechocean/050390001/original.wav) | [wav](final/1/speechocean/050390001/B2.wav) | [wav](final/1/speechocean/050390001/ssl_base.wav) | [wav](final/1/speechocean/050390001/ssl_ft.wav) |
| SpeechOcean (2) | [wav](final/1/speechocean/060670003/original.wav) | [wav](final/1/speechocean/060670003/B2.wav) | [wav](final/1/speechocean/060670003/ssl_base.wav) | [wav](final/1/speechocean/060670003/ssl_ft.wav) |
| MPS (1) | [wav](final/1/mps/3a11h_EN-OL-RC-234_2/original.wav) | [wav](final/1/mps/3a11h_EN-OL-RC-234_2/B2.wav) | [wav](final/1/mps/3a11h_EN-OL-RC-234_2/ssl_base.wav) | [wav](final/1/mps/3a11h_EN-OL-RC-234_2/ssl_ft.wav) |
| MPS (2) | [wav](final/1/mps/4a31k_EN-OL-RC-426_2/original.wav) | [wav](final/1/mps/4a31k_EN-OL-RC-426_2/B2.wav) | [wav](final/1/mps/4a31k_EN-OL-RC-426_2/ssl_base.wav) | [wav](final/1/mps/4a31k_EN-OL-RC-426_2/ssl_ft.wav) |

## Two-speaker mixtures (target anonymization)

**Mixture** = original mixture; **Anonymized** = target speaker anonymized (non-target unchanged).

### AA (Adult–Adult)

| Overlap | Mixture | Anonymized |
|---:|---:|---:|
| 0% | [wav](final/2/AA/0/000002_s1-libri_8230_s2-libri_2300_ov0_mix.wav) | [wav](final/2/AA/0/000002_s1-libri_8230_s2-libri_2300_ov0_anon.wav) |
| 20% | [wav](final/2/AA/20/000000_s1-libri_1995_s2-libri_4077_ov20.mix.wav) | [wav](final/2/AA/20/000000_s1-libri_1995_s2-libri_4077_ov20.mix_anon.wav) |
| 40% | [wav](final/2/AA/40/000000_s1-libri_5639_s2-libri_4970_ov40.mix.wav) | [wav](final/2/AA/40/000000_s1-libri_5639_s2-libri_4970_ov40.mix_anon.wav) |
| 60% | [wav](final/2/AA/60/000001_s1-libri_7021_s2-libri_4970_ov60.mix.wav) | [wav](final/2/AA/60/000001_s1-libri_7021_s2-libri_4970_ov60.mix_anon.wav) |
| 80% | [wav](final/2/AA/80/000000_s1-libri_908_s2-libri_7176_ov80.mix.wav) | [wav](final/2/AA/80/000000_s1-libri_908_s2-libri_7176_ov80.mix_anon.wav) |
| 100% | [wav](final/2/AA/100/000000_s1-libri_8455_s2-libri_4446_ov100.mix.wav) | [wav](final/2/AA/100/000000_s1-libri_8455_s2-libri_4446_ov100.mix_anon.wav) |

### CA (Child–Adult)

| Overlap | Mixture | Anonymized |
|---:|---:|---:|
| 0% | [wav](final/2/CA/0/000002_s1-myst_014028_s2-libri_1188_ov0_mix.wav) | [wav](final/2/CA/0/000002_s1-myst_014028_s2-libri_1188_ov0_anon.wav) |
| 20% | [wav](final/2/CA/20/000001_s1-myst_990502_s2-libri_5105_ov20.mix.wav) | [wav](final/2/CA/20/000001_s1-myst_990502_s2-libri_5105_ov20.mix_anon.wav) |
| 40% | [wav](final/2/CA/40/000001_s1-myst_007081_s2-libri_61_ov40.mix.wav) | [wav](final/2/CA/40/000001_s1-myst_007081_s2-libri_61_ov40.mix_anon.wav) |
| 60% | [wav](final/2/CA/60/000002_s1-myst_997487_s2-libri_6930_ov60.mix.wav) | [wav](final/2/CA/60/000002_s1-myst_997487_s2-libri_6930_ov60.mix_anon.wav) |
| 80% | [wav](final/2/CA/80/000000_s1-myst_014020_s2-libri_8230_ov80.mix.wav) | [wav](final/2/CA/80/000000_s1-myst_014020_s2-libri_8230_ov80.mix_anon.wav) |
| 100% | [wav](final/2/CA/100/000001_s1-myst_996712_s2-libri_4507_ov100.mix.wav) | [wav](final/2/CA/100/000001_s1-myst_996712_s2-libri_4507_ov100.mix_anon.wav) |

### CC (Child–Child)

| Overlap | Mixture | Anonymized |
|---:|---:|---:|
| 0% | [wav](final/2/CC/0/000013_s1-myst_005016_s2-myst_999459_ov0_mix.wav) | [wav](final/2/CC/0/000013_s1-myst_005016_s2-myst_999459_ov0_anon.wav) |
| 20% | [wav](final/2/CC/20/000000_s1-myst_996433_s2-myst_997690_ov20.mix.wav) | [wav](final/2/CC/20/000000_s1-myst_996433_s2-myst_997690_ov20.mix_anon.wav) |
| 40% | [wav](final/2/CC/40/000000_s1-myst_996725_s2-myst_996712_ov40.mix.wav) | [wav](final/2/CC/40/000000_s1-myst_996725_s2-myst_996712_ov40.mix_anon.wav) |
| 60% | [wav](final/2/CC/60/000000_s1-myst_007081_s2-myst_014028_ov60.mix.wav) | [wav](final/2/CC/60/000000_s1-myst_007081_s2-myst_014028_ov60.mix_anon.wav) |
| 80% | [wav](final/2/CC/80/000000_s1-myst_008005_s2-myst_007077_ov80.mix.wav) | [wav](final/2/CC/80/000000_s1-myst_008005_s2-myst_007077_ov80.mix_anon.wav) |
| 100% | [wav](final/2/CC/100/000001_s1-myst_996716_s2-myst_013001_ov100.mix.wav) | [wav](final/2/CC/100/000001_s1-myst_996716_s2-myst_013001_ov100.mix_anon.wav) |

---

_Auto-generated from `supplementary.html`._
