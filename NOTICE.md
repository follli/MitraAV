# Credits and third-party terms

MitraAV's own code is MIT-licensed (see `LICENSE`). It does not bundle the models below. The Add-ons dialog and `mitraav-setup` download them on
request from their publishers; you are using them under the publishers' terms. Check the upstream
licence files before redistributing anything or using MitraAV commercially.

| Component | Used for | Terms (as understood; verify upstream) |
|---|---|---|
| YAMNet (Google; trained on AudioSet) | `sounds` add-on | Apache-2.0 |
| Places365 (Zhou et al., MIT CSAIL, places2.csail.mit.edu) | `scene` add-on | code MIT; pretrained models CC BY: **credit the Places Project**. The category and indoor/outdoor label files in `src/mitraav/references/places365_*.txt` come from the MIT-licensed repository |
| BirdNET (Cornell Lab of Ornithology and Chemnitz University of Technology) | `birds` add-on | code MIT; **models CC BY-NC-SA 4.0 (non-commercial)**, per aggregator pages, not confirmed from the upstream files |
| BioCLIP (Imageomics Institute) and `pybioclip` | `nature` add-on | MIT (weights and package); the species embeddings come from the TreeOfLife-10M dataset on Hugging Face: check its terms |
| PyTorch, open_clip | `nature` add-on | open-source licences; installed by pip on request |
| `jpeglib` (Martin Benes) | `stego` add-on | MPL-2.0; installed by pip on request (it bundles libjpeg builds) |
| Steganalysis methods: chi-square attack (Westfeld and Pfitzmann), RS analysis (Fridrich, Goljan and Du), calibration attack on F5 (Fridrich, Goljan and Hogea) | `hidden_data_*` probes | published methods, re-implemented from the papers; no code copied |
| Tesseract, FFmpeg, ExifTool | system tools | installed by you; not bundled |
| Mains-frequency, telephone-tone and calling-code tables | reference data in `src/mitraav/references/` | compiled from public standards and references; nominal values, verify against the cited sources |

Follow-up links point to third-party sites; MitraAV does not control their content.
