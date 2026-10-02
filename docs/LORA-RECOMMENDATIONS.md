# LoRA recommendations

Popular Civitai LoRAs for the base models the app ships with.

1. [Detailers](#1-detailers)
2. [Style improvement](#2-style-improvement)
3. [GPT-image-2 style, including Krea 2](#3-gpt-image-2-style-including-krea-2)
4. [Art Nouveau](#4-art-nouveau)
5. [Ivan Bilibin](#5-ivan-bilibin)
6. [Speed LoRAs](#6-speed-loras)
7. [Adding a LoRA to the app](#7-adding-a-lora-to-the-app)

A LoRA only works on the base it was trained for, so each table says which of
your models it fits:

| Base | Your models |
|---|---|
| Illustrious | WAI, JANKU, RIN Flanime, One Obsession, CyberRealistic Illustrious |
| Anima | CyberRealistic Anima, LustrousMix Anima, BAS+ Anima |
| SDXL | Juggernaut XL Ragnarok, PerfectDeliberate |
| Krea 2 | Krea 2 Turbo |
| Flux.2 Klein 9B | FLUX.2 klein 9B |
| ZImageTurbo | Z-Image Turbo |
| Qwen Image Edit 2511 | Qwen Edit (Edit by instruction, Combine) |

Download counts are Civitai's all-time figures, taken from its API on
2026-09-18. They show popularity, not quality, and none of these has been
rendered in this app yet.

## 1. Detailers

| LoRA | Base | Downloads | Notes |
|---|---|---|---|
| [Detail Tweaker XL](https://civitai.com/models/122359) | SDXL | 452k | The standard SDXL detailer. Slider: about −1 to +2 |
| [Detailed Perfection style](https://civitai.com/models/411088) | Illustrious, SDXL | 133k | All-in-one: hands, feet, face, body |
| [Detail Slider LoRA](https://civitai.com/models/1001945) | Illustrious | 43k | Slider: positive weight adds detail, negative removes it |
| [Detailer IL](https://civitai.com/models/1231943) | Illustrious | 37k | |
| [Anima Highres/Aesthetic Boost](https://civitai.com/models/2540444) | Anima | 38k | The closest thing to a detailer for Anima; a search found no dedicated one |
| [Detailifier](https://civitai.com/models/430687) | Krea 2, SDXL | 28k | Older detailer with a Krea 2 version |
| [[KREA 2] Detail Slider](https://civitai.com/models/2729908) | Krea 2 | 18k | Made for Krea 2 |

## 2. Style improvement

**General quality and look**

| LoRA | Base | Downloads | Notes |
|---|---|---|---|
| [Stabilizer IL/NAI](https://civitai.com/models/971952) | Illustrious | 156k | Cleans up anatomy and coherence rather than adding a look; the usual first LoRA to try |
| [xl_more_art-full / Enhancer](https://civitai.com/models/124347) | SDXL | 128k | General aesthetic boost |
| [Illustrious Style Pack](https://civitai.com/models/1060551) | Illustrious | 113k | |
| [AI styles dump](https://civitai.com/models/723360) | Anima, Illustrious | 69k | Style collection |
| [Niji Sweet Spot](https://civitai.com/models/2554999) | Anima | 24k | |

**Specific looks**

| LoRA | Base | Downloads | Notes |
|---|---|---|---|
| [薄塗り / USNR](https://civitai.com/models/176554) | Illustrious | 91k | Flat, thin-paint anime |
| [748cm](https://civitai.com/models/943607) | Illustrious | 64k | Painterly |
| [Juggernaut Cinematic XL](https://civitai.com/models/120663) | SDXL | 57k | By Juggernaut's author |
| [Niji semi realism](https://civitai.com/models/534506) | Illustrious | 53k | Toward semi-realism |
| [Illustrious Realism Slider](https://civitai.com/models/1486904) | Illustrious | 53k | Slider toward realism |

**Photographic realism.** These push away from illustration.

| LoRA | Base | Downloads |
|---|---|---|
| [Realistic Skin Texture](https://civitai.com/models/580857) | SDXL | 186k |
| [Lenovo UltraReal](https://civitai.com/models/1662740) | Krea 2 | 168k |
| [Realistic Snapshot](https://civitai.com/models/2268008) | Krea 2, Z-Image Turbo | 80k |
| [Famegrid](https://civitai.com/models/2088956) | Krea 2, Z-Image, Qwen | 45k |
| [Phone Photography (2000–2025)](https://civitai.com/models/2537408) | Krea 2 | 21k |

## 3. GPT-image-2 style, including Krea 2

| LoRA | Base | Downloads | Notes |
|---|---|---|---|
| [Style > gpt-image-2](https://civitai.com/models/2564225) | Anima | 18k | By far the most used; Anima only |
| [Ps GPT-image-2 style for Krea2](https://civitai.com/models/2760678) | Krea 2 | 1.4k | Made for Krea 2 only, so the more direct fit there |
| [GPT-Image-2 Anime Illustration Style](https://civitai.com/models/2736538) | Krea 2, Anima, Illustrious | 0.9k | One style across three of your bases |

Both Krea 2 options are new and lightly used.

## 4. Art Nouveau

### [Art Nouveau](https://civitai.com/models/562604)

14.9k downloads, 1.2k likes. By far the most used, with a version for nearly
every base here. Its "MJ" versions were trained on Midjourney-style images, so
expect a polished decorative poster rather than a period print.

**Trigger:** `ArsMJStyle, art nouveau`

| Your models | Version to download |
|---|---|
| Krea 2 Turbo | Krea2 |
| FLUX.2 klein 9B | Klein9B |
| Z-Image Turbo | ZImageTurbo |
| Anima models | Anima |
| Illustrious models | Illustrious (older: IllustriousMJ7) |
| SDXL models | SDXLv2 (older: SDXLMJ7) |

No Qwen Image version is listed.

### [Art Nouveau Style](https://civitai.com/models/182980)

6k downloads, 730 likes. Flatter and more graphic, close to vector art. The
Illustrious version's own example prompt names Aubrey Beardsley, Ivan Bilibin
and Alphonse Mucha, which makes it the likely closer match for a Bilibin-style
picture.

| Version | Trigger |
|---|---|
| Illustrious | `art nouveau, fine art parody` |
| Anima | `art_nouveau, fine_art_parody` |
| Z-Image Turbo | `Art Nouveau` |

### Smaller ones

Few downloads, so less proven.

| LoRA | Base | Downloads | Notes |
|---|---|---|---|
| [Disney Nouveau](https://civitai.com/models/1030673) | Illustrious, SDXL | 2.7k | Disney-style animation with Art Nouveau ornament |
| [Mucha painting style](https://civitai.com/models/193756) | SDXL | 1.9k | Mucha specifically |
| [Nouveau Ukiyo](https://civitai.com/models/2801742) | Anima | 158 | Art Nouveau crossed with Japanese woodblock prints |
| [Nouveau Filigree SDXL](https://civitai.com/models/2337688) | SDXL | 66 | Ornamental borders and frames |

## 5. Ivan Bilibin

| LoRA | Base | Downloads |
|---|---|---|
| [Ivan Bilibin Style](https://civitai.com/models/69496) | SDXL, Flux.1 D | 2.5k |
| [Ivan Bilibin Style](https://civitai.com/models/2782465) | Krea 2 | 325 |
| [Ivan Bilibin – Traditional Russian Woodblock Print](https://civitai.com/models/2728522) | Illustrious | 136 |

**Where to start:** for a Bilibin-style illustration, try **Art Nouveau Style**
on an Illustrious model first.

## 6. Speed LoRAs

These don't change the look on purpose. They let a model render in far fewer
steps, usually at CFG 1, which also switches off the negative prompt.

| LoRA | Base | Status |
|---|---|---|
| Qwen Image Edit 2511 Lightning 4-Step | Qwen Image Edit 2511 | **Set up in the app**, on by default: edits take 4 steps instead of 16. Get it from Draw Things' LoRA list |
| [Anima Turbo LoRA](https://civitai.com/models/2560840) | Anima | **Set up in the app** for LustrousMix and CyberRealistic Anima, on by default: 8 steps at CFG 1. BAS+ Anima has it built in |
| [Wuli-art Qwen-Image-2512 Turbo (4 steps)](https://civitai.com/models/2267786?modelVersionId=2552570) | Qwen Image 2512 | **Set up in the app**, on by default: 4 steps at CFG 1, about 4× faster with the same look. Skin comes out harsher |
| DMD2, Hyper SDXL 8-Step, SDXL Lightning 8-Step | SDXL, Illustrious | **Not recommended.** On the Illustrious models they draw the subject two or three times at the final size, and on drafts their saturated poster look carries through into Faithful |
| Qwen Image 2512 Lightning 4-Step (Draw Things' list) | Qwen Image 2512 | **Doesn't work here**: with a real prompt at 1024px the render never finishes. Use the Wuli-art Turbo above instead |

The ticks are in the **LoRA** column of the ⚙ dialog, and
[Guide](GUIDE.md#speed-loras) says what each one costs: the Lightning LoRA
does replacements badly, and Anima Turbo changes the look and drops the
negative prompt.

## 7. Adding a LoRA to the app

The composer has no LoRA picker. A LoRA is set per model in
`config/models.yaml`, in one of two ways.

Always on, per stage:

```yaml
loras: [{ file: "detail_tweaker_xl_f16.ckpt", weight: 0.8 }]
```

Or as a tick in the ⚙ dialog, the way the speed LoRAs are. It names the
stages it applies to, and anything it changes there:

```yaml
lora_options:
  - key: detail
    label: "Detail Tweaker"
    file: "detail_tweaker_xl_f16.ckpt"
    weight: 0.8
    default: false          # ticked from the start, or not
    stages: { final: {} }   # e.g. { final: { steps: 8, cfg: 1.0 } }
```

- Import the file into Draw Things first. If an always-on LoRA is not
  installed, the app hides that model's chip; a tick just goes grey.
- Start at a weight of about 0.5–0.8.
- Judge it on the Faithful and final renders as well as on the drafts: a LoRA
  can change the whole look of a model (see the DMD2 notes in `models.yaml`).
- Most style LoRAs need their **trigger words** in the prompt. Enrich will not
  add them by itself, so type them into your own prompt before you enrich and
  check that the enriched variants kept them.
- Your Krea 2 chip is **Krea 2 Turbo**, and most Krea 2 LoRAs were probably
  trained on the base model. On a few-step turbo model a base-trained LoRA can
  come out weaker or muddier. Compare the same prompt and seed with and without
  it.
