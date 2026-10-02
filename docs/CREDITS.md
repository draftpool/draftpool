# Credits

draftpool connects other people's work together: the image generation,
upscaling and prompt writing are all done by the projects below. Each model has
its own license, set by its authors. This project's MIT license covers none of
them (see the [README](../README.md#license)).

**Apps**
- [Draw Things](https://drawthings.ai/) by Liu Liu: the on-device image
  generation app and API server that does all the rendering.
- [Ollama](https://ollama.com/): runs the local LLMs that write prompts and
  describe images.
- [Civitai](https://civitai.com/): hosts the community models below.

**Prompt-writing LLMs**
- [Gemma 3](https://ai.google.dev/gemma) (Google DeepMind): the default.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (Alibaba Qwen team): the
  vision model.

**Image models from Draw Things' catalogue**
- [FLUX.2 \[klein\] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  and [FLUX.1 \[schnell\]](https://huggingface.co/black-forest-labs/FLUX.1-schnell):
  Black Forest Labs.
- [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo): Alibaba
  Tongyi-MAI.
- [Krea 2 Turbo](https://www.krea.ai/models/krea-2-turbo): Krea AI.
- [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) and
  [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511):
  Alibaba Qwen team. The editor behind Edit and Combine.

**Video**
- [Wan 2.1 I2V 14B](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) (Wan-AI,
  Alibaba), in the
  [FusionX](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX) merge by
  vrgamedevgirl84. Some parts of the merge are licensed for non-commercial use
  only.

**Community models (Civitai)**
- [LustrousMix Anima v1.0](https://civitai.com/models/941345?modelVersionId=2982979): Hoseki.
- [BAS+ Better Anime Style Plus](https://civitai.com/models/2675648?modelVersionId=3004325): news_v_com.
- [CyberRealistic Anima](https://civitai.com/models/2668799?modelVersionId=3136380): Cyberdelia.
- [WAI-illustrious-SDXL v17.0](https://civitai.com/models/827184?modelVersionId=2883731): WAI0731.
- [CyberRealistic CyberIllustrious Anime v5.0](https://civitai.com/models/2081804?modelVersionId=2886849): Cyberdelia.
- [One obsession v24](https://civitai.com/models/1318945?modelVersionId=3218603): maxfeifei8.
- [JANKU Illustrious XL v7.77](https://civitai.com/models/1277670?modelVersionId=2786084): janxd.
- [RIN Flanime Illustrious v5.0](https://civitai.com/models/1544647?modelVersionId=3262295): Rin_x0.
- [PerfectDeliberate-Anime v5.0_A](https://civitai.com/models/111274?modelVersionId=2925672): Desync.
- [Juggernaut XL Ragnarok](https://civitai.com/models/133005?modelVersionId=1759168): KandooAI / RunDiffusion.

**Speed LoRAs**
- [Qwen-Image-Edit-2511 Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning): LightX2V.
- [Qwen-Image-2512 Turbo](https://civitai.com/models/2267786?modelVersionId=2552570): Wuli-art.
- [Anima Turbo LoRA](https://civitai.com/models/2560840?modelVersionId=2979642): circlestone_labs.

**Upscaling**
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (Xintao Wang et al.):
  the **2x fast** upscaler, via its ncnn-vulkan build.
- 4x UltraSharp (Kim2091): the upscaler used by **Upscale**, from Draw Things'
  catalogue.

**Other**
- [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) by
  nagadomi (MIT), with [OpenCV](https://opencv.org/): finds faces for **Fix
  faces**.
- [Danbooru](https://danbooru.donmai.us/): the tag list behind the tag checks
  and autocomplete.
