# Voice Cloning Model with Zero-Shot Attention-Based TTS 

# The AI used in this API is the YourTTS Zero-Shot Multispeaker TTS implementation of generative audio modeling. 

The [paper](https://arxiv.org/abs/2112.02418) that proposed the YourTTS model was used as a central building block of the API. 
YourTTS for a multilingual approach for zero-shot multi-speaker TTS which can be utilized on multilingual audio data while building on older VITS approaches.

## Reference Implementations used to study TTS concepts can be found [here](https://github.com/coqui-ai/tts)

## The Models Researched under open source as provided from Coqui

| Model                        | URL                                                                                            |
|------------------------------|------------------------------------------------------------------------------------------------|
| Speaker Encoder              | [link](https://drive.google.com/drive/folders/1WKK70aBnA-ZI2Z1Ka_zWgBK7O0Y3TLey?usp=sharing)   |
| Exp 1. YourTTS-EN(VCTK)         | [link](https://drive.google.com/drive/folders/15MfBCRyM8ViZ5Ghe16bGz0UtB_O0iovX?usp=sharing)   |
| Exp 1. YourTTS-EN(VCTK)  + SCL         | [link](https://drive.google.com/drive/folders/10hX5B_h0dzroY7bVNPodC8hzhz4nklzR?usp=sharing)   |
| Exp 2. YourTTS-EN(VCTK)-PT          | [link](https://drive.google.com/drive/folders/1Mdob20nFQEKUwavw_DhqMc1MfG3CNNNI?usp=sharing) |
| Exp 2. YourTTS-EN(VCTK)-PT  + SCL   | [link](https://drive.google.com/drive/folders/1uorMp_A0LNEfwdkM1QB9jz4Mf3kM5sGn?usp=sharing) |
| Exp 3. YourTTS-EN(VCTK)-PT-FR       | [link](https://drive.google.com/drive/folders/15NAhIeHXJZLxrZMoCaUlH_7mS7TRqdme?usp=sharing) |
| Exp 3. YourTTS-EN(VCTK)-PT-FR SCL   | [link](https://drive.google.com/drive/folders/1H7VrW6eUO0wle-e6Un3mp77udkZLMrMr?usp=sharing) |
| Exp 4. YourTTS-EN(VCTK+LibriTTS)-PT-FR SCL | [link](https://drive.google.com/drive/folders/15G-QS5tYQPkqiXfAdialJjmuqZV0azQV?usp=sharing) |

## TTS Retraining Data

The audios for the MOS are available [here](https://github.com/Edresson/YourTTS/releases/download/MOS/Audios_MOS.zip). 
Also, the MOS the audios are [here](https://github.com/Edresson/YourTTS/tree/main/metrics/MOS).

### Default TTS Audio Sources:
  LibriTTS (test clean): 1188, 1995, 260, 1284, 2300, 237, 908, 1580, 121 and 1089
  
  VCTK: p261, p225, p294, p347, p238, p234, p248, p335, p245, p326 and p302
  
  MLS Portuguese:  12710, 5677, 12249, 12287, 9351, 11995, 7925, 3050, 4367 and 1306


## Citation

```

@ARTICLE{2021arXiv211202418C,
  author = {{Casanova}, Edresson and {Weber}, Julian and {Shulby}, Christopher and {Junior}, Arnaldo Candido and {G{\"o}lge}, Eren and {Antonelli Ponti}, Moacir},
  title = "{YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone}",
  journal = {arXiv e-prints},
  keywords = {Computer Science - Sound, Computer Science - Computation and Language, Electrical Engineering and Systems Science - Audio and Speech Processing},
  year = 2021,
  month = dec,
  eid = {arXiv:2112.02418},
  pages = {arXiv:2112.02418},
  archivePrefix = {arXiv},
  eprint = {2112.02418},
  primaryClass = {cs.SD},
  adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv211202418C},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
