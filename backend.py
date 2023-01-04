"""backend framework for the voice cloning api"""
from subprocess import Popen, PIPE
from pydantic import BaseModel
from typing import Union, Optional, Any, List
import argparse
import json
import os
import string
import time
import sys
from pathlib import Path
import torch
import numpy as np

class GeneralBashCommands(BaseModel):
    """bash commands base class"""
    main_tts_clone: str = "git clone https://github.com/Edresson/TTS -b dev-gst-embeddings"
    install_espeak: str = 'brew install espeak'
    install_requirements: str = 'pip install -r requirements.txt'
    run_setup_py: str = "python setup.py develop"
    main_wget: str = "wget -c -q --show-progress -O ./TTS-checkpoint.zip https://github.com/Edresson/TTS/releases/" \
                     "download/v1.0.0/Checkpoints-TTS-MultiSpeaker-Jia-et-al-2018-with-GST-CorentinJ_SpeakerEncoder" \
                     "_and_DDC.zip"
    unzip_tts_main_folder: str = 'unzip ./TTS-checkpoint.zip'
    gst_wget: str = 'wget https://github.com/Edresson/TTS/releases/download/v1.0.0/gst-style-example.wav'
    load_ext: str = 'load_ext autoreload'
    load_ext_autoreload_backup: str = 'autoreload 2'


class CustomBashCommands(BaseModel):
    """custom bash commands for specific TTS application (non-generic voice)"""
    clone_encoder: str = "git clone https://github.com/Edresson/GE2E-Speaker-Encoder.git"
    encoder_requirements_installation: str = "python -m pip install umap-learn visdom webrtcvad " \
                                             "librosa>=0.5.1 matplotlib>=2.0.2 nump"
    download_encoder_checkpoint: str = "wget https://github.com/Edresson/Real-Time-Voice-Cloning/releases/download/" \
                                       "checkpoints/pretrained.zip"
    unzip_pretrained: str = "unzip pretrained.zip"


class ModelVars(BaseModel):
    """model variables for initiation and live usage"""
    TTS_PATH: str = "../content/TTS"
    MODEL_PATH: str = 'best_model.pth.tar'
    CONFIG_PATH: str = 'config.json'
    SPEAKER_JSON: str = 'speakers.json'
    OUT_PATH: str = 'tests-audios/'
    TEXT: str = ''
    SPEAKER_FILEID: Any = None
    VOCODER_PATH: str = ''
    VOCODER_CONFIG_PATH: str = ''
    USE_CUDA: str = False
    custom_model_path: str = "encoder/saved_models/pretrained.pt"


# command runner
def run_command(command: str, expect_output: bool = False) -> Any:
    """run an intended command in the bash environment and output the response"""

    # define a process to run in the bash environment
    process = Popen(command.split(), stdout=PIPE)

    # run the command and handle the output
    if expect_output:
        output, error = process.communicate()
        return output, error
    else:
        process.communicate()
        return None, None


class GeneralProjectSetup:
    """setup the project by cloning the backend contents from the TTS AI  providers"""

    def __init__(self, version: str = '1.0.0'):
        self.version: str = version
        self.command_model = GeneralBashCommands()

    def _switch_directory_func(self, directory: str) -> None:
        """function to switch directory"""

        # create bash command to switch directory
        switch_cd_command: str = f"os.chdir('{directory}')"
        run_command(switch_cd_command, expect_output=False)

    def _base_tts_clone(self):
        """run the subprocess bash command to clone the main tts clone to the current directory"""

        # run abstracted tts installation commands
        run_command(self.command_model.main_tts_clone, expect_output=False)
        run_command(self.command_model.install_espeak, expect_output=False)
        self._switch_directory_func("TTS")
        run_command(self.command_model.install_requirements, expect_output=False)
        run_command(self.command_model.run_setup_py, expect_output=False)
        self._switch_directory_func('..')

    def _wget_action(self):
        """run wget commands to install and fine tune the TTS model"""

        # run abstracted tts commands
        output, error = run_command(self.command_model.main_wget, expect_output=True)
        if error:
            raise BrokenPipeError

        # unzip the folder produced by download
        run_command(self.command_model.unzip_tts_main_folder, expect_output=False)

        # get the .wav gst file from wget download
        output, error = run_command(self.command_model.gst_wget, expect_output=True)
        if error:
            raise BrokenPipeError

    def _load_ext_setup_commands(self):
        """final commands to prepare download for further processing"""

        # run load ext commands
        run_command(self.command_model.load_ext, expect_output=False)
        run_command(self.command_model.load_ext_autoreload_backup, expect_output=False)

    def facade(self):
        """facade method to run full download"""
        self._base_tts_clone()
        self._wget_action()
        self._load_ext_setup_commands()


class CustomProjectSetup:
    """project setup for custom voice TTS application"""

    def __init__(self, version: str = '1.0.0'):
        self.version: str = version
        self.custom_command_model = CustomBashCommands()
        sys.path.append("../content/GE2E-Speaker-Encoder/")

    def custom_facade(self):
        """setup by running commands on bash to clone and install requirements"""

        # run the commands
        run_command(self.custom_command_model.clone_encoder, expect_output=False)
        run_command(self.custom_command_model.encoder_requirements_installation, expect_output=False)
        run_command(self.custom_command_model.download_encoder_checkpoint, expect_output=False)
        run_command(self.custom_command_model.unzip_pretrained, expect_output=False)


class Utils(GeneralProjectSetup, CustomProjectSetup):

    def __init__(self):
        super().__init__(version='1.0.0')
        self.facade() # run the download upon class creation
        self.custom_facade() # run custom downloads for non-generic voice mimicking
        self.model_vars = ModelVars()
        sys.path.append(self.model_vars.TTS_PATH) # set path

        # TTS_GAN functionality from backend
        self.synthesis: Any = None
        self.setup_model: Any = None
        self.make_symbols: Any = None
        self.phonemes: Any = None
        self.symbols: Any = None
        self.AudioProcessor: Any = None
        self.load_config: Any = None
        self.setup_generator: Any = None

        # custom autoencoder-GAN based model variables
        self.encoder: Any = None
        self.speaker_embedding_size: Any = None

    def conditional_imports(self):
        """import voice cloning TTS tools after installation is complete"""
        try:
            from TTS.tts.utils.generic_utils import setup_model
            from TTS.tts.utils.synthesis import synthesis
            from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
            from TTS.utils.audio import AudioProcessor
            from TTS.utils.io import load_config
            from TTS.vocoder.utils.generic_utils import setup_generator

            # apply to global variables
            self.synthesis = synthesis
            self.setup_model: Any = setup_model
            self.make_symbols: Any = make_symbols
            self.phonemes: Any = phonemes
            self.symbols: Any = symbols
            self.AudioProcessor: Any = AudioProcessor
            self.load_config: Any = load_config
            self.setup_generator: Any = setup_generator

        except ImportError:
            raise BrokenPipeError

    def custom_conditional_imports(self):
        """conditional imports for custom model variables"""
        try:
            from encoder import inference as encoder
            from encoder.params_model import model_embedding_size as speaker_embedding_size

            # custom voice cloning model variables assignment
            self.encoder: Any = encoder
            self.speaker_embedding_size: Any = speaker_embedding_size
        except ImportError:
            raise BrokenPipeError

    def tts(self, model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None,
            gst_style=None, verbose=False):
        t_1 = time.time()
        waveform, _, _, mel_postnet_spec, _, _ = self.synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid,
                        gst_style, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)
        if CONFIG.model == "Tacotron" and not use_gl:
            mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
        if not use_gl:
            waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
        if use_cuda and not use_gl:
            waveform = waveform.cpu()
        if not use_gl:
            waveform = waveform.numpy()
        waveform = waveform.squeeze()
        rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
        tps = (time.time() - t_1) / len(waveform)
        if verbose:
            print(" > Run-time: {}".format(time.time() - t_1))
            print(" > Real-time factor: {}".format(rtf))
            print(" > Time per step: {}".format(tps))
        return waveform


class GeneralMimic(Utils):
    """live voice cloning functionality in one class"""

    def __init__(self):
        super().__init__()
        self.conditional_imports() # run conditional imports method
        self.speaker_embedding = None
        self.speaker_embedding_dim = None
        self.num_speakers: int = 0

    def _load_backend_audio_processors(self) -> List[Any, Any]:
        """load the backend configurations and processors for the audio"""

        C = self.load_config(self.model_vars.CONFIG_PATH)
        C.forward_attn_mask = True  # set forward attention masking for GAN base

        # create the base audio processor with expanding C parameters
        ap = self.AudioProcessor(**C.audio)

        # initialize vocabulary corpus
        if 'characters' in C.keys():
            self.symbols, self.phonemes =self. make_symbols(**C.characters)

        return [C, ap]

    def _load_speakers(self) -> List[Any, Any, Any, Any]:
        """load in speakers"""

        # get C and ap from the
        C, ap = self._load_backend_audio_processors()

        if self.model_vars.SPEAKER_JSON != '':
            speaker_mapping = json.load(open(self.model_vars.SPEAKER_JSON, 'r'))
            num_speakers = len(speaker_mapping)
            if C.use_external_speaker_embedding_file:
                if self.model_vars.SPEAKER_FILEID is not None:
                    self.speaker_embedding = speaker_mapping[self.model_vars.SPEAKER_FILEID]['embedding']
                else:  # if speaker_fileid is not specified use the first sample in speakers.json
                    choice_speaker = list(speaker_mapping.keys())[0]
                    print(" Speaker: ", choice_speaker.split('_')[0], 'was chosen automatically',
                          "(this speaker seen in training)")
                    self.speaker_embedding = speaker_mapping[choice_speaker]['embedding']
                speaker_embedding_dim: Any = len(self.speaker_embedding)
            else:
                speaker_embedding_dim: Any = None
        else:
            speaker_embedding_dim: Any = None

        # pass on variables to model creation stage
        return [C, ap, speaker_embedding_dim, self.speaker_embedding]

    def _load_model(self) -> List[Any, Any, Any]:
        """load vocoder model for preprocessing before GAN layer"""

        # fetch model creation variables
        C, ap, speaker_embedding_dim, speaker_embedding = self._load_speakers()

        # setup the model on pytorch base adn loading in all states
        num_chars = len(self.phonemes) if C.use_phonemes else len(self.symbols)
        model = self.setup_model(num_chars, self.num_speakers, C, speaker_embedding_dim)
        cp = torch.load(self.model_vars.MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(cp['model'])
        model.eval()

        # run and train model with cuda if selected
        if self.model_vars.USE_CUDA:
            model.cuda()
        model.decoder.set_r(cp['r'])

        return [C, ap, model, speaker_embedding]

    def _load_vocoder(self) -> List[Any, Any]:
        """create the vocoder model in full and deploy to inference stage"""
        if self.model_vars.VOCODER_PATH != "":
            VC = self.load_config(self.model_vars.VOCODER_CONFIG_PATH)
            vocoder_model = self.setup_generator(VC)
            vocoder_model.load_state_dict(torch.load(self.model_vars.VOCODER_PATH, map_location="cpu")["model"])
            vocoder_model.remove_weight_norm()
            if self.model_vars.USE_CUDA:
                vocoder_model.cuda()
            vocoder_model.eval()
        else:
            vocoder_model = None
            VC = None

        return [VC, vocoder_model]

    def voice_synthesize(self, TEXT: str, verbose: bool = False) -> None:
        """synthesize an example voice"""

        # fetch initiated model
        C, ap, model, speaker_embedding = self._load_model()
        VC, vocoder_model = self._load_vocoder()

        # define general model parameters
        gst_style = {"0": 0, "1": 0, "3": 0, "4": 0}
        use_griffin_lim = self.model_vars.VOCODER_PATH == ""

        # read the embedding file and define the speaker id code to use (p244)
        if not C.use_external_speaker_embedding_file:
            if self.model_vars.SPEAKER_FILEID.isdigit():
                SPEAKER_FILEID = int(self.model_vars.SPEAKER_FILEID)
            else:
                SPEAKER_FILEID = None
        else:
            SPEAKER_FILEID = None

        # infer the wavelet from the Vocoder-GAN-TTS model
        wav = self.tts(model, vocoder_model, TEXT, C, self.model_vars.USE_CUDA, ap, use_griffin_lim, SPEAKER_FILEID,
                  speaker_embedding=speaker_embedding, gst_style=gst_style)

        # save to produced wavelet to file to later output from API endpoint
        file_name = TEXT.replace(" ", "_")
        file_name = file_name.translate(
            str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(self.model_vars.OUT_PATH, file_name)
        print(" > Saving output to {}".format(out_path) if verbose else None)
        ap.save_wav(wav, out_path)


class CustomMimic(GeneralMimic):
    """custom voice cloning application"""

    def __init__(self, file_list: list):
        super().__init__()
        self.custom_conditional_imports()
        self.encoder.load_model(Path(self.model_vars.custom_model_path))
        self.file_list: Any = file_list

    def autoencoder(self, verbose: bool = False):
        """autoencoder method to use the GAN voice cloner"""

        # create the wavelet from the encoder random sample
        wav = np.zeros(self.encoder.sampling_rate)
        embed = self.encoder.embed_utterance(wav)
        print(embed.shape if verbose else None)

        speaker_embeddings = []
        for name in self.file_list.keys():
            if '.wav' in name:
                preprocessed_wav = self.encoder.preprocess_wav(name)
                embed = self.encoder.embed_utterance(preprocessed_wav)
                # embed = se_model.compute_embedding(mel_spec).cpu().detach().numpy().reshape(-1)
                speaker_embeddings.append(embed)
            else:
                print("You need upload Wav files, others files is not supported !!")

        # takes the average of the embeddings samples of the announcers
        speaker_embedding = np.mean(np.array(speaker_embeddings), axis=0).tolist()
        return speaker_embedding

    def custom_voice_synthesize(self, TEXT: str) -> None:
        """synthesize the custom voice embedding"""

        # get the speaker embeddings
        speaker_embedding = self.autoencoder()
        gst_style: dict = {"0": 0, "1": 0.0, "3": 0, "4": 0}

        C, ap, model, speaker_embedding = self._load_model()
        VC, vocoder_model = self._load_vocoder()

        use_griffin_lim = self.model_vars.VOCODER_PATH == ""

        # read the embedding file and define the speaker id code to use (p244)
        if not C.use_external_speaker_embedding_file:
            if self.model_vars.SPEAKER_FILEID.isdigit():
                SPEAKER_FILEID = int(self.model_vars.SPEAKER_FILEID)
            else:
                SPEAKER_FILEID = None
        else:
            SPEAKER_FILEID = None

        wav = self.tts(model, vocoder_model, TEXT, C, self.model_vars.USE_CUDA, ap, use_griffin_lim, SPEAKER_FILEID,
                  speaker_embedding=speaker_embedding, gst_style=gst_style)

        # save wave results
        file_name = TEXT.replace(" ", "_")
        file_name = file_name.translate(
            str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(self.model_vars.OUT_PATH, file_name)
        ap.save_wav(wav, out_path)