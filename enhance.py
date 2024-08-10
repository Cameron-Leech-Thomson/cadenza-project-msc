""" Run the dummy enhancement. """

from __future__ import annotations

import json
import logging
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as lines
import librosa.display
import audio_dspy as adsp
import re
import subprocess
import random
import os
import soundfile as sf

from numpy import ndarray
from omegaconf import DictConfig
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.evaluator.haaqi import compute_haaqi
from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal
from clarity.utils.flac_encoder import FlacEncoder
from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    resample,
    to_16bit,
    compute_rms,
)
from clarity.utils.source_separation_support import get_device, separate_sources
from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    apply_ha,
    make_scene_listener_list,
    remix_stems,
    load_reference_stems,
)

logger = logging.getLogger(__name__)

def save_flac_signal(
    signal: np.ndarray,
    filename: Path,
    signal_sample_rate,
    output_sample_rate,
    do_clip_signal: bool = False,
    do_soft_clip: bool = False,
    do_scale_signal: bool = False,
) -> None:
    """
    Function to save output signals.

    - The output signal will be resample to ``output_sample_rate``
    - The output signal will be clipped to [-1, 1] if ``do_clip_signal`` is True
        and use soft clipped if ``do_soft_clip`` is True. Note that if
        ``do_clip_signal`` is False, ``do_soft_clip`` will be ignored.
        Note that if ``do_clip_signal`` is True, ``do_scale_signal`` will be ignored.
    - The output signal will be scaled to [-1, 1] if ``do_scale_signal`` is True.
        If signal is scale, the scale factor will be saved in a TXT file.
        Note that if ``do_clip_signal`` is True, ``do_scale_signal`` will be ignored.
    - The output signal will be saved as a FLAC file.

    Args:
        signal (np.ndarray) : Signal to save
        filename (Path) : Path to save signal
        signal_sample_rate (int) : Sample rate of the input signal
        output_sample_rate (int) : Sample rate of the output signal
        do_clip_signal (bool) : Whether to clip signal
        do_soft_clip (bool) : Whether to apply soft clipping
        do_scale_signal (bool) : Whether to scale signal
    """
    # Resample signal to expected output sample rate
    if signal_sample_rate != output_sample_rate:
        signal = resample(signal, signal_sample_rate, output_sample_rate)

    if do_scale_signal:
        # Scale stem signal
        max_value = np.max(np.abs(signal))
        signal = signal / max_value

        # Save scale factor
        with open(filename.with_suffix(".txt"), "w", encoding="utf-8") as file:
            file.write(f"{max_value}")

    elif do_clip_signal:
        # Clip the signal
        signal, n_clipped = clip_signal(signal, do_soft_clip)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")

    # Convert signal to 16-bit integer
    signal = to_16bit(signal)

    # Create flac encoder object to compress and save the signal
    FlacEncoder().encode(signal, output_sample_rate, filename)


# pylint: disable=unused-argument
def decompose_signal(
    model: torch.nn.Module,
    model_sample_rate: int,
    signal: ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: list[str],
    listener: Listener,
    normalise: bool = True,
) -> dict[str, ndarray]:
    """
    Decompose signal into 8 stems.

    The listener is ignored by the baseline system as it
     is not performing personalised decomposition.
    Instead, it performs a standard music decomposition using a pre-trained
     model trained on the MUSDB18 dataset.

    Args:
        model (torch.nn.Module): Torch model.
        model_sample_rate (int): Sample rate of the model.
        signal (ndarray): Signal to be decomposed.
        signal_sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        sources_list (list): List of strings used to index dictionary.
        listener (Listener).
        normalise (bool): Whether to normalise the signal.

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    if signal_sample_rate != model_sample_rate:
        signal = resample(signal, signal_sample_rate, model_sample_rate)

    if normalise:
        signal, ref = normalize_signal(signal)

    sources = separate_sources(
        model,
        torch.from_numpy(signal.astype(np.float32)),
        model_sample_rate,
        device=device,
    )

    # only one element in the batch
    sources = sources[0]
    if normalise:
        sources = denormalize_signals(sources, ref)

    sources = np.transpose(sources, (0, 2, 1))
    return dict(zip(sources_list, sources))

"""
Convert a dB gain to linear gain scale for audio_dspy.
"""
def db_to_linear_gain(d):
    x = d / 20
    return np.power(10, x)

"""
Convert a linear gain scale to dB gain.
"""
def linear_to_db_gain(g):
    return 20 * np.log10(g)

# Get all files in a given directory:
def filesInPath(path):
    for cPath, _, filenames in os.walk(path):
        for filename in filenames:
            yield os.path.join(cPath, filename)
    
"""
(https://www.armadamusic.com/university/music-production-articles/eq-explained-the-basics)

EQ Frequency Bands:
Sub Bass: 20Hz - ~60Hz
Bass: 60Hz - ~250Hz
Low Mids: 250Hz - ~1500Hz
High Mids: 1500Hz - ~4kHz
Presence: 4kHz - ~7kHz
Brilliance/Noise: 7kHz - ~20kHz

Shelf Filter:
Q in Shelf Filter: Transition band - Smaller = More gradual transition.

Pass Filter:
High Pass Filter / Low Cut Filter: Cuts anything of lower freq.
Low Pass Filter / High Cut Filter: Cuts anything of higher freq.
Pass filters can filter out entire sounds. Steep gradient down.
Q in Pass Filter: Decibels per Octave. Slope of falloff.

Parametric Filter:
Bell-curve style filter. Frequency = centre of curve. Gain = peak of curve. Q = bandwidth - How wide(low Q)/narrow(high Q) the curve is.
"""
# DEFAULT Frequency Q Factors:
lowq = 0.75
subq = 0.8
bassq = 0.9
lmidq = 1
hmidq = 0.9
presq = 1.2
noiseq = 1.5
topq = 0.8

# DEFAULT Frequency Gains in dB:
subg = 12
bassg = 9
lmidg = 5
hmidg = 10
presg = 5
noiseg = -1
topg = -6

def_q_factors = [ subq, bassq, lmidq, hmidq, presq, noiseq, topq ]
def_gains = [ subg, bassg, lmidg, hmidg, presg, noiseg, topg ]

def equaliser(
        sample_rate,
        q_factors = None,
        gains = None,
        importParams = False,
) -> adsp.eq.EQ:
    # Frequency Bands:
    LOW = 5
    SUB = 20
    BASS = 60
    LMID = 250
    HMID = 1500
    PRES = 4000
    NOISE = 7000
    TOP = 20000

    BANDS = [ SUB, BASS, LMID, HMID, PRES, NOISE, TOP ]

    if q_factors == None:
        q_factors = def_q_factors
    if gains == None:
        gains = def_gains
    
    save_dir = "<PATH-TO-CADENZA-PROJECT>/clarity/recipes/cad_icassp_2024/baseline/"
    
    if importParams:
        # Get most recent file from eqtraining directory:
        if os.path.isdir(save_dir + "eqtraining/"):
            files = filesInPath(save_dir + "eqtraining/")
            if len(list(files)) > 0:
                paramsFile = max(files, key=os.path.getmtime)
                eqParams = []
                with open(paramsFile, 'r') as jsonFile:
                    eqData = json.load(jsonFile)
                    eqParams = eqData['Individual']

                # Convert 'individual' array to gains and q arrays:
                importedGains = []
                importedQs = []

                lowq = eqParams[0]
                eqParams = eqParams[1:]
                for i in range(len(BANDS)):
                    print(2*i,2*i+1)
                    importedQs.append(eqParams[ 2 * i ])
                    importedGains.append(eqParams[ 2 * i + 1])

                gains = importedGains
                q_factors = importedQs
            else:
                logger.info(f"No files in directory: <{save_dir + "eqtraining/"}>.")
        else:
            logger.info(f"Directory: <{save_dir + "eqtraining/"}> Does not exist.")
    
    # Remove any negative q-factors
    for i in range(len(q_factors)):
        if q_factors[i] == 0:
            q_factors[i] = 0.1
        if q_factors[i] < 0:
            q_factors[i] = np.abs(q_factors[i])

    # Initialise EQ Instance
    eq = adsp.eq.EQ(sample_rate)

    ### Add Filters for Each Frequency Band:
    # High Pass Filter: Cut off frequencies lower than 20Hz.
    eq.add_HPF(LOW, lowq)

    # Parametric Filters: Increase/decrease gains centred in each frequency band.
    for i in range(len(BANDS) - 1):
        # Add Parametric (Bell) Filter: (Midpoint Between Frequency Bands, Q Factor, Gain in Linear Units)
        mp = ( BANDS[i + 1] + BANDS[i] ) / 2
        g = db_to_linear_gain(gains[i])
        eq.add_bell(int(np.round(mp)), q_factors[i], g)

    # High Shelf Filter: Reduce gain on frequencies above 20kHz.
    eq.add_highshelf(TOP, q_factors[-1], db_to_linear_gain(gains[-1]))
    #eq.add_LPF(TOP, topq)
    
    eq.reset()
    
    eq.print_eq_info()

    eqFig = plt.figure(figsize=(16, 6))
    eq.plot_eq_curve(worN=np.logspace(-1, 4, num=2000, base=20))

    colours = [ 'r', 'g', 'y' ]
    for i in range(len(BANDS) - 1):
        mp = ( BANDS[i + 1] + BANDS[i] ) / 2
        db = gains[i]
        sign = '+' if db >= 0 else ''
        plt.axvline(x = np.round(mp), color = colours[i % 3], linestyle = '--', linewidth = 0.75,
                    label = f"{BANDS[i]}-{BANDS[i + 1]} [{sign}{db}dB/{q_factors[i]}] ({int(np.round(mp))})")

    plt.axhline(y = 1, color = 'k', linestyle = '--', linewidth = 0.5, alpha = 0.5)
    plt.gca().set_ylim(bottom = 0)
    plt.gca().set_xlim(right = 20000, left = 1)
    
    plt.legend(loc='upper left', title="Frequency Bands [Gain/Q] (Centre)")
    
    ylabels = [ x.get_text() for x in plt.gca().get_yticklabels() ]
    for i in range(len(ylabels)):
        val = float(re.sub(r'[^\x00-\x7F]+','-', ylabels[i]))
        if val <= 0:
            continue
        db = np.round(linear_to_db_gain(val), 1)
        sign = '+' if db >= 0 else ''
        ylabels[i] = ylabels[i] + f"({sign}{db}dB)"
    
    plt.gca().set_yticks(plt.gca().get_yticks()[1:-1], ylabels[1:-1], rotation=45)
    plt.ylabel("Magnitude [dB] (Gain [+/-dB])")
    
    plt.tight_layout()

    # eq.print_eq_info()
    eqFig.savefig(f"{save_dir}EQCurve.png")
    plt.close()

    logger.info("EQ Created & EQ Curve Saved")

    return eq

def process_remix_for_listener(
    signal: ndarray,
    enhancer: NALR,
    compressor: Compressor,
    listener: Listener,
    apply_compressor: bool = False,
) -> ndarray:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        sample_rate (float) : Sample rate of the signal
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        listener: Listener object
        apply_compressor (bool) : Whether to apply the compressor
    Returns:
        ndarray: Processed signal.
    """
    left_output = apply_ha(
        enhancer, compressor, signal[:, 0], listener.audiogram_left, apply_compressor
    )
    right_output = apply_ha(
        enhancer, compressor, signal[:, 1], listener.audiogram_right, apply_compressor
    )

    return np.stack([left_output, right_output], axis=1)

def display_stem_waveforms(stems, sr, savedir):
    stemFig = plt.figure(figsize=(18,16))
    plot = 1
    for stem, signal in stems.items():
        side = [ 'Left', 'Right' ]
        col = [ 'r', 'b' ]
            
        for i in range(2):
            # Waveforms:
            plt.subplot(4, 4, plot + i)
            librosa.display.waveshow(signal[:, i], sr=sr, color=col[i])
            plt.title(f"{stem.capitalize()} {side[i]} Signal")

            # Spectrogram:
            plt.subplot(4, 4, plot + i + 4)
            stft = librosa.stft(signal[:, i])  # STFT of signal
            decibels = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            librosa.display.specshow(decibels, sr=sr, y_axis='linear', x_axis='time')
            ax = plt.gca()
            ax.set_ylim(bottom = 0)
            ylabels = [ '{:.1f}'.format(x) for x in ax.get_yticks()/1000 ]
            ax.set_yticks(ax.get_yticks().tolist())
            ax.set_yticklabels(ylabels)
            
            if (plot + i + 4) % 4 != 1:
                plt.ylabel(None)
            else:
                plt.ylabel(f"{ax.yaxis.get_label().get_text()} ($10^3$)")
                
            plt.title(f"{stem.capitalize()} {side[i]} Frequencies")
        
        if plot == 3:
            plot = 9
        else:    
            plot = plot + 2

    plt.tight_layout()
    stemFig.add_artist(lines.Line2D([0.4111, 0.4111], [0, 1], color='k'))
    stemFig.add_artist(lines.Line2D([0, 0.8111], [0.5, 0.5], color='k'))
    
    plt.colorbar(location='right', format="%+2.f dB", ax=stemFig.axes)
    stemFig.savefig(savedir)
    plt.close()

    logger.info("Saved.")

BATCH_SIZE = 10

@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig, batch_size=None) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocal, drums, bass, and other stems.
    Then, the NAL-R prescription procedure is applied to each stem.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """

    # Set the output directory where processed signals will be saved
    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Loading pretrained source separation model
    if config.separator.model == "demucs":
        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = separation_model.sources
        normalise = True
    elif config.separator.model == "openunmix":
        separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
        model_sample_rate = separation_model.sample_rate
        sources_order = ["vocals", "drums", "bass", "other"]
        normalise = False
    else:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    #
    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Create EQ:
    eq = equaliser(model_sample_rate)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    previous_song = ""
    num_scenes = len(scene_listener_pairs)
    if BATCH_SIZE == 0:
        batch_size = num_scenes
    else:
        batch_size = BATCH_SIZE + 1
    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

        if idx >= batch_size:
            logger.info(f"Batch (Size {batch_size}) Complete.")
            break

        logger.info(
            f"[{idx:03d}/{batch_size:03d}] "
            f"Processing {scene_id}: {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_id]

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        if song_name != previous_song:
            sn = songs[song_name]["Path"]
            mixture_signal = read_signal(
                filename= f"{config.path.music_dir}/{sn}/mixture.wav",
                sample_rate=config.sample_rate,
                allow_resample=True,
            )

            stems: dict[str, ndarray] = decompose_signal(
                model=separation_model,
                model_sample_rate=model_sample_rate,
                signal=mixture_signal,
                signal_sample_rate=config.sample_rate,
                device=device,
                sources_list=sources_order,
                listener=listener,
                normalise=normalise,
            )

            stems = apply_gains(stems, config.sample_rate, gains[scene["gain"]])

            # savePath = f"<PATH-TO-CADENZA-PROJECT>clarity/recipes/cad_icassp_2024/baseline/waveforms/stems_{scene_id}_{listener.id}"
            # display_stem_waveforms(stems, model_sample_rate, savePath + "_pre_eq")

            for stem, signal in stems.items():
                # Apply EQ to both outputs:
                logger.info(f"Applying EQ to {stem}...")
                eq.reset()
                # Normalise signal to avoid overflow:
                # original_max_value = np.max(np.abs(signal[:, 0]))
                # normalised_signal = signal[:, 0] / original_max_value
                stems[stem][:, 0] = eq.process_block(signal[:, 0])# * original_max_value

                eq.reset()
                # original_max_value = np.max(np.abs(signal[:, 1]))
                # normalised_signal = signal[:, 1] / original_max_value
                stems[stem][:, 1] = eq.process_block(signal[:, 1])# * original_max_value
                                
            # display_stem_waveforms(stems, model_sample_rate, savePath + "_post_eq")
            
            enhanced_signal = remix_stems(stems, mixture_signal, model_sample_rate)

        enhanced_signal = process_remix_for_listener(
            signal=enhanced_signal,
            enhancer=enhancer,
            compressor=compressor,
            listener=listener,
            apply_compressor=config.apply_compressor,
        )

        filename = Path(enhanced_folder) / f"{scene_id}_{listener.id}_remix.flac"

        filename.parent.mkdir(parents=True, exist_ok=True)
        save_flac_signal(
            signal=enhanced_signal,
            filename=filename,
            signal_sample_rate=config.sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )

    logger.info("Done!")

def extract_features(sr, audio_path=None, signal=None):
    if signal is not None:
        y = signal
    else:
        y, _ = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    features = np.hstack((
        np.mean(mfccs, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(zcr, axis=1),
        np.mean(spectral_centroid, axis=1),
        np.mean(mel_spectrogram, axis=1)
    ))
    
    return features

from deap import(
    base,
    creator,
    tools,
    algorithms,
)

def initIndividual(icls):
    q_factors = [ 0.75, 0.8, 0.9, 1, 0.9, 1.2, 2.5, 0.8 ]
    gains = [ 12, 9, 5, 10, 5, 4, -6 ]

    q_offsets = [ x/10 for x in range(-7, 11) ]
    gain_offsets = [ x/2 for x in range(-10, 11) ]

    # Apply offsets to filters:
    q_vals = random.sample(q_offsets, len(q_factors))
    g_vals = random.sample(gain_offsets, len(gains))

    individual = []

    for i in range(len(q_factors)):
        if i == 0:
            val = round(q_factors[i] + q_vals[i], 2)
            individual.append(val)
        else:
            qVal = round(q_factors[i] + q_vals[i], 2)
            individual.append(qVal)
            gVal = round(gains[i-1] + g_vals[i-1], 2)
            individual.append(gVal)
    
    return icls(individual)

def individual_to_eq_params(individual):
    filters = ['low','sub','bass','lmid','hmid','pres','noise','top']
    eqParams = {filters[0] + '_q': individual[0]}
    
    values = list(individual)[1:]
    # Extract Q values and gain values from the individual list
    for i in range(0, len(filters) - 1):
        eqParams[filters[1:][i] + '_q'] = values[2*i]
        eqParams[filters[1:][i] + '_g'] = values[2*i + 1]
    
    return eqParams

def clipSongs(signal):
    # 10% of 180 seconds is enough for a sample:
    SEC = 0.1
    segmentLength = len(signal[0]) * SEC
    # Get start position of sample:
    start = random.randint(0, len(signal[0]) - segmentLength - 1)
    end = start + segmentLength
    return np.stack(signal[ start : end, 0 ], signal[ start : end, 1 ], axis=1)

SAMPLE_SIZE = 10

def create_dataset(config: DictConfig, individual):
    results = "<PATH-TO-CADENZA-PROJECT>//clarity//recipes//cad_icassp_2024//baseline//eqtraining//"

    eqParams = individual_to_eq_params(individual)

    # Loading pretrained source separation model
    if config.separator.model == "demucs":
        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = separation_model.sources
        normalise = True
    elif config.separator.model == "openunmix":
        separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
        model_sample_rate = separation_model.sample_rate
        sources_order = ["vocals", "drums", "bass", "other"]
        normalise = False
    else:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    with Path(str(config.path.gains_file)).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    with Path(str(config.path.scenes_file)).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    with Path(str(config.path.scene_listeners_file)).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    with Path(str(config.path.music_file)).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    songSamples = scene_listener_pairs[:SAMPLE_SIZE]#random.sample(scene_listener_pairs, SAMPLE_SIZE)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    previous_song = ""
    num_scenes = len(songSamples)

    # print(songSamples)

    qVals = []
    gVals = []

    for key, value in eqParams.items():
        if key[-2:] == "_q":
            qVals.append(value)
        if key[-2:] == "_g":
            gVals.append(value)

    eq = equaliser(model_sample_rate, qVals, gVals)

    haaqiDict = {}

    for idx, scene_listener_pair in enumerate(songSamples, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = f"{scene['music']}-{scene['head_loudspeaker_positions']}"

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Processing {scene_id}: {song_name} for listener {listener_id}"
        )

        # Get the listener's audiogram
        listener = listener_dict[listener_id]

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        if song_name != previous_song:
            sn = songs[song_name]["Path"]
            mixture_signal = read_signal(
                filename= f"{config.path.music_dir}/{sn}/mixture.wav",
                sample_rate=config.sample_rate,
                allow_resample=True,
            )

            stems: dict[str, ndarray] = decompose_signal(
                model=separation_model,
                model_sample_rate=model_sample_rate,
                signal=mixture_signal,
                signal_sample_rate=config.sample_rate,
                device=device,
                sources_list=sources_order,
                listener=listener,
                normalise=normalise,
            )

            stems = apply_gains(stems, config.sample_rate, gains[scene["gain"]])

            logger.info("Loading Reference Signals...")

            # Load reference signals
            reference_stems, original_mixture = load_reference_stems(
                Path(config.path.music_dir) / songs[song_name]["Path"]
            )
            reference_stems = apply_gains(
                reference_stems, config.sample_rate, gains[scene["gain"]]
            )
            reference_mixture = remix_stems(
                reference_stems, original_mixture, config.sample_rate
            )

            logger.info("Done. Applying HA to signals...")

            # Apply hearing aid to reference signals
            left_reference = apply_ha(
                enhancer=enhancer,
                compressor=None,
                signal=reference_mixture[:, 0],
                audiogram=listener.audiogram_left,
                apply_compressor=False,
            )
            right_reference = apply_ha(
                enhancer=enhancer,
                compressor=None,
                signal=reference_mixture[:, 1],
                audiogram=listener.audiogram_right,
                apply_compressor=False,
            )            

            logger.info("Done. Remixing stems...")

            enhanced_signal = remix_stems(stems, mixture_signal, model_sample_rate)

            logger.info("Done. Applying EQ...")

            eq.reset()
            left_signal = eq.process_block(enhanced_signal[:, 0])
            eq.reset()
            right_signal = eq.process_block(enhanced_signal[:, 1])
            eq.reset()

            logger.info("Done. Applying HA to EQ Signals...")

            eq_signal = np.stack([left_signal, right_signal], axis=1)
        
        remixedSignal = process_remix_for_listener(
            signal=eq_signal,
            enhancer=enhancer,
            compressor=compressor,
            listener=listener,
            apply_compressor=config.apply_compressor,
        )

        logger.info("Done. Computing HAAQI Scores...")

        # Compute the scores
        left_score = compute_haaqi(
            processed_signal=resample(
                remixedSignal[:, 0],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                left_reference, config.sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_left,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 0])),
        )

        right_score = compute_haaqi(
            processed_signal=resample(
                remixedSignal[:, 1],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                right_reference, config.sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_right,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
        )        

        logger.info("Left Score: {:.6f}, Right Score: {:.6f}, Mean: {:.6f}".format(left_score, right_score,
                                                                             np.mean([left_score, right_score])))
        haaqiDict[f"{scene_id}: {song_name}, Listener: {listener_id}"] = {"LeftScore": left_score,
                                                                "RightScore": right_score,
                                                                "MeanScore": np.mean([left_score, right_score])}
    
    logger.info(haaqiDict)

    with open(Path(f"{results}scores_{round(np.mean(list(individual)), 3)}.json"), 'w') as f:
        json.dump({"Individual": individual, "HAAQI": haaqiDict}, f, indent=6)

    leftMean = np.mean([ x['LeftScore'] for x in list(haaqiDict.values()) ])
    rightMean = np.mean([ x['LeftScore'] for x in list(haaqiDict.values()) ])

    return (leftMean, rightMean)

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

def wrapper(config):
    def evaluate(individual):
        return create_dataset(config, individual)
    return evaluate

@hydra.main(config_path="", config_name="config")
def run_ga(config: DictConfig):
    toolbox = base.Toolbox()

    # Define the attributes for the individual
    toolbox.register("attr_q_gain", initIndividual, creator.Individual)

    # Initialize the individual using the custom function
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_q_gain)

    # Define the population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the crossover, mutation, and selection functions
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Define the evaluation function
    toolbox.register("evaluate", wrapper(config))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    hall_of_fame = tools.HallOfFame(3) 

    # Create the population
    population = toolbox.population(n=50)  # Set the population size

    # Define the number of generations and probabilities
    ngen = 50
    cxpb = 0.5
    mutpb = 0.2

    # Run the algorithm
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
                        stats=stats, halloffame=hall_of_fame, verbose=True)
    
    for gen in range(ngen):
        logger.info(f"Generation {gen + 1}:")
        for i, individual in enumerate(hall_of_fame):
            logger.info(f"{i + 1}. Individual: {individual}, Fitness: {individual.fitness.values[0]}")
        print()

# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    # run_ga() will run the genetic algorithm on samples of the dataset:
    run_ga()
    # enhance() will run the enhancement procedure on the entire database, using the most recent individual:
    # enhance()
