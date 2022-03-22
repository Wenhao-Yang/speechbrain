#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: cnceleb_prepare.py
@Time: 2022/3/19 23:48
@Overview:
"""
"""

Data preparation.

Download: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
"""

import os
import csv
import logging
import glob
import pdb
import random
import shutil
import sys  # noqa F401
import numpy as np
import torch
import soundfile as sf
import torchaudio
from tqdm.contrib import tqdm
from multiprocessing import Pool, Manager, Process
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_cnceleb_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000


DEV_WAV = "vox1_dev_wav.zip"
TEST_WAV = "vox1_test_wav.zip"
META = "meta"


def prepare_cnceleb(
    data_folder,
    save_folder,
    verification_pairs_file,
    splits=["train", "dev", "test"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    source=None,
    split_speaker=False,
    random_segment=False,
    skip_prep=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the README.md file for
    preparing Voxceleb2.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VoxCeleb dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    split_speaker : bool
        Speaker-wise split
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from recipes.VoxCeleb.voxceleb1_prepare import prepare_voxceleb
    >>> data_folder = 'data/VoxCeleb1/'
    >>> save_folder = 'VoxData/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_opt = os.path.join(save_folder, OPT_FILE)
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)

    # Create the data folder contains VoxCeleb1 test data from the source
    if source is not None:
        if not os.path.exists(os.path.join(data_folder, "eval", "test")):
            logger.info(f"Missing eval/test in {data_folder}")
            # shutil.unpack_archive(os.path.join(source, TEST_WAV), data_folder)

        if not os.path.exists(os.path.join(data_folder, "eval", "lists", 'enroll.lst')):
            logger.info(f"Missing enroll list in {data_folder}")
            # shutil.copytree(
            #     os.path.join(source, "meta"), os.path.join(data_folder, "meta")
            # )

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = data_folder.replace(" ", "").split(",")
    else:
        data_folder = [data_folder]

    # _check_voxceleb1_folders(data_folder, splits)

    msg = "\tCreating csv file for the CNCeleb Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, verification_pairs_file, split_speaker
    )

    # pdb.set_trace()
    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)

    # For PLDA verification
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder, save_folder, verification_pairs_file
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, save_opt)


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
        "test": TEST_CSV,
        "enrol": ENROL_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def _check_voxceleb_folders(data_folders, splits):
    """
    Check if the data folder actually contains the Voxceleb1 dataset.

    If it does not, raise an error.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
    """
    for data_folder in data_folders:

        if "train" in splits:
            folder_vox1 = os.path.join(data_folder, "wav", "id10001")
            folder_vox2 = os.path.join(data_folder, "wav", "id00012")

            if not os.path.exists(folder_vox1) or not os.path.exists(
                folder_vox2
            ):
                err_msg = "the specified folder does not contain Voxceleb"
                raise FileNotFoundError(err_msg)

        if "test" in splits:
            folder = os.path.join(data_folder, "wav", "id10270")
            if not os.path.exists(folder):
                err_msg = (
                    "the folder %s does not exist (as it is expected in "
                    "the Voxceleb dataset)" % folder
                )
                raise FileNotFoundError(err_msg)

        folder = os.path.join(data_folder, "meta")
        if not os.path.exists(folder):
            err_msg = (
                "the folder %s does not exist (as it is expected in "
                "the Voxceleb dataset)" % folder
            )
            raise FileNotFoundError(err_msg)


# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, verification_pairs_file, split_speaker=False
):
    """
    Tot. number of speakers cnc1= 797.
    Tot. number of speakers cnc2= 1997.
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the training and dev set (if any).
    """
    train_lst = []
    dev_lst = []
    # pdb.set_trace()
    print("Getting file list...")
    for data_folder in data_folders:

        test_lst = [
            line.rstrip("\n").split(" ")[1]
            for line in open(verification_pairs_file)
        ]
        test_lst = set(sorted(test_lst))
        print('There are %d utterances in test trials!' % (len(test_lst)))

        test_spks = set([snt.split("/")[1].split('-')[0] for snt in test_lst])
        # test_spks = set([snt.split("-")[0] for snt in test_lst])
        # print(test_spks.pop())
        print('There are %d spks in test set!' % (len(test_spks)))

        path = os.path.join(data_folder, "data", "**", "*.flac")
        if split_speaker:
            # avoid test speakers for train and dev splits
            audio_files_dict = {}
            for f in glob.glob(path, recursive=True):
                spk_id = f.split("/data/")[1].split("/")[0]
                if spk_id not in test_spks:
                    audio_files_dict.setdefault(spk_id, []).append(f)
            # pdb.set_trace()
            spk_id_list = list(audio_files_dict.keys())
            random.shuffle(spk_id_list)
            split = int(0.01 * split_ratio[0] * len(spk_id_list))
            for spk_id in spk_id_list[:split]:
                train_lst.extend(audio_files_dict[spk_id])

            for spk_id in spk_id_list[split:]:
                dev_lst.extend(audio_files_dict[spk_id])
        else:
            # avoid test speakers for train and dev splits
            audio_files_list = []
            audio_files_dict = {}
            train_snts = []
            dev_snts = []

            pbar = tqdm(glob.glob(path, recursive=True), ncols=100)
            for f in pbar:
                try:
                    spk_id = f.split("/data/")[1].split("/")[0]
                except ValueError:
                    logger.info(f"Malformed path: {f}")
                    continue
                if spk_id not in test_spks:
                    audio_files_dict.setdefault(spk_id, []).append(f)

                    # audio_files_list.append(f)

            print('There are %d spks in train set!' % (len(audio_files_dict)))
            train_spk = set()
            dev_spk = set()
            for spk_id in audio_files_dict:
                spk_id_utts = audio_files_dict[spk_id]
                random.shuffle(spk_id_utts)

                train_split = int(max(np.ceil(0.01 * split_ratio[0] * len(spk_id_utts)), 1))
                # valid_split = int(max(len(spk_id_utts)-train_split, 0))

                for i in range(train_split):
                    train_snts.append(spk_id_utts.pop())
                    train_spk.add(spk_id)

                for utts in spk_id_utts:
                    dev_snts.append(utts)
                    dev_spk.add(spk_id)

            print('Split %d spks\'utterances for training and %d spks\'utterances for dev.' % (len(train_spk), len(dev_spk)))
            # print(len(train_spk))
            # print(len(dev_spk))
            # split = int(0.01 * split_ratio[0] * len(audio_files_list))
            # train_snts = audio_files_list[:split]
            # dev_snts = audio_files_list[split:]

            train_lst.extend(train_snts)
            dev_lst.extend(dev_snts)

        print('Split %d utterances for training and %d utterances for dev.' % (len(train_lst), len(dev_lst)))

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    chunk_lst = set()
    if audio_duration >= seg_dur:
        num_chunks = int(audio_duration / seg_dur)  # all in milliseconds
        for i in range(num_chunks):
            chunk_lst.add(audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur))

        # if audio_duration > seg_dur:
        #     for i in range(num_chunks):
        #         start = np.random.randint(0, int((audio_duration - seg_dur) * SAMPLERATE)) / SAMPLERATE
        #         chunk_lst.add(audio_id + "_" + str(start) + "_" + str(start + seg_dur))

    return list(chunk_lst)


def PrepareCsvProcess(lock_t, t_queue, e_queue, my_sep, random_segment, seg_dur, amp_th, q_queue):

    while True:
        lock_t.acquire()  # 加上锁
        # print(os.getpid(), " acqing lock i")
        if not t_queue.empty():
            wav_file = t_queue.get()
            lock_t.release()
            q_queue.put(1)
        else:
            lock_t.release()
            break

        try:
            [spk_id, utt_id] = wav_file.split("/")[-2:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        # signal, fs = sf.read(wav_file, dtype='float')
        signal, fs = torchaudio.load(wav_file)
        signal = np.array(signal)
        if len(signal.shape) == 2:
            signal = signal.mean(axis=0)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            # Composition of the csv_line
            csv_line = [audio_id, str(audio_duration), wav_file, start_sample, stop_sample, spk_id]
            e_queue.put(csv_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE
            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)

            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = np.abs(signal[start_sample:end_sample]).mean()
                if mean_sig < amp_th:
                    continue
                # print("9: mean")
                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                e_queue.put(csv_line)
        # print('\rProcess [{:8>s}]: [{:>8d}] wav Left'.format
        #       (str(os.getpid()), t_queue.qsize()), end='')

def listener(q, total_num=10000):
    pbar = tqdm(total=total_num, ncols=60)
    for item in iter(q.get, None):
     pbar.update()

def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)
    # print(os.getpid(), " main process")

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []

    manager = Manager()
    lock_t = manager.Lock()

    t_queue = manager.Queue()
    e_queue = manager.Queue()
    q_queue = manager.Queue()

    # Processing all the wav files in the list
    # for wav_file in tqdm(wav_lst, dynamic_ncols=True):
    #     # Getting sentence and speaker ids
    #     try:
    #         [spk_id, sess_id, utt_id] = wav_file.split("/")[-3:]
    #     except ValueError:
    #         logger.info(f"Malformed path: {wav_file}")
    #         continue
    #     audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])
    #
    #     # Reading the signal (to retrieve duration in seconds)
    #     signal, fs = torchaudio.load(wav_file)
    #     signal = signal.squeeze(0)
    #     audio_duration = signal.shape[0] / SAMPLERATE
    #
    #     if random_segment:
    #         start_sample = 0
    #         stop_sample = signal.shape[0]
    #
    #         # Composition of the csv_line
    #         csv_line = [
    #             audio_id,
    #             str(audio_duration),
    #             wav_file,
    #             start_sample,
    #             stop_sample,
    #             spk_id,
    #         ]
    #         entry.append(csv_line)
    #     else:
    #         uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
    #         for chunk in uniq_chunks_list:
    #             s, e = chunk.split("_")[-2:]
    #             start_sample = int(float(s) * SAMPLERATE)
    #             end_sample = int(float(e) * SAMPLERATE)
    #
    #             #  Avoid chunks with very small energy
    #             mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
    #             if mean_sig < amp_th:
    #                 continue
    #
    #             # Composition of the csv_line
    #             csv_line = [
    #                 chunk,
    #                 str(audio_duration),
    #                 wav_file,
    #                 start_sample,
    #                 end_sample,
    #                 spk_id,
    #             ]
    #             entry.append(csv_line)

    for wav in tqdm(wav_lst, ncols=60):
        t_queue.put(wav)

    length_pbar = len(wav_lst)

    # PrepareCsvProcess(lock_t, t_queue, e_queue, my_sep, random_segment, seg_dur, amp_th)
    nj = 16
    proc = Process(target=listener, args=(q_queue, length_pbar))
    proc.start()
    pool = Pool(processes=nj)
    for i in range(0, nj):
        pool.apply_async(PrepareCsvProcess, args=(lock_t, t_queue, e_queue, my_sep, random_segment, seg_dur, amp_th, q_queue))

    pool.close()  # 关闭进程池，表示不能在往进程池中添加进程
    pool.join()  # 等待进程池中的所有进程执行完毕，必须在close

    q_queue.put(None)
    proc.join()

    while not e_queue.empty():
        entry.append(e_queue.get())

    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s successfully created!" % (csv_file)
    logger.info(msg)


def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)

    Arguments
    ---------
    data_folder : str
        Path of the data folders
    save_folder : str
        The directory where to store the csv files.

    Returns
    -------
    None
    """

    # msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    # logger.debug(msg)

    csv_output_head = [
        ["ID", "duration", "wav", "start", "stop", "spk_id"]
    ]  # noqa E231

    for data_folder in data_folders:

        test_lst_file = verification_pairs_file

        enrol_ids, test_ids = [], []

        # Get unique ids (enrol and test utterances)
        for line in open(test_lst_file): # id00800-enroll test/id00800-singing-01-005.wav 1
            e_id = line.split(" ")[0] #.rstrip().split(".")[0].strip()
            t_id = line.split(" ")[1].split("/")[1].split(".")[0].strip()

            enrol_ids.append(e_id)
            test_ids.append(t_id)

        enrol_ids = list(np.unique(np.array(enrol_ids)))
        test_ids = list(np.unique(np.array(test_ids)))

        # Prepare enrol csv
        logger.info("preparing enrol csv")
        enrol_csv = []
        for id in enrol_ids: # id00800-enroll
            if os.path.exists(data_folder + "/eval/enroll/" + id + ".flac"):
                wav = data_folder + "/eval/enroll/" + id + ".flac"
            else:
                assert os.path.exists(data_folder + "/eval/enroll/" + id + ".wav")
                wav = data_folder + "/eval/enroll/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]

            spk_id = id.split("-")[0]
            csv_line = [id, audio_duration, wav, start_sample, stop_sample, spk_id,]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file = os.path.join(save_folder, ENROL_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        # Prepare test csv
        logger.info("preparing test csv")
        test_csv = []
        for id in test_ids: # id00800-singing-01-005
            if os.path.exists(data_folder + "/eval/test/" + id + ".flac"):
                wav = data_folder + "/eval/test/" + id + ".flac"
            else:
                assert os.path.exists(data_folder + "/eval/test/" + id + ".wav")
                wav = data_folder + "/eval/test/" + id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            signal, fs = torchaudio.load(wav)
            signal = signal.squeeze(0)
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            stop_sample = signal.shape[0]
            spk_id = id.split("-")[0]

            csv_line = [
                id,
                audio_duration,
                wav,
                start_sample,
                stop_sample,
                spk_id,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file = os.path.join(save_folder, TEST_CSV)

        # Writing the csv lines
        with open(csv_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)
