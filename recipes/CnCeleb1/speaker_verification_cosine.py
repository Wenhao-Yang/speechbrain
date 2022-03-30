#!/usr/bin/python3
"""Recipe for training a speaker verification system based on cosine distance.
The cosine distance is computed on the top of pre-trained embeddings.
The pre-trained model is automatically downloaded from the web if not specified.
This recipe is designed to work on a single GPU.

To run this recipe, run the following command:
    >  python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml

Authors
    * Hwidong Na 2020
    * Mirco Ravanelli 2020
"""
import os
import pdb
import random
import sys
import torch
import logging
import torchaudio
import speechbrain as sb
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import EER, minDCF
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
import pickle
import numpy as np


# Compute embeddings from the waveforms
def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings.

    Arguments
    ---------
    wavs : Torch.Tensor
        Tensor containing the speech waveform (batch, time).
        Make sure the sample rate is fs=16000 Hz.
    wav_lens: Torch.Tensor
        Tensor containing the relative length for each sentence
        in the length (e.g., [0.8 0.6 1.0])
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)
        embeddings = params["embedding_model"](feats, wav_lens)
        embeddings = params["mean_var_norm_emb"](
            embeddings, torch.ones(embeddings.shape[0]).to(embeddings.device)
        )
    return embeddings.squeeze(1)


def compute_embedding_loop(data_loader):
    """Computes the embeddings of all the waveforms specified in the
    dataloader.
    """
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, ncols=100):
            batch = batch.to(params["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            found = False
            for seg_id in seg_ids:
                if seg_id not in embedding_dict:
                    found = True
            if not found:
                continue
            wavs, lens = wavs.to(params["device"]), lens.to(params["device"])
            emb = compute_embedding(wavs, lens).unsqueeze(1)
            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()

    return embedding_dict


def get_verification_scores(veri_test, fast=False):
    """ Computes positive and negative scores given the verification split.
    """
    scores = []
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    if os.path.isfile(save_file):
        with open(save_file, "r") as f:
            for l in f.readlines():
                enrol_id, test_id, lab_pair, score = l.split()
                lab_pair = int(lab_pair)
                score = float(score)

                if lab_pair == 1:
                    positive_scores.append(score)
                else:
                    negative_scores.append(score)

                scores.append(score)

        if len(scores) == len(veri_test):
            logger.info("Loading scores from %s ..." % os.path.join(params["output_folder"], "scores.txt"))
            return positive_scores, negative_scores
        else:
            positive_scores = []
            negative_scores = []
            scores = []

    s_file = open(save_file, "w")

    # Cosine similarity initialization
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    # creating cohort for score normalization
    # pdb.set_trace()
    if "score_norm" in params:
        train_cohort = list(train_dict.values())
        if fast:
            random.shuffle(train_cohort)
            train_cohort = train_cohort[:50000]

        train_cohort = torch.stack(train_cohort)

    enroll_cohort = {}
    test_cohort = {}

    for i, line in enumerate(veri_test):
        enrol_id = line.split(" ")[0].rstrip().split(".")[0].strip()
        if enrol_id not in enroll_cohort:
            enrol = enrol_dict[enrol_id]

            enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            score_e_c = similarity(enrol_rep, train_cohort)

            if "cohort_size" in params:
                score_e_c = torch.topk(
                    score_e_c, k=params["cohort_size"], dim=0
                )[0]

            mean_e_c = torch.mean(score_e_c, dim=0)
            std_e_c = torch.std(score_e_c, dim=0)

            enroll_cohort[enrol_id] = {
                'mean_e_c': mean_e_c,
                'std_e_c': std_e_c,
            }

        test_id = line.split(" ")[1].rstrip().split(".")[0].strip().split("/")[1]
        if test_id not in test_cohort:
            test = test_dict[test_id]

            test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            score_t_c = similarity(test_rep, train_cohort)

            if "cohort_size" in params:
                score_t_c = torch.topk(
                    score_t_c, k=params["cohort_size"], dim=0
                )[0]

            mean_t_c = torch.mean(score_t_c, dim=0)
            std_t_c = torch.std(score_t_c, dim=0)

            test_cohort[test_id] = {
                'mean_t_c': mean_t_c,
                'std_t_c': std_t_c,
            }

    for i, line in tqdm(enumerate(veri_test), ncols=100):

        # Reading verification file (enrol_file test_file label)
        lab_pair = int(line.split(" ")[2].rstrip().split(".")[0].strip())
        enrol_id = line.split(" ")[0].rstrip().split(".")[0].strip()
        test_id = line.split(" ")[1].rstrip().split(".")[0].strip().split("/")[1]

        enrol = enrol_dict[enrol_id]
        test = test_dict[test_id]

        if "score_norm" in params:
            # Getting norm stats for enrol impostors
            # enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
            # score_e_c = similarity(enrol_rep, train_cohort)
            #
            # if "cohort_size" in params:
            #     score_e_c = torch.topk(
            #         score_e_c, k=params["cohort_size"], dim=0
            #     )[0]

            mean_e_c = enroll_cohort[enrol_id]['mean_e_c']
            std_e_c = enroll_cohort[enrol_id]['std_e_c']

            # Getting norm stats for test impostors
            # test_rep = test.repeat(train_cohort.shape[0], 1, 1)
            # score_t_c = similarity(test_rep, train_cohort)
            #
            # if "cohort_size" in params:
            #     score_t_c = torch.topk(
            #         score_t_c, k=params["cohort_size"], dim=0
            #     )[0]

            mean_t_c = test_cohort[test_id]['mean_t_c']
            std_t_c = test_cohort[test_id]['std_t_c']

        # Compute the score for the given sentence
        score = similarity(enrol, test)[0]

        # Perform score normalization
        if "score_norm" in params:
            if params["score_norm"] == "z-norm":
                score = (score - mean_e_c) / std_e_c
            elif params["score_norm"] == "t-norm":
                score = (score - mean_t_c) / std_t_c
            elif params["score_norm"] == "s-norm":
                score_e = (score - mean_e_c) / std_e_c
                score_t = (score - mean_t_c) / std_t_c
                score = 0.5 * (score_e + score_t)

        # write score file
        s_file.write("%s %s %i %f\n" % (enrol_id, test_id, lab_pair, score))
        scores.append(score)

        if lab_pair == 1:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    s_file.close()
    return positive_scores, negative_scores


def dataio_prep(params):
    "Creates the dataloaders and their data processing pipelines."

    data_folder = params["data_folder"]

    # 1. Declarations:

    # Train data (used for normalization)
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"], replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration", select_n=params["n_train_snts"]
    )

    # Enrol data
    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"], replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    # Test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 4 Create dataloaders
    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader


def compute_eer(positive_scores, negative_scores, fast=False):
    """Computes the EER (and its threshold).

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    if fast:
        thresholds_steps = torch.arange(thresholds.min(), thresholds.max(),
                                        (thresholds.max() - thresholds.min()) / 100000)
        if len(thresholds_steps) < len(thresholds):
            thresholds = thresholds_steps

    # Computing False Rejection Rate (miss detection)
    FRR = []
    positive_scores = torch.sort(positive_scores).values
    FAR = []
    negative_scores = torch.sort(negative_scores).values

    # pdb.set_trace()
    neg_idx = 0
    pos_idx = 0
    for t in tqdm(thresholds, ncols=100):

        if t <= negative_scores[0]:
            FAR.append(1)
        else:
            for i in range(neg_idx, len(negative_scores)):
                s = negative_scores[len(negative_scores) - 1 - i]
                if s < t:
                    FAR.append((i + 1) / len(negative_scores))
                    neg_idx = i
                    break

        if t >= positive_scores[-1]:
            FRR.append(1)
        else:
            for i in range(pos_idx, len(positive_scores)):
                s = positive_scores[i]
                if s > t:
                    FRR.append((i + 1) / len(positive_scores))
                    pos_idx = i
                    break

    # positive_scores = torch.cat(
    #     len(thresholds) * [positive_scores.unsqueeze(0)]
    # )
    # pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    # FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    # del positive_scores
    # del pos_scores_threshold
    #
    # # Computing False Acceptance Rate (false alarm)
    # negative_scores = torch.cat(
    #     len(thresholds) * [negative_scores.unsqueeze(0)]
    # )
    # neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    # FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    # del negative_scores
    # del neg_scores_threshold

    # Finding the threshold for EER
    FAR = torch.tensor(FAR)
    FRR = torch.tensor(FRR)
    pdb.set_trace()

    min_index = (FAR - FRR).abs().argmin()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (FAR[min_index] + FRR[min_index]) / 2

    return float(EER), float(thresholds[min_index])


def evaluate_kaldi_eer(target, non_target, cos=True, re_thre=False):
    """
    The distance score should be larger when two samples are more similar.
    :param distances:
    :param labels:
    :param cos:
    :return:
    """
    # split the target and non-target distance array
    # target = []
    # non_target = []
    # new_distances = []

    # new_distances = np.array(new_distances).astype(np.float)

    target = np.sort(target).astype(np.float)
    non_target = np.sort(non_target).astype(np.float)

    target_size = target.size
    nontarget_size = non_target.size
    # pdb.set_trace()
    target_position = 0
    steps = max(1, int(target_size / 1e4))
    while target_position + steps < target_size:
        # for target_position in range(target_size):
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)

        if (nontarget_position < 0):
            nontarget_position = 0
        # The exceptions from non targets are samples where cosine score is > the target score
        # if (non_target[nontarget_position] <= target[target_position]):
        #     break
        if (non_target[nontarget_position] < target[target_position]):
            # print('target[{}]={} is < non_target[{}]={}.'.format(target_position, target[target_position], nontarget_position, non_target[nontarget_position]))
            break
        target_position += steps

    eer_threshold = target[target_position]
    eer = target_position * 1.0 / target_size

    return eer, eer_threshold


# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted([(index, threshold) for index, threshold in enumerate(scores)],
                                             key=itemgetter(1)))
    sorted_labels = []
    labels = [int(labels[i]) for i in sorted_indexes]
    fnrs = []  # 小于阈值的正例数目
    fprs = []  # 小于阈值的反例数目

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])

    fnrs_norm = sum(labels)  # 样本中的正例数目
    fprs_norm = len(labels) - fnrs_norm  # 样本中的反例数目

    # Now divide by the total number of false negative errors to obtain the false positive rates across all thresholds.
    # 小于阈值而被认为是反例的正例在所有正例的样本比重
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the true positive rate.
    # Subtract these quantities from 1 to get the false positive rates.
    # 大于阈值而被认为是正例的反例在所有反例中的样本比重

    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    :param fnrs: 正例的错误拒绝率
    :param fprs: 反例的错误接受率
    :param thresholds: 判断的阈值
    :param p_target: a priori probability of the specified target speaker
    :param c_miss: cost of a missed detection 遗漏正例的损失值
    :param c_fa: cost of a spurious detection 错误接受的损失值
    :return:
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            # 找到最小的det值
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def evaluate_kaldi_mindcf(positive_scores, negative_scores, return_threshold=False):
    c_miss = 1
    c_fa = 1
    labels = [int(x) for x in labels]

    labels = [1 for i in positive_scores]
    labels += [0 for i in negative_scores]

    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)

    p_target = 0.01
    mindcf_01, threshold_01 = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    p_target = 0.001
    mindcf_001, threshold_001 = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    if return_threshold:
        return (mindcf_01, threshold_01, mindcf_001, threshold_001)

    return mindcf_01, mindcf_001


if __name__ == "__main__":

    # Logger setup
    logger = logging.getLogger(__name__)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(current_dir))

    # Load hyperparameters file with command-line overrides
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(
        params["save_folder"], os.path.basename(params["verification_file"])
    )
    if not os.path.exists(veri_file_path):
        download_file(params["verification_file"], veri_file_path)

    from cnceleb_prepare import prepare_cnceleb  # noqa E402

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Prepare data from dev of Voxceleb1
    prepare_cnceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev", "test"],
        split_ratio=params["split_ratio"],
        seg_dur=params["sentence_len"],
        source=params["voxceleb_source"]
        if "voxceleb_source" in params
        else None,
        skip_prep=params["skip_prep"],
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(params["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(params["device"])

    # Computing  enrollment and test embeddings
    logger.info("Computing enroll/test embeddings...")

    enroll_dict_pickle = os.path.join(params["save_folder"], 'xvectors', 'enroll.pickle')
    test_dict_pickle = os.path.join(params["save_folder"], 'xvectors', 'test.pickle')
    train_dict_pickle = os.path.join(params["save_folder"], 'xvectors', 'train.pickle')
    if not os.path.exists(os.path.join(params["save_folder"], 'xvectors')):
        os.makedirs(os.path.join(params["save_folder"], 'xvectors'))

    if os.path.exists(enroll_dict_pickle) and os.path.exists(test_dict_pickle):
        with open(enroll_dict_pickle, 'rb') as f:
            enrol_dict = pickle.load(f)
        with open(test_dict_pickle, 'rb') as f:
            test_dict = pickle.load(f)
    else:
        # First run
        enrol_dict = compute_embedding_loop(enrol_dataloader)
        test_dict = compute_embedding_loop(test_dataloader)

        # Second run (normalization stats are more stable)
        enrol_dict = compute_embedding_loop(enrol_dataloader)
        test_dict = compute_embedding_loop(test_dataloader)

        with open(enroll_dict_pickle, 'wb') as f:
            pickle.dump(enrol_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(test_dict_pickle, 'wb') as f:
            pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "score_norm" in params:
        if os.path.exists(train_dict_pickle):
            with open(train_dict_pickle, 'rb') as f:
                train_dict = pickle.load(f)
        else:
            train_dict = compute_embedding_loop(train_dataloader)
            with open(train_dict_pickle, 'wb') as f:
                pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute the EER
    logger.info("Computing Scores ..")
    # Reading standard verification split
    with open(veri_file_path) as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)  # , fast=params['fast_score'])
    del enrol_dict, test_dict

    logger.info("Computing EER..")
    # eer, th = evaluate_kaldi_eer(torch.tensor(positive_scores), torch.tensor(negative_scores), fast=params['fast_score'])
    eer, th = evaluate_kaldi_eer(torch.tensor(positive_scores),
                                 torch.tensor(negative_scores))  # , fast=params['fast_score'])

    logger.info("EER(%%)=%f", eer * 100)

    # min_dcf, th = minDCF(
    #     torch.tensor(positive_scores), torch.tensor(negative_scores)
    # )
    mindcf_01, mindcf_001 = evaluate_kaldi_mindcf(positive_scores, negative_scores)

    logger.info("minDCF(0.01)=%f", mindcf_01)
    logger.info("minDCF(0.01)=%f", mindcf_001)
