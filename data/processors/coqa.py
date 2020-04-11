import collections
import json
import logging
import math
import os
import re
import string
from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count

import spacy
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from .utils import DataProcessor
from transformers import BasicTokenizer

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def coqa_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def coqa_convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []

    query_tokens = tokenizer.tokenize(example.question_text)

    cls_idx = 3
    if example.orig_answer_text == 'yes':
        cls_idx = 0  # yes
    elif example.orig_answer_text == 'no':
        cls_idx = 1  # no
    elif example.orig_answer_text == 'unknown':
        cls_idx = 2  # unknown

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    tok_r_start_position, tok_r_end_position = None, None
    tok_pre_sp, tok_pre_ep = [], []

    # rational part
    tok_r_start_position = orig_to_tok_index[
        example.rational_start_position]
    if example.rational_end_position < len(example.doc_tokens) - 1:
        tok_r_end_position = orig_to_tok_index[
                                 example.rational_end_position + 1] - 1
    else:
        tok_r_end_position = len(all_doc_tokens) - 1
    # rational part end

    if cls_idx < 3:
        tok_start_position, tok_end_position = 0, 0
    else:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position +
                                                 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        slice_cls_idx = cls_idx
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans,
                                                   doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        rational_start_position = None
        ration_end_position = None

        # rational_part
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if example.rational_start_position == -1 or not (
                tok_r_start_position >= doc_start
                and tok_r_end_position <= doc_end):
            out_of_span = True
        if out_of_span:
            rational_start_position = 0
            rational_end_position = 0
        else:
            doc_offset = len(query_tokens) + 2
            rational_start_position = tok_r_start_position - doc_start + doc_offset
            rational_end_position = tok_r_end_position - doc_start + doc_offset
        # rational_part_end

        rational_mask = [0] * len(input_ids)
        if not out_of_span:
            rational_mask[rational_start_position:rational_end_position +
                                                  1] = [1] * (rational_end_position -
                                                              rational_start_position + 1)

        if cls_idx >= 3:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start
                    and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                slice_cls_idx = 2
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
        else:
            start_position = 0
            end_position = 0

        features.append(
            CoqaFeatures(example_index=0,
                         # Can not set unique_id and example_index here. They will be set after multiple processing.
                         unique_id=0,
                         doc_span_index=doc_span_index,
                         tokens=tokens,
                         token_to_orig_map=token_to_orig_map,
                         token_is_max_context=token_is_max_context,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         start_position=start_position,
                         end_position=end_position,
                         cls_idx=slice_cls_idx,
                         rational_mask=rational_mask))

    return features


def coqa_convert_examples_to_features(examples, max_seq_length, doc_stride, max_query_length, is_training, threads=1):

    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=coqa_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            coqa_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert coqa examples to features",
            )
        )

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if not is_training:
        all_example_index = torch.arange(all_input_ids.size(0),
                                         dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask,
                                all_segment_ids, all_example_index)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        all_rational_mask = torch.tensor([f.rational_mask for f in features], dtype=torch.long)
        all_cls_idx = torch.tensor([f.cls_idx for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_start_positions,
                                   all_end_positions, all_rational_mask,
                                   all_cls_idx)

    return features, dataset

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index",
            "start_index",
            "end_index",
            "score",
            "cls_idx",
        ])

    # all_predictions = collections.OrderedDict()
    all_predictions = []
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index,
         example) in enumerate(tqdm(all_examples, desc="Writing preditions")):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        part_prelim_predictions = []

        score_yes, score_no, score_span, score_unk = -float('INF'), -float(
            'INF'), -float('INF'), float('INF')
        min_unk_feature_index, max_yes_feature_index, max_no_feature_index, max_span_feature_index = -1, -1, -1, -1  # the paragraph slice with min null score
        max_span_start_indexes, max_span_end_indexes = [], []
        max_start_index, max_end_index = -1, -1
        # null_start_logit = 0  # the start logit at the slice with min null score
        # null_end_logit = 0  # the end logit at the slice with min null score

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            # if we could have irrelevant answers, get the min score of irrelevant
            # feature_null_score = result.start_logits[0] + result.end_logits[0]

            # feature_yes_score, feature_no_score, feature_unk_score, feature_span_score = result.cls_logits

            feature_yes_score, feature_no_score, feature_unk_score = result.yes_logits[
                0] * 2, result.no_logits[0] * 2, result.unk_logits[0] * 2
            start_indexes, end_indexes = _get_best_indexes(
                result.start_logits,
                n_best_size), _get_best_indexes(result.end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    feature_span_score = result.start_logits[
                        start_index] + result.end_logits[end_index]
                    prelim_predictions.append(
                        _PrelimPrediction(feature_index=feature_index,
                                          start_index=start_index,
                                          end_index=end_index,
                                          score=feature_span_score,
                                          cls_idx=3))

            if feature_unk_score < score_unk:  # find min score_noanswer
                score_unk = feature_unk_score
                min_unk_feature_index = feature_index
            if feature_yes_score > score_yes:  # find max score_yes
                score_yes = feature_yes_score
                max_yes_feature_index = feature_index
            if feature_no_score > score_no:  # find max score_no
                score_no = feature_no_score
                max_no_feature_index = feature_index

        prelim_predictions.append(
            _PrelimPrediction(feature_index=min_unk_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_unk,
                              cls_idx=2))
        prelim_predictions.append(
            _PrelimPrediction(feature_index=max_yes_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_yes,
                              cls_idx=0))
        prelim_predictions.append(
            _PrelimPrediction(feature_index=max_no_feature_index,
                              start_index=0,
                              end_index=0,
                              score=score_no,
                              cls_idx=1))

        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda p: p.score,
                                    reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "score", "cls_idx"])

        seen_predictions = {}
        nbest = []
        cls_rank = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:  # including yes/no/noanswer pred
                break
            feature = features[pred.feature_index]
            if pred.cls_idx == 3:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index +
                                                              1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case,
                                            verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
                nbest.append(
                    _NbestPrediction(text=final_text,
                                     score=pred.score,
                                     cls_idx=pred.cls_idx))
            else:
                text = ['yes', 'no', 'unknown']
                nbest.append(
                    _NbestPrediction(text=text[pred.cls_idx],
                                     score=pred.score,
                                     cls_idx=pred.cls_idx))

        if len(nbest) < 1:
            nbest.append(
                _NbestPrediction(text='unknown',
                                 score=-float('inf'),
                                 cls_idx=2))

        assert len(nbest) >= 1

        probs = _compute_softmax([p.score for p in nbest])

        nbest_json = []

        for i, entry in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["socre"] = entry.score
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        _id, _turn_id = example.qas_id.split()
        all_predictions.append({
            'id': _id,
            'turn_id': int(_turn_id),
            'answer': confirm_preds(nbest_json)
        })

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

def confirm_preds(nbest_json):
    # Do something for some obvious wrong-predictions
    subs = [
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'eleven', 'twelve', 'true', 'false'
    ]  # very hard-coding, can be extended.
    ori = nbest_json[0]['text']
    if len(ori) < 2:  # mean span like '.', '!'
        for e in nbest_json[1:]:
            if _normalize_answer(e['text']) in subs:
                return e['text']
        return 'unknown'
    return ori

# TODO：move to metrics
def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" %
                        (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits),
                             key=lambda x: x[1],
                             reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def score(pred, truth):
    def _f1_score(pred, answers):
        def _score(g_tokens, a_tokens):
            common = Counter(g_tokens) & Counter(a_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1. * num_same / len(g_tokens)
            recall = 1. * num_same / len(a_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        if pred is None or answers is None:
            return 0

        if len(answers) == 0:
            return 1. if len(pred) == 0 else 0.

        g_tokens = _normalize_answer(pred).split()
        ans_tokens = [_normalize_answer(answer).split() for answer in answers]
        scores = [_score(g_tokens, a) for a in ans_tokens]
        if len(ans_tokens) == 1:
            score = scores[0]
        else:
            score = 0
            for i in range(len(ans_tokens)):
                scores_one_out = scores[:i] + scores[(i + 1):]
                score += max(scores_one_out)
            score /= len(ans_tokens)
        return score

    #Main Stream
    assert len(pred) == len(truth)
    pred, truth = pred.items(), truth.items()
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for (p_id, p), (t_id, t), in zip(pred, truth):
        assert p_id == t_id
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'unknown':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s


class CoqaFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 cls_idx=None,
                 rational_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.cls_idx = cls_idx
        self.rational_mask = rational_mask


class CoqaExample(object):
    """
    A single training/test example for the CoQA dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
            self,
            qas_id,
            question_text,
            doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            rational_start_position=None,
            rational_end_position=None,
            additional_answers=None,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.additional_answers = additional_answers
        self.rational_start_position = rational_start_position
        self.rational_end_position = rational_end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class CoqaProcessor(DataProcessor):
    train_file = "coqa-train-v1.0.json"
    dev_file = "coqa-dev-v1.0.json"

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

    def space_extend(self, matchobj):
        return ' ' + matchobj.group(0) + ' '

    def pre_proc(self, text):
        text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', self.space_extend, text)
        text = text.strip(' \n')
        text = re.sub('\s+', ' ', text)
        return text

    def process(self, parsed_text):
        output = {'word': [], 'offsets': [], 'sentences': []}

        for token in parsed_text:
            output['word'].append(self._str(token.text))
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output

    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:',
                      raw_text)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)

        return raw_context_offsets

    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)

    def normalize_answer(self, s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(self.pre_proc(ground_truth)).split()

        ls = [
            i for i in range(len(offsets))
            if context[offsets[i][0]:offsets[i][1]].lower() in gt
        ]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = self.normalize_answer(
                    self.pre_proc(
                        context[offsets[ls[i]][0]:offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (ls[i], ls[j])
        return best_span

    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)

    def get_examples(self, data_dir, history_len, add_qa_tag, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `coqa-train-v1.0.json`.

        """
        if data_dir is None:
            data_dir = ""

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, history_len, add_qa_tag)

    def _create_examples(self, input_data, history_len, add_qa_tag):
        nlp = spacy.load('en_core_web_sm', parser=False)
        examples = []
        for data_idx in tqdm(range(len(input_data)), desc='Generating examples'):
            datum = input_data[data_idx]
            context_str = datum['story']
            _datum = {
                'context': context_str,
                'source': datum['source'],
                'id': datum['id'],
                'filename': datum['filename']
            }
            nlp_context = nlp(self.pre_proc(context_str))
            _datum['annotated_context'] = self.process(nlp_context)
            _datum['raw_context_offsets'] = self.get_raw_context_offsets(
                _datum['annotated_context']['word'], context_str)
            assert len(datum['questions']) == len(datum['answers'])
            additional_answers = {}
            if 'additional_answers' in datum:
                for k, answer in datum['additional_answers'].items():
                    if len(answer) == len(datum['answers']):
                        for ex in answer:
                            idx = ex['turn_id']
                            if idx not in additional_answers:
                                additional_answers[idx] = []
                            additional_answers[idx].append(ex['input_text'])
            for i in range(len(datum['questions'])):
                question, answer = datum['questions'][i], datum['answers'][i]
                assert question['turn_id'] == answer['turn_id']

                idx = question['turn_id']
                _qas = {
                    'turn_id': idx,
                    'question': question['input_text'],
                    'answer': answer['input_text']
                }
                if idx in additional_answers:
                    _qas['additional_answers'] = additional_answers[idx]

                _qas['raw_answer'] = answer['input_text']

                if _qas['raw_answer'].lower() in ['yes', 'yes.']:
                    _qas['raw_answer'] = 'yes'
                if _qas['raw_answer'].lower() in ['no', 'no.']:
                    _qas['raw_answer'] = 'no'
                if _qas['raw_answer'].lower() in ['unknown', 'unknown.']:
                    _qas['raw_answer'] = 'unknown'

                _qas['answer_span_start'] = answer['span_start']
                _qas['answer_span_end'] = answer['span_end']
                start = answer['span_start']
                end = answer['span_end']
                chosen_text = _datum['context'][start:end].lower()
                while len(chosen_text) > 0 and self.is_whitespace(chosen_text[0]):
                    chosen_text = chosen_text[1:]
                    start += 1
                while len(chosen_text) > 0 and self.is_whitespace(chosen_text[-1]):
                    chosen_text = chosen_text[:-1]
                    end -= 1
                r_start, r_end = self.find_span(_datum['raw_context_offsets'], start,
                                                end)
                input_text = _qas['answer'].strip().lower()
                if input_text in chosen_text:
                    p = chosen_text.find(input_text)
                    _qas['answer_span'] = self.find_span(_datum['raw_context_offsets'],
                                                         start + p,
                                                         start + p + len(input_text))
                else:
                    _qas['answer_span'] = self.find_span_with_gt(
                        _datum['context'], _datum['raw_context_offsets'],
                        input_text)
                long_question = ''
                for j in range(i - history_len, i + 1):
                    if j < 0:
                        continue
                    long_question += (' <Q> ' if add_qa_tag else
                                      ' ') + datum['questions'][j]['input_text']
                    if j < i:
                        long_question += (' <A> ' if add_qa_tag else
                                          ' ') + datum['answers'][j]['input_text']

                long_question = long_question.strip()
                _qas['raw_long_question'] = long_question
                _qas['annotated_long_question'] = self.process(
                    nlp(self.pre_proc(long_question)))
                example = CoqaExample(
                    qas_id=_datum['id'] + ' ' + str(_qas['turn_id']),
                    question_text=_qas['raw_long_question'],
                    doc_tokens=_datum['annotated_context']['word'],
                    orig_answer_text=_qas['raw_answer'],
                    start_position=_qas['answer_span'][0],
                    end_position=_qas['answer_span'][1],
                    rational_start_position=r_start,
                    rational_end_position=r_end,
                    additional_answers=_qas['additional_answers'] if 'additional_answers' in _qas else None,
                )
                examples.append(example)

        return examples


class CoqaResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, yes_logits, no_logits, unk_logits):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.yes_logits = yes_logits
        self.no_logits = no_logits
        self.unk_logits = unk_logits
