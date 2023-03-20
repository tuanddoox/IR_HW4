import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import LTRData


def evaluate_per_query(all_scores, all_labels, print_results=False, q_level=False):
    results = {}
    for q_scores, q_labels in zip(all_scores, all_labels):
        add_to_results(results, evaluate_labels_scores(q_labels, q_scores))

    if print_results:
        print('"metric": "mean" ("standard deviation")')
    mean_results = {}
    for k in sorted(results.keys()):
        v = results[k]
        mean_v = np.mean(v)
        std_v = np.std(v)
        mean_results[k] = (mean_v, std_v)
        if print_results:
            print("%s: %0.04f (%0.05f)" % (k, mean_v, std_v))

    if q_level:
        return mean_results, results

    return mean_results


def evaluate(data_split, all_scores, all_labels, print_results=False, q_level=False):
    results = {}
    for qid in np.arange(data_split.num_queries()):
        if included(qid, data_split):
            add_to_results(results, evaluate_query(data_split, qid, all_scores))

    if print_results:
        print('"metric": "mean" ("standard deviation")')
    mean_results = {}
    for k in sorted(results.keys()):
        v = results[k]
        mean_v = np.mean(v)
        std_v = np.std(v)
        mean_results[k] = (mean_v, std_v)
        if print_results:
            print("%s: %0.04f (%0.05f)" % (k, mean_v, std_v))

    if q_level:
        return mean_results, results

    return mean_results


# this function evaluates a model, on a given split
def evaluate_model(
    data, pred_fn, split, batch_size=256, print_results=False, q_level=False
):
    dataset = LTRData(data, split)
    all_scores = []
    all_labels = dataset.labels
    for idx in tqdm(
        range(0, len(dataset), batch_size), desc=f"Eval ({split})", leave=False
    ):
        batch_features = dataset.features[idx : idx + batch_size, :]
        with torch.no_grad():
            output = pred_fn(batch_features)
            all_scores.extend(output.squeeze().numpy().tolist())

    split = {"train": data.train, "validation": data.validation, "test": data.test}.get(
        split
    )
    grouped_scores = []
    grouped_labels = []
    for idx in range(dataset.num_queries):
        grouped_scores.append(
            np.asarray(
                all_scores[
                    dataset.doclist_ranges[idx] : dataset.doclist_ranges[idx + 1]
                ]
            )
        )
        grouped_labels.append(
            np.asarray(
                all_labels[
                    dataset.doclist_ranges[idx] : dataset.doclist_ranges[idx + 1]
                ]
            )
        )
    results = evaluate_per_query(
        grouped_scores, grouped_labels, print_results=print_results, q_level=q_level,
    )
    return results


################# BELOW FUNCTIONS ALL NEED TO BE CHECKED


def dcg_at_k(sorted_labels, k):
    if k > 0:
        k = min(sorted_labels.shape[0], k)
    else:
        k = sorted_labels.shape[0]
    denom = 1.0 / np.log2(np.arange(k) + 2.0)
    nom = 2 ** sorted_labels - 1.0
    dcg = np.sum(nom[:k] * denom)
    return dcg


def ndcg10(scores, labels):
    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind]
    ideal_labels = np.sort(labels)[::-1]
    return dcg_at_k(sorted_labels, 10) / dcg_at_k(ideal_labels, 10)


def ndcg_at_k(sorted_labels, ideal_labels, k):
    return dcg_at_k(sorted_labels, k) / dcg_at_k(ideal_labels, k)


def evaluate_query(data_split, qid, all_scores, q_labels=None):
    s_i, e_i = data_split.doclist_ranges[qid : qid + 2]
    q_scores = all_scores[s_i:e_i]
    q_labels = data_split.query_labels(qid)
    return evaluate_labels_scores(q_labels, q_scores)


def evaluate_labels_scores(labels, scores):
    n_docs = labels.shape[0]

    random_i = np.random.permutation(np.arange(scores.shape[0]))
    labels = labels[random_i]
    scores = scores[random_i]

    sort_ind = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_ind]
    ideal_labels = np.sort(labels)[::-1]

    bin_labels = np.greater(sorted_labels, 2)
    bin_ideal_labels = np.greater(ideal_labels, 2)

    rel_i = np.arange(1, len(sorted_labels) + 1)[bin_labels]

    total_labels = float(np.sum(bin_labels))
    assert total_labels > 0 or np.any(np.greater(labels, 0))
    if total_labels > 0:
        result = {
            "relevant rank": list(rel_i),
            "relevant rank per query": np.sum(rel_i),
            "precision@01": np.sum(bin_labels[:1]) / 1.0,
            "precision@03": np.sum(bin_labels[:3]) / 3.0,
            "precision@05": np.sum(bin_labels[:5]) / 5.0,
            "precision@10": np.sum(bin_labels[:10]) / 10.0,
            "precision@20": np.sum(bin_labels[:20]) / 20.0,
            "recall@01": np.sum(bin_labels[:1]) / total_labels,
            "recall@03": np.sum(bin_labels[:3]) / total_labels,
            "recall@05": np.sum(bin_labels[:5]) / total_labels,
            "recall@10": np.sum(bin_labels[:10]) / total_labels,
            "recall@20": np.sum(bin_labels[:20]) / total_labels,
            "dcg": dcg_at_k(sorted_labels, 0),
            "dcg@03": dcg_at_k(sorted_labels, 3),
            "dcg@05": dcg_at_k(sorted_labels, 5),
            "dcg@10": dcg_at_k(sorted_labels, 10),
            "dcg@20": dcg_at_k(sorted_labels, 20),
            "ndcg": ndcg_at_k(sorted_labels, ideal_labels, 0),
            "ndcg@03": ndcg_at_k(sorted_labels, ideal_labels, 3),
            "ndcg@05": ndcg_at_k(sorted_labels, ideal_labels, 5),
            "ndcg@10": ndcg_at_k(sorted_labels, ideal_labels, 10),
            "ndcg@20": ndcg_at_k(sorted_labels, ideal_labels, 20),
        }
    else:
        result = {
            "dcg": dcg_at_k(sorted_labels, 0),
            "dcg@03": dcg_at_k(sorted_labels, 3),
            "dcg@05": dcg_at_k(sorted_labels, 5),
            "dcg@10": dcg_at_k(sorted_labels, 10),
            "dcg@20": dcg_at_k(sorted_labels, 20),
            "ndcg": ndcg_at_k(sorted_labels, ideal_labels, 0),
            "ndcg@03": ndcg_at_k(sorted_labels, ideal_labels, 3),
            "ndcg@05": ndcg_at_k(sorted_labels, ideal_labels, 5),
            "ndcg@10": ndcg_at_k(sorted_labels, ideal_labels, 10),
            "ndcg@20": ndcg_at_k(sorted_labels, ideal_labels, 20),
        }
    return result


def included(qid, data_split):
    return np.any(np.greater(data_split.query_labels(qid), 0))


def add_to_results(results, cur_results):
    for k, v in cur_results.items():
        if not (k in results):
            results[k] = []
        if type(v) == list:
            results[k].extend(v)
        else:
            results[k].append(v)

