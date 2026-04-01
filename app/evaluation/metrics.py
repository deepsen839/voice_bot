# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from jiwer import wer

# def evaluate_intent(y_true, y_pred):
#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, average="weighted"),
#         "recall": recall_score(y_true, y_pred, average="weighted"),
#         "f1": f1_score(y_true, y_pred, average="weighted"),
#         "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
#     }

# def evaluate_wer(true, pred):
#     return wer(true, pred)
import evaluate

wer_metric = evaluate.load("wer")

def compute_wer(references, predictions):
    # Ensure list format
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions, str):
        predictions = [predictions]

    return wer_metric.compute(
        references=references,
        predictions=predictions
    )