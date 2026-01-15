import os
import torch
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path):
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'epoch': epoch,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)

def calculate_metric(preds, labels):

    sums = preds.sum(dim=1)
    # Check if all sums are close to 1 (allowing for some numerical tolerance)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
        preds = F.softmax(preds, dim=1)

    binary_preds = preds.argmax(1)
    
    # Confusion matrix components
    TP = (binary_preds * labels).sum().float()   # True Positives
    TN = ((1-binary_preds) * (1-labels)).sum().float()   # True Negatives
    FP = (binary_preds * (1-labels)).sum().float()   # False Positives
    FN = ((1-binary_preds) * labels).sum().float()   # False Negatives

    # True Positive Rate (TPR), or Sensitivity, or Recall
    TPR = TP / (TP + FN)
    # True Negative Rate (TNR), or Specificity
    TNR = TN / (TN + FP)
    # F1 Score
    precision = TP / (TP + FP)
    recall = TPR
    F1 = 2 * (precision * recall) / (precision + recall)
    # Accuracy
    ACC = (TP + TN) / (TP + TN + FP + FN)
    # AUC
    positive_class_probs = preds[:, 1]
    AUC = torch.tensor(roc_auc_score(labels.cpu().detach().numpy(), positive_class_probs.cpu().detach().numpy()))
    return TPR, TNR, F1, AUC, ACC
