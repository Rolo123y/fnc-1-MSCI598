from distutils.errors import CompileError
from traceback import print_tb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from transformers.modeling_utils import load_state_dict
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import device, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tensorflow.python.client import device_lib
import time

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MODEL = 'bert-base-multilingual-uncased'
transformers.logging.set_verbosity_error()

class GPReviewDataset(Dataset):

  def __init__(self, headlines, articles, targets, tokenizer, max_len):
    self.headline = headlines
    self.article = articles
    self.target = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.headline)
  
  def __getitem__(self, item):
    headline = str(self.headline[item])
    article = str(self.article[item])
    target = self.target[item]

    encoding = self.tokenizer(
      headline,
      article,
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation=True,
      return_token_type_ids=False,
      return_attention_mask=True,
      return_tensors='pt',
      verbose=False
    )

    return {
      'headline_text': headline,
      'article_text': article,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.float)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    headlines = df['Headline'].to_numpy(),
    articles = df['article'].to_numpy(),
    targets = df['Stance'].to_numpy(),
    tokenizer = tokenizer,
    max_len = max_len
  )

  return DataLoader(
    dataset=ds,
    batch_size=batch_size,
    num_workers=4
  )

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids.to(device),
      attention_mask=attention_mask.to(device),
      return_dict=False,
    )

    return self.out(pooled_output)

def get_predictions(model, data_loader):
  model = model.eval()
  
  headlines_texts = []
  articles_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      headlines = d["headline_text"],
      articles = d['article_text']
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, preds = torch.max(outputs, dim=1)
      probs = F.softmax(outputs, dim=1)

      headlines_texts.extend(headlines)
      articles_texts.extend(articles)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()

  return headlines_texts, articles_texts, predictions, prediction_probs, real_values

def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment')
  plt.show()

if __name__ == "__main__":
  RANDOM_SEED = 43
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f'RUNNING ON DEVICE: {device}')

  # LOAD THE TOKENIZER AND MODEL
  model = SentimentClassifier(4)
  model.load_state_dict(torch.load('model/acc_best_model_state.bin'))
  model = model.to(device)
  tokenizer = BertTokenizer('model/acc_best_model_tokenizer.bin', do_lower_case=True, local_files_only=True)

# GET THE COMPETITION DATASET
  Competition_dataset = DataSet("competition_test")
  Competition_dataset_stances_frame = pd.DataFrame(Competition_dataset.stances)
  Competition_dataset_articles_frame = pd.DataFrame(list(Competition_dataset.articles.items()),columns = ['Body ID','article']) 
  Competition_dataset_frame = Competition_dataset_stances_frame.merge(Competition_dataset_articles_frame[["Body ID","article"]], on="Body ID", how="left")
  Competition_dataset_frame.loc[Competition_dataset_frame["Stance"] == "unrelated", "Stance"] = 0
  Competition_dataset_frame.loc[Competition_dataset_frame["Stance"] == "agree", "Stance"] = 1
  Competition_dataset_frame.loc[Competition_dataset_frame["Stance"] == "discuss", "Stance"] = 2
  Competition_dataset_frame.loc[Competition_dataset_frame["Stance"] == "disagree", "Stance"] = 3
  # Competition_dataset_frame = Competition_dataset_frame.iloc[:5]

# CREATE A DATALOADER WITH THE COMPETITION DATASET AND THE TOKENIZER
  start_time = time.time()

  test_data_loader = create_data_loader(Competition_dataset_frame, tokenizer, 500, 1)
  y_headline_text, y_article_text, y_pred, y_pred_probs, y_test = get_predictions(
      model,
      test_data_loader
  )
  
  print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))

  class_names = ['class 0', 'class 1', 'class 2', 'class 3']
  labels=[0,1,2,3]
  print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

  dy = pd.DataFrame(y_pred.numpy())
  dx = pd.DataFrame(y_test.numpy())
  print(dy)
  print(dx)

  cm = confusion_matrix(dx, dy, labels=labels)
  df_cm = pd.DataFrame(cm)
  show_confusion_matrix(df_cm)

  actual_stance = Competition_dataset_frame['Stance'].values.tolist()
  df = Competition_dataset_frame
  df['Stance'] = y_pred
  predicted_stance = df['Stance'].values.tolist()  
  df.loc[df["Stance"] == 0, "Stance"] = "unrelated"
  df.loc[df["Stance"] == 1, "Stance"] = "agree"
  df.loc[df["Stance"] == 2, "Stance"] = "discuss"
  df.loc[df["Stance"] == 3, "Stance"] = "disagree"
  df.reset_index(drop=True, inplace=True)
  df = df[["Headline", "Body ID", "Stance"]]
  df.to_csv('answer.csv', index=False, encoding='utf-8')

  report_score([LABELS[e] for e in actual_stance],[LABELS[e] for e in predicted_stance])

