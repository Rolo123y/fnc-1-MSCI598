from traceback import print_tb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, pipeline, AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import sys
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from feature_engineering import clean
from transformers.modeling_utils import load_state_dict
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version
from transformers import BertModel, get_linear_schedule_with_warmup
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
from nltk.corpus import stopwords

stop = stopwords.words('english')
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
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids.to(device),
      attention_mask=attention_mask.to(device),
      return_dict=False,
    )

    output = self.drop(pooled_output)
    return self.out(output)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  counter=0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].type(torch.LongTensor).to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    counter = counter+1

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].type(torch.LongTensor).to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

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
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'RUNNING ON DEVICE: {device}')

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    TrainSet = DataSet()
    train_stances_frame = pd.DataFrame(TrainSet.stances)
    train_articles_frame = pd.DataFrame(list(TrainSet.articles.items()),columns = ['Body ID','article']) 
    TrainingSet_frame = train_stances_frame.merge(train_articles_frame[["Body ID","article"]], on="Body ID", how="left")
    TrainingSet_frame.loc[TrainingSet_frame["Stance"] == "unrelated", "Stance"] = 0
    TrainingSet_frame.loc[TrainingSet_frame["Stance"] == "agree", "Stance"] = 1
    TrainingSet_frame.loc[TrainingSet_frame["Stance"] == "discuss", "Stance"] = 2
    TrainingSet_frame.loc[TrainingSet_frame["Stance"] == "disagree", "Stance"] = 3

    # REMOVE SPECIAL CHARACTERS
    TrainingSet_frame['Headline'] = TrainingSet_frame['Headline'].apply(lambda x: clean(x))
    TrainingSet_frame['article'] = TrainingSet_frame['article'].apply(lambda x: clean(x))

    # REMOVE STOPWORDS
    TrainingSet_frame['Headline'] = TrainingSet_frame['Headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    TrainingSet_frame['article'] = TrainingSet_frame['article'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    TrainingSet_frame = TrainingSet_frame.iloc[:400]

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)

    # Split the training set into train + validation + test
    df_train, df_test = train_test_split(TrainingSet_frame, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)
    # print(df_train.shape, df_val.shape, df_test.shape)

    BATCH_SIZE = 2
    train_data_loader = create_data_loader(df_train, tokenizer, 500, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, 500, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, 500, BATCH_SIZE)

    train_data = next(iter(train_data_loader))
    val_data = next(iter(val_data_loader))
    test_data = next(iter(test_data_loader))

    # Sentiment Classification with BERT
    model = SentimentClassifier(4)
    model = model.to(device)
    input_ids = train_data['input_ids'].to(device)
    attention_mask = train_data['attention_mask'].to(device)
    a = F.softmax(model(input_ids, attention_mask), dim=1)

    # TRAINING THE MODEL HERE
    EPOCHS = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    import time
    start_time = time.time()

    # FIND THE MODEL WHICH HAS THE BEST ACCURACY BY TRAINING AND EVALUATING ON EACH EPOCH
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):

      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)

      train_acc, train_loss = train_epoch(
        model.to(device),
        train_data_loader,    
        loss_fn, 
        optimizer, 
        device, 
        scheduler, 
        len(df_train)
      )

      print(f'Train loss {train_loss} accuracy {train_acc}')

      val_acc, val_loss = eval_model(
        model.to(device),
        val_data_loader,
        loss_fn, 
        device, 
        len(df_val)
      )

      print(f'Val loss {val_loss} accuracy {val_acc}')

      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)

      if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'model/best_model_state.bin')
        tokenizer.save_vocabulary('model/best_model_tokenizer.bin')

        best_accuracy = val_acc
        print(f'Better accuracy found: {best_accuracy}')

      print("--- %s minutes ---" % round((time.time() - start_time)/60, 2))

    print("\n--- %s minutes ---" % round((time.time() - start_time)/60, 2))

    for i in range(0, len(history['train_acc'])):
      tens1 = history['train_acc'][i].cpu().detach().numpy()
      tens2 = history['val_acc'][i].cpu().detach().numpy()
      history['train_acc'][i] = tens1
      history['val_acc'][i] = tens2

    a = history['train_acc']
    b = history['val_acc']
    plt.plot(a, label='train accuracy')
    plt.plot(b, label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()

    y_headline_text, y_article_text, y_pred, y_pred_probs, y_test = get_predictions(
      model,
      test_data_loader
    )
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
    