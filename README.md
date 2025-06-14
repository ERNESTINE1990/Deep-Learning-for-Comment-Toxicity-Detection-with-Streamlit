# Deep-Learning-for-Comment-Toxicity-Detection-with-Streamlit

Deep Learning for Comment Toxicity Detection is a field of research and application focused on using deep learning models to identify and classify toxic or harmful language in user-generated comments. 
This can include hate speech, harassment, profanity, insults, and other forms of abusive content. Below is a structured overview to help you understand and/or write a paper or project in this domain.

# Problem Statement:

Online communities and social media platforms have become integral parts of modern communication, facilitating interactions and discussions on various topics. However, the prevalence of toxic comments, which include harassment, hate speech, and offensive language, poses significant challenges to maintaining healthy and constructive online discourse. To address this issue, there is a pressing need for automated systems capable of detecting and flagging toxic comments in real-time.

# objective:

The objective of this project is to develop a deep learning-based comment toxicity model using Python. This model will analyze text input from online comments and predict the likelihood of each comment being toxic. By accurately identifying toxic comments, the model will assist platform moderators and administrators in taking appropriate actions to mitigate the negative impact of toxic behavior, such as filtering, warning users, or initiating further review processes.

# Business Use Cases:

. Social Media Platforms: Social media platforms can utilize the developed comment toxicity model to automatically detect and filter out toxic comments in real-time.​

. Online Forums and Communities: Forums and community websites can integrate the toxicity detection model to moderate user-generated content efficiently.​

. Content Moderation Services: Companies offering content moderation services for online platforms can leverage the developed model to enhance their moderation capabilities.​

. Brand Safety and Reputation Management: Brands and advertisers can use the toxicity detection model to ensure that their advertisements and sponsored content appear in safe and appropriate online environments.​

. E-learning Platforms and Educational Websites: E-learning platforms and educational websites can employ the toxicity detection model to create safer online learning environments for students and educators.​

. News Websites and Media Outlets: News websites and media outlets can utilize the toxicity detection model to moderate user comments on articles and posts.​

# Technical Tags:

 *  Python
 *  Deep Learning
 *  Neural Networks
 *  NLP
 *  Model Training
 *  Model Evaluation
 *  Streamlit
 *  Model Deployment

# Dataset:

https://drive.google.com/drive/folders/1WXLTp57_TYa61rcPfQIzRUcE1Rz76Emk

# Necessary libraries:

* Basic libraries for data manipulation and visualization
  
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re # for regular expressions text cleaning

* Deep learning libraries - pytorch

import torch

from torch.utils.data import DataLoader, Dataset

from torch import nn

import torch.nn as nn

from torch.optim import AdamW

* Hugging Face Transformers library

from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

 * Sklearn for model evaluation
   
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,classification_report,hamming_loss,accuracy_score



