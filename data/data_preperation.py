import pandas as pd
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import regex as re
import time
from multiprocessing import Pool

class HateSpeechDS(Dataset):
    def __init__(self, path):
        self.path = path
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()
        self.df = self.preprocess_df()
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    def preprocess_df(self):
        df = pd.read_csv(self.path)
        desired_cols = ['tweet', 'class']
        label_map = {'hate_speech': 1, 'offensive_language': 1, 'neither': 0}
        df = df[desired_cols].copy()
        df['class'] = df['class'].replace(label_map)
        df['class'] = df['class'].replace({2:0})
        return df

    def process_text(self, text):
        modified_text = text.split(":")
        if len(modified_text) > 1: text = "".join(modified_text[1:])
        else : text = modified_text[0] 
        text = " ".join(text.split())
        text = str(text).lower()
        text = re.sub("[\.\,\!\?\:\;\-\=]", " ", text)
        pattern = r"^\s*" 
        text = re.sub(pattern, "", text)
        text = re.sub('\rt:/g',' ',text)#r
        text = re.sub('https?://\S+|www\.\S+', '', text)#urls
        text = re.sub('\w*\d\w*', '', text)#digits
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(self.punctuation), '', text)
        text = re.sub('\n', '', text)
        ## remove retweets
        text = re.sub('(rt\: )|(rt\:)|(rt \: )|(rt )', ' ', text)
        text = [word for word in text.split(' ') if word not in self.stop_words]
        text =" ".join(text)
        text = [self.lemmatizer.lemmatize(word) for word in text.split(' ')]
        text =" ".join(text).lstrip()
        return text

    # def transform_df(self):
    #     for i in range(len(self.df)):
    #         val = self.process_text(self.df["tweet"][i])
    #         self.df.loc[i, "tweet"] = val

    def transform_df(self, num_workers = 4):
        with Pool(processes=num_workers) as pool:
            processed_tweets = pool.map(self.process_text, self.df['tweet'].tolist())
        self.df['tweet'] = processed_tweets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return list(self.df.iloc[idx, :])


if __name__ == "__main__":
    path = "/workspaces/Hate-Speech-Detection/data/labeled_data.csv"
    dataset = HateSpeechDS(path)
    start = time.time()
    dataset.transform_df()
    end = time.time()
    print(f"time used: {end - start}")
    print(dataset.df.head(3))
    print(dataset.df["tweet"][2])
    