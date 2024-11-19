# zillizcloud_test_for_expedia

## 0. Config

modify `./config.txt`
```
uri#https://
token#
```


## 1. Dataset Preparation

```sh
mkdir data
wget -P ./data https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip ./data/glove.840B.300d.zip -d ./data
```

## 2. Create Collection and Insert Data
use `Python >= 3.8`

```sh
pip install -r requirements.txt
python data_prepare.py
```

## 3. Concurrent Test with Pymilvus
use `Python >= 3.8`

```sh
python conc_pymilvus.py
```

## 4. Concurrent Test with Restful
use `Java 8`
```sh
javac -d . ConcTestRestful.java
java ConcTestRestful config.txt
```