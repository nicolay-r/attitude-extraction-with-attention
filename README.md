# Studying Attention Models in Sentiment Attitude Extraction Task

## Dependencies


## Installation

* **Core library installation:** All the implementation depends on 
core library for *sentiment attitude extraction*, 
dubbed as [**arekit-0.20.0**](https://github.com/nicolay-r/AREkit/blob/0.20.0-nldb-rc/README.md):
> **NOTE:** it is important to download in ``arekit`` directory.
```
# Download arekit-0.20.0
git clone --single-branch --branch 0.20.0-nldb-rc https://github.com/nicolay-r/AREkit arekit

# Install dependencies
pip install -r arekit/dependencies.txt
```
* **Resources**: Since **arekit-0.20.0** all the resources such as collections 
(RuAttitudes, RuSentRel) and lexicons 
(RuSentiLex) are a part of the related library.

* **Word2Vec**: installation assumes to run a ``download.sh`` script:
```
cd data && ./download.sh
```
