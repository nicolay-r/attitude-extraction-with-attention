# Studying Attention Models in Sentiment Attitude Extraction Task

This section will be updated.

> **Upd May 28'th, 2020:** An additional restriction towards entity pairs considered as an attitude in context. We treat pairs between object and subject appeared in context **only when** the distance between them in words (terms) not greater than 10. [[source code reference]](https://github.com/nicolay-r/attitude-extraction-with-attention/blob/058e779a82a076089e3c961cfab996c62066ee41/experiments/rusentrel/neutrals.py#L180)

## Dependencies

* Python-2.7
* arekit-0.20.0 

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
(RuSentiLex) are a part o``f the related library.

* **Word2Vec**: installation assumes to run a ``download.sh`` script:
```
cd data && ./download.sh
```

## References

This section will be updated.
