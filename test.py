#!/usr/bin/python
from arekit.common.entities.base import Entity
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.token import Token
from arekit.processing.text.tokens import Tokens
from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO


x = RuSentRelNetworkIO('a')

s = MystemWrapper()
x.read_synonyms_collection(stemmer=s)
news, parsed_news = x.read_parsed_news(doc_id=1, keep_tokens=True, stemmer=s)
assert(isinstance(parsed_news, ParsedNews))


def show(t):
    if isinstance(t, unicode):
        return t
    elif isinstance(t, Entity):
        return u'{{{} ({})}}'.format(t.Value, t.IdInDocument)
    elif isinstance(t, TextFrameVariant):
        return u'<{} ({})>'.format(t.Variant.get_value(), t.Variant.FrameID)
    elif isinstance(t, Token):
        return t.get_token_value()


for t in parsed_news.iter_terms():
    print show(t),
    if isinstance(t, Token) and t.get_token_value() == Tokens.DOT:
        print