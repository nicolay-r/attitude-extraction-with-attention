#!/usr/bin/python
from arekit.common.entities.base import Entity
from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.common.linked_text_opinions.collection import LabeledLinkedTextOpinionCollection
from arekit.common.parsed_news.collection import ParsedNewsCollection
from arekit.common.parsed_news.base import ParsedNews
from arekit.common.text_frame_variant import TextFrameVariant
from arekit.common.text_opinions.end_type import EntityEndType
from arekit.common.text_opinions.helper import TextOpinionHelper
from arekit.common.text_opinions.base import TextOpinion
from arekit.networks.context.sample import InputSample
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.token import Token
from arekit.source.ruattitudes.helpers.linked_text_opinions import RuAttitudesNewsTextOpinionExtractorHelper
from arekit.source.ruattitudes.news import RuAttitudesNews
from arekit.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.source.rusentiframes.helpers.parse import RuSentiFramesParseHelper
from experiments.rusentrel_ds.rusentrel_ds_io import RuSentRelWithRuAttitudesIO


def show(t):
    if isinstance(t, unicode):
        return t
    elif isinstance(t, Entity):
        return u'{{{}}}'.format(t.Value)
    elif isinstance(t, Token):
        return t.get_token_value()
    elif isinstance(t, TextFrameVariant):
        return u'<{} ({})>'.format(t.Variant.get_value(), t.Variant.FrameID)


io = RuSentRelWithRuAttitudesIO('a')
s = MystemWrapper()
io.read_synonyms_collection(stemmer=s)


def __check_text_opinion(text_opinion):
    assert (isinstance(text_opinion, TextOpinion))
    return InputSample.check_ability_to_create_sample(
        window_size=50,
        text_opinion=text_opinion)


frames = RuSentiFramesCollection.read_collection()
frame_variants = FrameVariantsCollection.from_iterable(
    variants_with_id=frames.iter_frame_id_and_variants(),
    stemmer=s)

pnc = ParsedNewsCollection()
text_opinions = LabeledLinkedTextOpinionCollection(pnc)

doc_ids = [400, 479, 532, 2530, 3610]
for doc_id in doc_ids:
    print "-------------------------"
    print "DOC_ID: {}".format(doc_id)
    news, parsed_news = io.read_parsed_news(doc_id=doc_id, keep_tokens=True, stemmer=s)
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(news, RuAttitudesNews))

    parsed_news.modify_parsed_sentences(
        lambda sentence: RuSentiFramesParseHelper.parse_frames_in_parsed_text(
            frame_variants_collection=frame_variants,
            parsed_text=sentence)
    )

    print u" ".join([show(t) for t in parsed_news.iter_terms()]).encode('utf-8')

    pnc.add(parsed_news)
    opinion_collection = io.read_etalon_opinion_collection(doc_id=doc_id)

    for o in opinion_collection:
        print u"OPINION: {}->{} ({})".format(o.SourceValue, o.TargetValue, o.Sentiment.to_str())

    RuAttitudesNewsTextOpinionExtractorHelper.add_entries(
        text_opinion_collection=text_opinions,
        news=news,
        check_text_opinion_is_correct=__check_text_opinion)

for o in text_opinions:
    assert(isinstance(o, TextOpinion))

    s = TextOpinionHelper.extract_entity_value(o, EntityEndType.Source)
    t = TextOpinionHelper.extract_entity_value(o, EntityEndType.Target)
    s_si = TextOpinionHelper.extract_entity_sentence_index(o, EntityEndType.Source)
    t_si = TextOpinionHelper.extract_entity_sentence_index(o, EntityEndType.Target)

    # print "NEWS: {}".format(o.NewsID)
    # if TextOpinionHelper.CheckEndsHasSameSentenceIndex(o):
    #     print u"OK: ({}->{}) -- {}".format(s, t, str(o.Sentiment.to_str())).encode('utf-8')

    if not TextOpinionHelper.check_ends_has_same_sentence_index(o):
        print s_si, t_si
        print u"FAILED:  ({}->{}) -- {}".format(s, t, str(o.Sentiment.to_str())).encode('utf-8')
