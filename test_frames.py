from arekit.common.frame_variants.collection import FrameVariantsCollection
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.rusentiframes.collection import RuSentiFramesCollection


def iter_frames():
    stemmer = MystemWrapper()
    frames_collection = RuSentiFramesCollection.read_collection()
    frame_variants = FrameVariantsCollection.from_iterable(
        variants_with_id=frames_collection.iter_frame_id_and_variants(),
        stemmer=stemmer)

    for v, _ in frame_variants.iter_variants():
        yield v


frame_values_list = list(iter_frames())
for fv in frame_values_list:
    print u'"{}"'.format(fv)
