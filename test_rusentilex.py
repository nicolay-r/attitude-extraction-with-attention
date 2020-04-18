# -*- coding: utf-8 -*-

from arekit.source.lexicons.rusentilex import RuSentiLexLexicon

l = RuSentiLexLexicon.from_zip()
for term in l:
    print term

print u'юдофоб' in l
