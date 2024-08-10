// The following table comes from <https://www.unicode.org/Public/12.0.0/ucd/Blocks.txt>.
// Keep it synchronized with <https://www.unicode.org/Public/UCD/latest/ucd/Blocks.txt>.

type UnicodeBlockLookup = {[key: string]: (char: number) => boolean};

export const unicodeBlockLookup: UnicodeBlockLookup = {
    // 'Basic Latin': (char) => char >= 0x0000 && char <= 0x007F,
    'Latin-1 Supplement': (char) => char >= 0x0080 && char <= 0x00FF,
    // 'Latin Extended-A': (char) => char >= 0x0100 && char <= 0x017F,
    // 'Latin Extended-B': (char) => char >= 0x0180 && char <= 0x024F,
    // 'IPA Extensions': (char) => char >= 0x0250 && char <= 0x02AF,
    // 'Spacing Modifier Letters': (char) => char >= 0x02B0 && char <= 0x02FF,
    // 'Combining Diacritical Marks': (char) => char >= 0x0300 && char <= 0x036F,
    // 'Greek and Coptic': (char) => char >= 0x0370 && char <= 0x03FF,
    // 'Cyrillic': (char) => char >= 0x0400 && char <= 0x04FF,
    // 'Cyrillic Supplement': (char) => char >= 0x0500 && char <= 0x052F,
    // 'Armenian': (char) => char >= 0x0530 && char <= 0x058F,
    //'Hebrew': (char) => char >= 0x0590 && char <= 0x05FF,
    'Arabic': (char) => char >= 0x0600 && char <= 0x06FF,
    //'Syriac': (char) => char >= 0x0700 && char <= 0x074F,
    'Arabic Supplement': (char) => char >= 0x0750 && char <= 0x077F,
    // 'Thaana': (char) => char >= 0x0780 && char <= 0x07BF,
    // 'NKo': (char) => char >= 0x07C0 && char <= 0x07FF,
    // 'Samaritan': (char) => char >= 0x0800 && char <= 0x083F,
    // 'Mandaic': (char) => char >= 0x0840 && char <= 0x085F,
    // 'Syriac Supplement': (char) => char >= 0x0860 && char <= 0x086F,
    'Arabic Extended-A': (char) => char >= 0x08A0 && char <= 0x08FF,
    'Devanagari': (char) => char >= 0x0900 && char <= 0x097F,
    'Bengali': (char) => char >= 0x0980 && char <= 0x09FF,
    // 'Gurmukhi': (char) => char >= 0x0A00 && char <= 0x0A7F,
    'Gujarati': (char) => char >= 0x0A80 && char <= 0x0AFF,
    // 'Oriya': (char) => char >= 0x0B00 && char <= 0x0B7F,
    'Tamil': (char) => char >= 0x0B80 && char <= 0x0BFF,
    'Telugu': (char) => char >= 0x0C00 && char <= 0x0C7F,
    // 'Kannada': (char) => char >= 0x0C80 && char <= 0x0CFF,
    // 'Malayalam': (char) => char >= 0x0D00 && char <= 0x0D7F,
    'Sinhala': (char) => char >= 0x0D80 && char <= 0x0DFF,
    // 'Thai': (char) => char >= 0x0E00 && char <= 0x0E7F,
    // 'Lao': (char) => char >= 0x0E80 && char <= 0x0EFF,
    'Tibetan': (char) => char >= 0x0F00 && char <= 0x0FFF,
    'Myanmar': (char) => char >= 0x1000 && char <= 0x109F,
    // 'Georgian': (char) => char >= 0x10A0 && char <= 0x10FF,
    'Hangul Jamo': (char) => char >= 0x1100 && char <= 0x11FF,
    // 'Ethiopic': (char) => char >= 0x1200 && char <= 0x137F,
    // 'Ethiopic Supplement': (char) => char >= 0x1380 && char <= 0x139F,
    // 'Cherokee': (char) => char >= 0x13A0 && char <= 0x13FF,
    'Unified Canadian Aboriginal Syllabics': (char) => char >= 0x1400 && char <= 0x167F,
    // 'Ogham': (char) => char >= 0x1680 && char <= 0x169F,
    // 'Runic': (char) => char >= 0x16A0 && char <= 0x16FF,
    // 'Tagalog': (char) => char >= 0x1700 && char <= 0x171F,
    // 'Hanunoo': (char) => char >= 0x1720 && char <= 0x173F,
    // 'Buhid': (char) => char >= 0x1740 && char <= 0x175F,
    // 'Tagbanwa': (char) => char >= 0x1760 && char <= 0x177F,
    'Khmer': (char) => char >= 0x1780 && char <= 0x17FF,
    // 'Mongolian': (char) => char >= 0x1800 && char <= 0x18AF,
    'Unified Canadian Aboriginal Syllabics Extended': (char) => char >= 0x18B0 && char <= 0x18FF,
    // 'Limbu': (char) => char >= 0x1900 && char <= 0x194F,
    // 'Tai Le': (char) => char >= 0x1950 && char <= 0x197F,
    // 'New Tai Lue': (char) => char >= 0x1980 && char <= 0x19DF,
    // 'Khmer Symbols': (char) => char >= 0x19E0 && char <= 0x19FF,
    // 'Buginese': (char) => char >= 0x1A00 && char <= 0x1A1F,
    // 'Tai Tham': (char) => char >= 0x1A20 && char <= 0x1AAF,
    // 'Combining Diacritical Marks Extended': (char) => char >= 0x1AB0 && char <= 0x1AFF,
    // 'Balinese': (char) => char >= 0x1B00 && char <= 0x1B7F,
    // 'Sundanese': (char) => char >= 0x1B80 && char <= 0x1BBF,
    // 'Batak': (char) => char >= 0x1BC0 && char <= 0x1BFF,
    // 'Lepcha': (char) => char >= 0x1C00 && char <= 0x1C4F,
    // 'Ol Chiki': (char) => char >= 0x1C50 && char <= 0x1C7F,
    // 'Cyrillic Extended-C': (char) => char >= 0x1C80 && char <= 0x1C8F,
    // 'Georgian Extended': (char) => char >= 0x1C90 && char <= 0x1CBF,
    // 'Sundanese Supplement': (char) => char >= 0x1CC0 && char <= 0x1CCF,
    // 'Vedic Extensions': (char) => char >= 0x1CD0 && char <= 0x1CFF,
    // 'Phonetic Extensions': (char) => char >= 0x1D00 && char <= 0x1D7F,
    // 'Phonetic Extensions Supplement': (char) => char >= 0x1D80 && char <= 0x1DBF,
    // 'Combining Diacritical Marks Supplement': (char) => char >= 0x1DC0 && char <= 0x1DFF,
    // 'Latin Extended Additional': (char) => char >= 0x1E00 && char <= 0x1EFF,
    // 'Greek Extended': (char) => char >= 0x1F00 && char <= 0x1FFF,
    'General Punctuation': (char) => char >= 0x2000 && char <= 0x206F,
    // 'Superscripts and Subscripts': (char) => char >= 0x2070 && char <= 0x209F,
    // 'Currency Symbols': (char) => char >= 0x20A0 && char <= 0x20CF,
    // 'Combining Diacritical Marks for Symbols': (char) => char >= 0x20D0 && char <= 0x20FF,
    'Letterlike Symbols': (char) => char >= 0x2100 && char <= 0x214F,
    'Number Forms': (char) => char >= 0x2150 && char <= 0x218F,
    // 'Arrows': (char) => char >= 0x2190 && char <= 0x21FF,
    // 'Mathematical Operators': (char) => char >= 0x2200 && char <= 0x22FF,
    'Miscellaneous Technical': (char) => char >= 0x2300 && char <= 0x23FF,
    'Control Pictures': (char) => char >= 0x2400 && char <= 0x243F,
    'Optical Character Recognition': (char) => char >= 0x2440 && char <= 0x245F,
    'Enclosed Alphanumerics': (char) => char >= 0x2460 && char <= 0x24FF,
    // 'Box Drawing': (char) => char >= 0x2500 && char <= 0x257F,
    // 'Block Elements': (char) => char >= 0x2580 && char <= 0x259F,
    'Geometric Shapes': (char) => char >= 0x25A0 && char <= 0x25FF,
    'Miscellaneous Symbols': (char) => char >= 0x2600 && char <= 0x26FF,
    // 'Dingbats': (char) => char >= 0x2700 && char <= 0x27BF,
    // 'Miscellaneous Mathematical Symbols-A': (char) => char >= 0x27C0 && char <= 0x27EF,
    // 'Supplemental Arrows-A': (char) => char >= 0x27F0 && char <= 0x27FF,
    // 'Braille Patterns': (char) => char >= 0x2800 && char <= 0x28FF,
    // 'Supplemental Arrows-B': (char) => char >= 0x2900 && char <= 0x297F,
    // 'Miscellaneous Mathematical Symbols-B': (char) => char >= 0x2980 && char <= 0x29FF,
    // 'Supplemental Mathematical Operators': (char) => char >= 0x2A00 && char <= 0x2AFF,
    'Miscellaneous Symbols and Arrows': (char) => char >= 0x2B00 && char <= 0x2BFF,
    // 'Glagolitic': (char) => char >= 0x2C00 && char <= 0x2C5F,
    // 'Latin Extended-C': (char) => char >= 0x2C60 && char <= 0x2C7F,
    // 'Coptic': (char) => char >= 0x2C80 && char <= 0x2CFF,
    // 'Georgian Supplement': (char) => char >= 0x2D00 && char <= 0x2D2F,
    // 'Tifinagh': (char) => char >= 0x2D30 && char <= 0x2D7F,
    // 'Ethiopic Extended': (char) => char >= 0x2D80 && char <= 0x2DDF,
    // 'Cyrillic Extended-A': (char) => char >= 0x2DE0 && char <= 0x2DFF,
    // 'Supplemental Punctuation': (char) => char >= 0x2E00 && char <= 0x2E7F,
    'CJK Radicals Supplement': (char) => char >= 0x2E80 && char <= 0x2EFF,
    'Kangxi Radicals': (char) => char >= 0x2F00 && char <= 0x2FDF,
    'Ideographic Description Characters': (char) => char >= 0x2FF0 && char <= 0x2FFF,
    'CJK Symbols and Punctuation': (char) => char >= 0x3000 && char <= 0x303F,
    'Hiragana': (char) => char >= 0x3040 && char <= 0x309F,
    'Katakana': (char) => char >= 0x30A0 && char <= 0x30FF,
    'Bopomofo': (char) => char >= 0x3100 && char <= 0x312F,
    'Hangul Compatibility Jamo': (char) => char >= 0x3130 && char <= 0x318F,
    'Kanbun': (char) => char >= 0x3190 && char <= 0x319F,
    'Bopomofo Extended': (char) => char >= 0x31A0 && char <= 0x31BF,
    'CJK Strokes': (char) => char >= 0x31C0 && char <= 0x31EF,
    'Katakana Phonetic Extensions': (char) => char >= 0x31F0 && char <= 0x31FF,
    'Enclosed CJK Letters and Months': (char) => char >= 0x3200 && char <= 0x32FF,
    'CJK Compatibility': (char) => char >= 0x3300 && char <= 0x33FF,
    'CJK Unified Ideographs Extension A': (char) => char >= 0x3400 && char <= 0x4DBF,
    'Yijing Hexagram Symbols': (char) => char >= 0x4DC0 && char <= 0x4DFF,
    'CJK Unified Ideographs': (char) => char >= 0x4E00 && char <= 0x9FFF,
    'Yi Syllables': (char) => char >= 0xA000 && char <= 0xA48F,
    'Yi Radicals': (char) => char >= 0xA490 && char <= 0xA4CF,
    // 'Lisu': (char) => char >= 0xA4D0 && char <= 0xA4FF,
    // 'Vai': (char) => char >= 0xA500 && char <= 0xA63F,
    // 'Cyrillic Extended-B': (char) => char >= 0xA640 && char <= 0xA69F,
    // 'Bamum': (char) => char >= 0xA6A0 && char <= 0xA6FF,
    // 'Modifier Tone Letters': (char) => char >= 0xA700 && char <= 0xA71F,
    // 'Latin Extended-D': (char) => char >= 0xA720 && char <= 0xA7FF,
    // 'Syloti Nagri': (char) => char >= 0xA800 && char <= 0xA82F,
    // 'Common Indic Number Forms': (char) => char >= 0xA830 && char <= 0xA83F,
    // 'Phags-pa': (char) => char >= 0xA840 && char <= 0xA87F,
    // 'Saurashtra': (char) => char >= 0xA880 && char <= 0xA8DF,
    // 'Devanagari Extended': (char) => char >= 0xA8E0 && char <= 0xA8FF,
    // 'Kayah Li': (char) => char >= 0xA900 && char <= 0xA92F,
    // 'Rejang': (char) => char >= 0xA930 && char <= 0xA95F,
    'Hangul Jamo Extended-A': (char) => char >= 0xA960 && char <= 0xA97F,
    // 'Javanese': (char) => char >= 0xA980 && char <= 0xA9DF,
    // 'Myanmar Extended-B': (char) => char >= 0xA9E0 && char <= 0xA9FF,
    // 'Cham': (char) => char >= 0xAA00 && char <= 0xAA5F,
    // 'Myanmar Extended-A': (char) => char >= 0xAA60 && char <= 0xAA7F,
    // 'Tai Viet': (char) => char >= 0xAA80 && char <= 0xAADF,
    // 'Meetei Mayek Extensions': (char) => char >= 0xAAE0 && char <= 0xAAFF,
    // 'Ethiopic Extended-A': (char) => char >= 0xAB00 && char <= 0xAB2F,
    // 'Latin Extended-E': (char) => char >= 0xAB30 && char <= 0xAB6F,
    // 'Cherokee Supplement': (char) => char >= 0xAB70 && char <= 0xABBF,
    // 'Meetei Mayek': (char) => char >= 0xABC0 && char <= 0xABFF,
    'Hangul Syllables': (char) => char >= 0xAC00 && char <= 0xD7AF,
    'Hangul Jamo Extended-B': (char) => char >= 0xD7B0 && char <= 0xD7FF,
    // 'High Surrogates': (char) => char >= 0xD800 && char <= 0xDB7F,
    // 'High Private Use Surrogates': (char) => char >= 0xDB80 && char <= 0xDBFF,
    // 'Low Surrogates': (char) => char >= 0xDC00 && char <= 0xDFFF,
    'Private Use Area': (char) => char >= 0xE000 && char <= 0xF8FF,
    'CJK Compatibility Ideographs': (char) => char >= 0xF900 && char <= 0xFAFF,
    // 'Alphabetic Presentation Forms': (char) => char >= 0xFB00 && char <= 0xFB4F,
    'Arabic Presentation Forms-A': (char) => char >= 0xFB50 && char <= 0xFDFF,
    // 'Variation Selectors': (char) => char >= 0xFE00 && char <= 0xFE0F,
    'Vertical Forms': (char) => char >= 0xFE10 && char <= 0xFE1F,
    // 'Combining Half Marks': (char) => char >= 0xFE20 && char <= 0xFE2F,
    'CJK Compatibility Forms': (char) => char >= 0xFE30 && char <= 0xFE4F,
    'Small Form Variants': (char) => char >= 0xFE50 && char <= 0xFE6F,
    'Arabic Presentation Forms-B': (char) => char >= 0xFE70 && char <= 0xFEFF,
    'Halfwidth and Fullwidth Forms': (char) => char >= 0xFF00 && char <= 0xFFEF,
    // 'Specials': (char) => char >= 0xFFF0 && char <= 0xFFFF,
    // 'Linear B Syllabary': (char) => char >= 0x10000 && char <= 0x1007F,
    // 'Linear B Ideograms': (char) => char >= 0x10080 && char <= 0x100FF,
    // 'Aegean Numbers': (char) => char >= 0x10100 && char <= 0x1013F,
    // 'Ancient Greek Numbers': (char) => char >= 0x10140 && char <= 0x1018F,
    // 'Ancient Symbols': (char) => char >= 0x10190 && char <= 0x101CF,
    // 'Phaistos Disc': (char) => char >= 0x101D0 && char <= 0x101FF,
    // 'Lycian': (char) => char >= 0x10280 && char <= 0x1029F,
    // 'Carian': (char) => char >= 0x102A0 && char <= 0x102DF,
    // 'Coptic Epact Numbers': (char) => char >= 0x102E0 && char <= 0x102FF,
    // 'Old Italic': (char) => char >= 0x10300 && char <= 0x1032F,
    // 'Gothic': (char) => char >= 0x10330 && char <= 0x1034F,
    // 'Old Permic': (char) => char >= 0x10350 && char <= 0x1037F,
    // 'Ugaritic': (char) => char >= 0x10380 && char <= 0x1039F,
    // 'Old Persian': (char) => char >= 0x103A0 && char <= 0x103DF,
    // 'Deseret': (char) => char >= 0x10400 && char <= 0x1044F,
    // 'Shavian': (char) => char >= 0x10450 && char <= 0x1047F,
    // 'Osmanya': (char) => char >= 0x10480 && char <= 0x104AF,
    // 'Osage': (char) => char >= 0x104B0 && char <= 0x104FF,
    // 'Elbasan': (char) => char >= 0x10500 && char <= 0x1052F,
    // 'Caucasian Albanian': (char) => char >= 0x10530 && char <= 0x1056F,
    // 'Linear A': (char) => char >= 0x10600 && char <= 0x1077F,
    // 'Cypriot Syllabary': (char) => char >= 0x10800 && char <= 0x1083F,
    // 'Imperial Aramaic': (char) => char >= 0x10840 && char <= 0x1085F,
    // 'Palmyrene': (char) => char >= 0x10860 && char <= 0x1087F,
    // 'Nabataean': (char) => char >= 0x10880 && char <= 0x108AF,
    // 'Hatran': (char) => char >= 0x108E0 && char <= 0x108FF,
    // 'Phoenician': (char) => char >= 0x10900 && char <= 0x1091F,
    // 'Lydian': (char) => char >= 0x10920 && char <= 0x1093F,
    // 'Meroitic Hieroglyphs': (char) => char >= 0x10980 && char <= 0x1099F,
    // 'Meroitic Cursive': (char) => char >= 0x109A0 && char <= 0x109FF,
    // 'Kharoshthi': (char) => char >= 0x10A00 && char <= 0x10A5F,
    // 'Old South Arabian': (char) => char >= 0x10A60 && char <= 0x10A7F,
    // 'Old North Arabian': (char) => char >= 0x10A80 && char <= 0x10A9F,
    // 'Manichaean': (char) => char >= 0x10AC0 && char <= 0x10AFF,
    // 'Avestan': (char) => char >= 0x10B00 && char <= 0x10B3F,
    // 'Inscriptional Parthian': (char) => char >= 0x10B40 && char <= 0x10B5F,
    // 'Inscriptional Pahlavi': (char) => char >= 0x10B60 && char <= 0x10B7F,
    // 'Psalter Pahlavi': (char) => char >= 0x10B80 && char <= 0x10BAF,
    // 'Old Turkic': (char) => char >= 0x10C00 && char <= 0x10C4F,
    // 'Old Hungarian': (char) => char >= 0x10C80 && char <= 0x10CFF,
    // 'Hanifi Rohingya': (char) => char >= 0x10D00 && char <= 0x10D3F,
    // 'Rumi Numeral Symbols': (char) => char >= 0x10E60 && char <= 0x10E7F,
    // 'Old Sogdian': (char) => char >= 0x10F00 && char <= 0x10F2F,
    // 'Sogdian': (char) => char >= 0x10F30 && char <= 0x10F6F,
    // 'Elymaic': (char) => char >= 0x10FE0 && char <= 0x10FFF,
    // 'Brahmi': (char) => char >= 0x11000 && char <= 0x1107F,
    // 'Kaithi': (char) => char >= 0x11080 && char <= 0x110CF,
    // 'Sora Sompeng': (char) => char >= 0x110D0 && char <= 0x110FF,
    // 'Chakma': (char) => char >= 0x11100 && char <= 0x1114F,
    // 'Mahajani': (char) => char >= 0x11150 && char <= 0x1117F,
    // 'Sharada': (char) => char >= 0x11180 && char <= 0x111DF,
    // 'Sinhala Archaic Numbers': (char) => char >= 0x111E0 && char <= 0x111FF,
    // 'Khojki': (char) => char >= 0x11200 && char <= 0x1124F,
    // 'Multani': (char) => char >= 0x11280 && char <= 0x112AF,
    // 'Khudawadi': (char) => char >= 0x112B0 && char <= 0x112FF,
    // 'Grantha': (char) => char >= 0x11300 && char <= 0x1137F,
    // 'Newa': (char) => char >= 0x11400 && char <= 0x1147F,
    // 'Tirhuta': (char) => char >= 0x11480 && char <= 0x114DF,
    // 'Siddham': (char) => char >= 0x11580 && char <= 0x115FF,
    // 'Modi': (char) => char >= 0x11600 && char <= 0x1165F,
    // 'Mongolian Supplement': (char) => char >= 0x11660 && char <= 0x1167F,
    // 'Takri': (char) => char >= 0x11680 && char <= 0x116CF,
    // 'Ahom': (char) => char >= 0x11700 && char <= 0x1173F,
    // 'Dogra': (char) => char >= 0x11800 && char <= 0x1184F,
    // 'Warang Citi': (char) => char >= 0x118A0 && char <= 0x118FF,
    // 'Nandinagari': (char) => char >= 0x119A0 && char <= 0x119FF,
    // 'Zanabazar Square': (char) => char >= 0x11A00 && char <= 0x11A4F,
    // 'Soyombo': (char) => char >= 0x11A50 && char <= 0x11AAF,
    // 'Pau Cin Hau': (char) => char >= 0x11AC0 && char <= 0x11AFF,
    // 'Bhaiksuki': (char) => char >= 0x11C00 && char <= 0x11C6F,
    // 'Marchen': (char) => char >= 0x11C70 && char <= 0x11CBF,
    // 'Masaram Gondi': (char) => char >= 0x11D00 && char <= 0x11D5F,
    // 'Gunjala Gondi': (char) => char >= 0x11D60 && char <= 0x11DAF,
    // 'Makasar': (char) => char >= 0x11EE0 && char <= 0x11EFF,
    // 'Tamil Supplement': (char) => char >= 0x11FC0 && char <= 0x11FFF,
    // 'Cuneiform': (char) => char >= 0x12000 && char <= 0x123FF,
    // 'Cuneiform Numbers and Punctuation': (char) => char >= 0x12400 && char <= 0x1247F,
    // 'Early Dynastic Cuneiform': (char) => char >= 0x12480 && char <= 0x1254F,
    // 'Egyptian Hieroglyphs': (char) => char >= 0x13000 && char <= 0x1342F,
    // 'Egyptian Hieroglyph Format Controls': (char) => char >= 0x13430 && char <= 0x1343F,
    // 'Anatolian Hieroglyphs': (char) => char >= 0x14400 && char <= 0x1467F,
    // 'Bamum Supplement': (char) => char >= 0x16800 && char <= 0x16A3F,
    // 'Mro': (char) => char >= 0x16A40 && char <= 0x16A6F,
    // 'Bassa Vah': (char) => char >= 0x16AD0 && char <= 0x16AFF,
    // 'Pahawh Hmong': (char) => char >= 0x16B00 && char <= 0x16B8F,
    // 'Medefaidrin': (char) => char >= 0x16E40 && char <= 0x16E9F,
    // 'Miao': (char) => char >= 0x16F00 && char <= 0x16F9F,
    // 'Ideographic Symbols and Punctuation': (char) => char >= 0x16FE0 && char <= 0x16FFF,
    // 'Tangut': (char) => char >= 0x17000 && char <= 0x187FF,
    // 'Tangut Components': (char) => char >= 0x18800 && char <= 0x18AFF,
    // 'Kana Supplement': (char) => char >= 0x1B000 && char <= 0x1B0FF,
    // 'Kana Extended-A': (char) => char >= 0x1B100 && char <= 0x1B12F,
    // 'Small Kana Extension': (char) => char >= 0x1B130 && char <= 0x1B16F,
    // 'Nushu': (char) => char >= 0x1B170 && char <= 0x1B2FF,
    // 'Duployan': (char) => char >= 0x1BC00 && char <= 0x1BC9F,
    // 'Shorthand Format Controls': (char) => char >= 0x1BCA0 && char <= 0x1BCAF,
    // 'Byzantine Musical Symbols': (char) => char >= 0x1D000 && char <= 0x1D0FF,
    // 'Musical Symbols': (char) => char >= 0x1D100 && char <= 0x1D1FF,
    // 'Ancient Greek Musical Notation': (char) => char >= 0x1D200 && char <= 0x1D24F,
    // 'Mayan Numerals': (char) => char >= 0x1D2E0 && char <= 0x1D2FF,
    // 'Tai Xuan Jing Symbols': (char) => char >= 0x1D300 && char <= 0x1D35F,
    // 'Counting Rod Numerals': (char) => char >= 0x1D360 && char <= 0x1D37F,
    // 'Mathematical Alphanumeric Symbols': (char) => char >= 0x1D400 && char <= 0x1D7FF,
    // 'Sutton SignWriting': (char) => char >= 0x1D800 && char <= 0x1DAAF,
    // 'Glagolitic Supplement': (char) => char >= 0x1E000 && char <= 0x1E02F,
    // 'Nyiakeng Puachue Hmong': (char) => char >= 0x1E100 && char <= 0x1E14F,
    // 'Wancho': (char) => char >= 0x1E2C0 && char <= 0x1E2FF,
    // 'Mende Kikakui': (char) => char >= 0x1E800 && char <= 0x1E8DF,
    // 'Adlam': (char) => char >= 0x1E900 && char <= 0x1E95F,
    // 'Indic Siyaq Numbers': (char) => char >= 0x1EC70 && char <= 0x1ECBF,
    // 'Ottoman Siyaq Numbers': (char) => char >= 0x1ED00 && char <= 0x1ED4F,
    // 'Arabic Mathematical Alphabetic Symbols': (char) => char >= 0x1EE00 && char <= 0x1EEFF,
    // 'Mahjong Tiles': (char) => char >= 0x1F000 && char <= 0x1F02F,
    // 'Domino Tiles': (char) => char >= 0x1F030 && char <= 0x1F09F,
    // 'Playing Cards': (char) => char >= 0x1F0A0 && char <= 0x1F0FF,
    // 'Enclosed Alphanumeric Supplement': (char) => char >= 0x1F100 && char <= 0x1F1FF,
    // 'Enclosed Ideographic Supplement': (char) => char >= 0x1F200 && char <= 0x1F2FF,
    // 'Miscellaneous Symbols and Pictographs': (char) => char >= 0x1F300 && char <= 0x1F5FF,
    // 'Emoticons': (char) => char >= 0x1F600 && char <= 0x1F64F,
    // 'Ornamental Dingbats': (char) => char >= 0x1F650 && char <= 0x1F67F,
    // 'Transport and Map Symbols': (char) => char >= 0x1F680 && char <= 0x1F6FF,
    // 'Alchemical Symbols': (char) => char >= 0x1F700 && char <= 0x1F77F,
    // 'Geometric Shapes Extended': (char) => char >= 0x1F780 && char <= 0x1F7FF,
    // 'Supplemental Arrows-C': (char) => char >= 0x1F800 && char <= 0x1F8FF,
    // 'Supplemental Symbols and Pictographs': (char) => char >= 0x1F900 && char <= 0x1F9FF,
    // 'Chess Symbols': (char) => char >= 0x1FA00 && char <= 0x1FA6F,
    // 'Symbols and Pictographs Extended-A': (char) => char >= 0x1FA70 && char <= 0x1FAFF,
    'CJK Unified Ideographs Extension B': (char) => char >= 0x20000 && char <= 0x2A6DF,
    'CJK Unified Ideographs Extension C': (char) => char >= 0x2A700 && char <= 0x2B73F,
    'CJK Unified Ideographs Extension D': (char) => char >= 0x2B740 && char <= 0x2B81F,
    'CJK Unified Ideographs Extension E': (char) => char >= 0x2B820 && char <= 0x2CEAF,
    'CJK Unified Ideographs Extension F': (char) => char >= 0x2CEB0 && char <= 0x2EBEF,
    'CJK Compatibility Ideographs Supplement': (char) => char >= 0x2F800 && char <= 0x2FA1F,
    // 'Tags': (char) => char >= 0xE0000 && char <= 0xE007F,
    // 'Variation Selectors Supplement': (char) => char >= 0xE0100 && char <= 0xE01EF,
    // 'Supplementary Private Use Area-A': (char) => char >= 0xF0000 && char <= 0xFFFFF,
    // 'Supplementary Private Use Area-B': (char) => char >= 0x100000 && char <= 0x10FFFF,
};
