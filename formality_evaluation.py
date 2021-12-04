import logging
import re
import Mykytea
import os
import re
from tqdm import tqdm

# informal -> polite example
# input (informal):
# 戦/动词/せん わ/语尾/わ な/助动词/な い/语尾/い
# match: 动词 (语尾|助动|助动词)+
# head: '戦', suffixes: 'わ な い'
# identify conjugation pattern:
# '戦' -> v5u conjugation
# convert suffixes to polite:
# 'わ な い' -> 'い ま せ ん'
# output (polite):
# 戦 い ま せ ん

out_format = 'polite'
formats = set(['informal', 'polite', 'formal'])

verb_tag = '動詞'
auxv_tag = '助動詞'
tails = set(['語尾', '助動', '助動詞'])
copula = set(['だ', 'で'])

groups = {
    'u' : set(['う', 'い', 'わ']),
    'k' : set(['く', 'き', 'か']),
    's' : set(['す', 'し', 'さ']),
    't' : set(['つ', 'ち', 'た']),
    'n' : set(['ぬ', 'に', 'な']),
    'm' : set(['む', 'み', 'ま']),
    'r' : set(['る', 'り', 'ら']),
    'g' : set(['ぐ', 'ぎ', 'が']),
    'b' : set(['ぶ', 'び', 'ば']),
}

pattern_map = {}

# TODO: add patterns for irregular verbs
# suru     する
# kuru     来る
# v1-s     呉れ/动词/くれ る/语尾/る
# v5k-s    行/动词/い く/语尾/く
# v5r-i    有/动词/あ る/语尾/る
to_polite_conj_map = {
    # -u verbs pattern e.g. 言う iu
    'v5u' : {
        'う' : 'い ま す',
        'わ な い' : 'い ま せ ん',
        'っ た' : 'い ま し た',
        'わ な かっ た' : 'い ま せ ん で し た',
        'わ れ る' : 'わ れ ま す',
        'わ れ な い' : 'わ れ ま せ ん',
        'わ れ た' : 'わ れ ま し た',
        'わ れ な かっ た' : 'わ れ ま せ ん で し た', 
        'わ せ る' : 'わ せ ま す',
        'わ せ な い' : 'わ せ ま せ ん',
        'わ せ た' : 'わ せ ま し た',
        'わ せ られ な かっ た' : 'わ せ られ ま せ ん で し た',
        'わ せ られ る' : 'わ せ られ ま す',
        'わ せ られ な い' : 'わ せ られ ま せ ん',
        'わ せ られ た' : 'わ せ られ ま し た',
        'わ せ られ な かっ た' : 'わ せ られ ま せ ん で し た',
        },
    # -ku verbs pattern e.g. 开く hiraku
    'v5k' : {
        'く' : 'き ま す',
        'か な い' : 'き ま せ ん',
        'い た' : 'き ま し た',
        'か な かっ た' : 'き ま せ ん で し た',
        'か れ る' : 'か れ ま す',
        'か れ な い' : 'か れ ま せ ん',
        'か れ た' : 'か れ ま し た',
        'か れ な かっ た' : 'か れ ま せ ん で し た', 
        'か せ る' : 'か せ ま す',
        'か せ な い' : 'か せ ま せ ん',
        'か せ た' : 'か せ ま し た',
        'か せ られ な かっ た' : 'か せ られ ま せ ん で し た',
        'か せ られ る' : 'か せ られ ま す',
        'か せ られ な い' : 'か せ られ ま せ ん',
        'か せ られ た' : 'か せ られ ま し た',
        'か せ られ な かっ た' : 'か せ られ ま せ ん で し た',
        },
    # -su verbs pattern e.g. 话す hanasu
    'v5s' : {
        'す' : 'し ま す',
        'さ な い' : 'し ま せ ん',
        'し た' : 'し ま し た',
        'さ な かっ た' : 'し ま せ ん で し た',
        'さ れ る' : 'さ れ ま す',
        'さ れ な い' : 'さ れ ま せ ん',
        'さ れ た' : 'さ れ ま し た',
        'さ れ な かっ た' : 'さ れ ま せ ん で し た', 
        'さ せ る' : 'さ せ ま す',
        'さ せ な い' : 'さ せ ま せ ん',
        'さ せ た' : 'さ せ ま し た',
        'さ せ られ な かっ た' : 'さ せ られ ま せ ん で し た',
        'さ せ られ る' : 'さ せ られ ま す',
        'さ せ られ な い' : 'さ せ られ ま せ ん',
        'さ せ られ た' : 'さ せ られ ま し た',
        'さ せ られ な かっ た' : 'さ せ られ ま せ ん で し た',
        },
    # -tsu verbs pattern e.g. 持つ motsu
    'v5t' : {
        'つ' : 'ち ま す',
        'た な い' : 'ち ま せ ん',
        'っ た' : 'ち ま し た',
        'た な かっ た' : 'ち ま せ ん で し た',
        'た れ る' : 'た れ ま す',
        'た れ な い' : 'た れ ま せ ん',
        'た れ た' : 'た れ ま し た',
        'た れ な かっ た' : 'た れ ま せ ん で し た', 
        'た せ る' : 'た せ ま す',
        'た せ な い' : 'た せ ま せ ん',
        'た せ た' : 'た せ ま し た',
        'た せ られ な かっ た' : 'た せ られ ま せ ん で し た',
        'た せ られ る' : 'た せ られ ま す',
        'た せ られ な い' : 'た せ られ ま せ ん',
        'た せ られ た' : 'た せ られ ま し た',
        'た せ られ な かっ た' : 'た せ られ ま せ ん で し た',
        },
    # -nu verbs pattern e.g. 死ぬ shinu
    'v5n' : {
        'ぬ' : 'に ま す',
        'な な い' : 'に ま せ ん',
        'ん だ' : 'に ま し た',
        'な な かっ た' : 'に ま せ ん で し た',
        'な れ る' : 'な れ ま す',
        'な れ な い' : 'な れ ま せ ん',
        'な れ た' : 'な れ ま し た',
        'な れ な かっ た' : 'な れ ま せ ん で し た', 
        'な せ る' : 'な せ ま す',
        'な せ な い' : 'な せ ま せ ん',
        'な せ た' : 'な せ ま し た',
        'な せ られ な かっ た' : 'な せ られ ま せ ん で し た',
        'な せ られ る' : 'な せ られ ま す',
        'な せ られ な い' : 'な せ られ ま せ ん',
        'な せ られ た' : 'な せ られ ま し た',
        'な せ られ な かっ た' : 'な せ られ ま せ ん で し た',
        },
    # -mu verbs pattern e.g. 読む yomu
    'v5m' : {
        'む' : 'み ま す',
        'ま な い' : 'み ま せ ん',
        'ん だ' : 'み ま し た',
        'ま な かっ た' : 'み ま せ ん で し た',
        'ま れ る' : 'ま れ ま す',
        'ま れ な い' : 'ま れ ま せ ん',
        'ま れ た' : 'ま れ ま し た',
        'ま れ な かっ た' : 'ま れ ま せ ん で し た', 
        'ま せ る' : 'ま せ ま す',
        'ま せ な い' : 'ま せ ま せ ん',
        'ま せ た' : 'ま せ ま し た',
        'ま せ られ な かっ た' : 'ま せ られ ま せ ん で し た',
        'ま せ られ る' : 'ま せ られ ま す',
        'ま せ られ な い' : 'ま せ られ ま せ ん',
        'ま せ られ た' : 'ま せ られ ま し た',
        'ま せ られ な かっ た' : 'ま せ られ ま せ ん で し た',
        },
    # -ru verbs pattern (godan) e.g. 走る hashiru
    'v5r' : {
        'る' : 'り ま す',
        'ら な い' : 'り ま せ ん',
        'っ た' : 'り ま し た',
        'ら な かっ た' : 'り ま せ ん で し た',
        'ら れ る' : 'ら れ ま す',
        'ら れ な い' : 'ら れ ま せ ん',
        'ら れ た' : 'ら れ ま し た',
        'ら れ な かっ た' : 'ら れ ま せ ん で し た', 
        'ら せ る' : 'ら せ ま す',
        'ら せ な い' : 'ら せ ま せ ん',
        'ら せ た' : 'ら せ ま し た',
        'ら せ られ な かっ た' : 'ら せ られ ま せ ん で し た',
        'ら せ られ る' : 'ら せ られ ま す',
        'ら せ られ な い' : 'ら せ られ ま せ ん',
        'ら せ られ た' : 'ら せ られ ま し た',
        'ら せ られ な かっ た' : 'ら せ られ ま せ ん で し た',
        },
    # -ru verbs pattern (ichidan) e.g. 食べる taberu
    'v1' : {
        'る' : 'ま す',
        'な い' : 'ま せ ん',
        'た' : 'ま し た',
        'な かっ た' : 'ま せ ん で し た',
        'れ る' : 'れ ま す',
        'れ な い' : 'れ ま せ ん',
        'れ た' : 'れ ま し た',
        'れ な かっ た' : 'れ ま せ ん で し た', 
        'せ る' : 'せ ま す',
        'せ な い' : 'せ ま せ ん',
        'せ た' : 'せ ま し た',
        'せ られ な かっ た' : 'せ られ ま せ ん で し た',
        'せ られ る' : 'せ られ ま す',
        'せ られ な い' : 'せ られ ま せ ん',
        'せ られ た' : 'せ られ ま し た',
        'せ られ な かっ た' : 'せ られ ま せ ん で し た',
        },
    # -gu verbs pattern e.g. 泳ぐ oyogu
    'v5g' : {
        'ぐ' : 'ぎ ま す',
        'が な い' : 'ぎ ま せ ん',
        'い だ' : 'ぎ ま し た',
        'が な かっ た' : 'ぎ ま せ ん で し た',
        'が れ る' : 'が れ ま す',
        'が れ な い' : 'が れ ま せ ん',
        'が れ た' : 'が れ ま し た',
        'が れ な かっ た' : 'が れ ま せ ん で し た', 
        'が せ る' : 'が せ ま す',
        'が せ な い' : 'が せ ま せ ん',
        'が せ た' : 'が せ ま し た',
        'が せ られ な かっ た' : 'が せ られ ま せ ん で し た',
        'が せ られ る' : 'が せ られ ま す',
        'が せ られ な い' : 'が せ られ ま せ ん',
        'が せ られ た' : 'が せ られ ま し た',
        'が せ られ な かっ た' : 'が せ られ ま せ ん で し た',
        },
    # -bu verbs pattern e.g. 呼ぶ yobu
    'v5b' : {
        'ぶ' : 'び ま す',
        'ば な い' : 'び ま せ ん',
        'ん だ' : 'び ま し た',
        'ば な かっ た' : 'び ま せ ん で し た',
        'ば れ る' : 'ば れ ま す',
        'ば れ な い' : 'ば れ ま せ ん',
        'ば れ た' : 'ば れ ま し た',
        'ば れ な かっ た' : 'ば れ ま せ ん で し た', 
        'ば せ る' : 'ば せ ま す',
        'ば せ な い' : 'ば せ ま せ ん',
        'ば せ た' : 'ば せ ま し た',
        'ば せ られ な かっ た' : 'ば せ られ ま せ ん で し た',
        'ば せ られ る' : 'ば せ られ ま す',
        'ば せ られ な い' : 'ば せ られ ま せ ん',
        'ば せ られ た' : 'ば せ られ ま し た',
        'ば せ られ な かっ た' : 'ば せ られ ま せ ん で し た',
        },
}

# labels for sentence formality classification
formality_labels = {
    "informal": 0,
    "formal": 1,
    "inconclusive": .5
}

# reverse to_polite_conj_map to get to_informal_conj_map
to_informal_conj_map = {}
for group in to_polite_conj_map:
    to_informal_conj_map[group] = {}
    for key in to_polite_conj_map[group]:
        value = to_polite_conj_map[group][key]
        to_informal_conj_map[group][value] = key

# suffixes maps
to_polite_suffixes_map = {}
for pattern in to_polite_conj_map:
    # take last letter from pattern as group label
    # e.g. v5k -> k
    group = pattern[-1]
    for s in to_polite_conj_map[pattern].keys():
        to_polite_suffixes_map[s] = group

to_informal_suffixes_map = {}
for pattern in to_informal_conj_map:
    # take last letter from pattern as group label
    # e.g. v5k -> k
    group = pattern[-1]
    for s in to_informal_conj_map[pattern].keys():
        to_informal_suffixes_map[s] = group

maps = {
    'to_polite': {
        'conj': to_polite_conj_map,
        'suffixes': to_polite_suffixes_map
    },
    'to_informal': {
        'conj': to_informal_conj_map,
        'suffixes': to_informal_suffixes_map
    }
}

def load_pattern_map():
    # load dictionary file into memory
    for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'verb_dict.txt')):
        pattern, tokens = line.strip().split('\t')
        group = '1'
        if len(pattern) > 2:
            # take group from pattern
            # e.g. v5k-s -> k
            group = pattern[2]
        tokens = tokens.split()
        # take stem from both tokens and reading
        stem = tokens[0].split('/')[0]
        stem_r = tokens[0].split('/')[2]
        # map from (stem, group) to pattern label
        # e.g. ('行', 'k') -> 'v5k-s'
        # e.g. ('行', 'u') -> 'v5u'
        pattern_map[(stem, group)] = pattern
        pattern_map[(stem_r, group)] = pattern
    return

# find verb conjugation group by looking at suffixes
def find_group(suffixes):
    if not suffixes:
        return ''
    # check direction
    suffixes_map = to_polite_suffixes_map
    if out_format == 'informal':
        suffixes_map = to_informal_suffixes_map
    # try to match suffixes map
    if suffixes in suffixes_map:
        return suffixes_map[suffixes]
    # try to match first suffix
    first = suffixes.split()[0]
    for g in groups:
        if first in groups[g]:
            return g
    # return empty string if no match
    return ''

def identify(verb):
    # handle copula
    if verb == 'で す':
        logging.debug('Identified copula from group: [' + str(1) + ']')
        logging.debug('Identified in group: [' + str(1) + ']')
        return 1
    if verb == 'だ':
        logging.debug('Identified copula from group: [' + str(0) + ']')
        logging.debug('Identified in group: [' + str(0) + ']')
        return 0
    # TODO: need more sophisticated handling of noun/na-adjectives
    # identify each verb
    for conj_map_index, map_keys in enumerate(maps):
        conj_map = maps[map_keys]['conj']
        pieces = verb.split()
        stem = pieces[0]
        logging.debug('stem = [' + stem + ']')
        suffixes = ' '.join(pieces[1:])
        logging.debug('suffixes = [' + suffixes + ']')
        # identify group based on suffixes
        group = find_group(suffixes)
        logging.debug('Verb in group: [' + group + ']')
        if (stem, group) in pattern_map:
            pattern = pattern_map[(stem, group)]
            logging.debug('Identified as pattern: [' + pattern + ']')
            if pattern in conj_map and suffixes in conj_map[pattern]:
                logging.debug('Identified in group: [' + str(conj_map_index) + ']')
                return conj_map_index
    return -1

def convert(verb):
    # handle copula
    if verb == 'で す':
        if out_format == 'informal':
            return '' # delete
        else:
            return verb # nothing changed
    if verb == 'だ':
        if out_format == 'polite':
            return 'で す' # convert da to desu
        else:
            return verb # nothing changed
    # TODO: need more sophisticated handling of noun/na-adjectives
    # convert each verb
    conj_map = to_polite_conj_map
    if out_format == 'informal':
        conj_map = to_informal_conj_map
    ans = verb
    pieces = verb.split()
    stem = pieces[0]
    logging.debug('stem = [' + stem + ']')
    suffixes = ' '.join(pieces[1:])
    logging.debug('suffixes = [' + suffixes + ']')
    # identify group based on suffixes
    group = find_group(suffixes)
    logging.debug('Verb in group: [' + group + ']')
    if (stem, group) in pattern_map:
        pattern = pattern_map[(stem, group)]
        logging.debug('Identified as pattern: [' + pattern + ']')
        if pattern in conj_map and suffixes in conj_map[pattern]:
            ans = stem + ' ' + conj_map[pattern][suffixes]
        else:
            logging.debug('Unable to convert verb conjugation for: [' + pattern + '][' + suffixes + ']')
    else:
        logging.debug('Unable to find matching pattern for: (' + stem + ', ' + group + ')')
    return ans

def search(tokens, tags):
    logging.debug('Tokens: [' + tokens + ']')
    logging.debug('Tags: [' + tags + ']')
    verb = []
    verbs = []
    tokens = tokens.strip().split()
    tags = tags.strip().split()
    verb_end = False
    for i in range(len(tokens) - 1, -1, -1):
        logging.debug('tokens[' + str(i) + ']: [' + tokens[i] + ']')
        logging.debug('tags[' + str(i) + ']: [' + tags[i] + ']')
        if tokens[i] in copula and tags[i] == auxv_tag:
            verb.append(tokens[i])
            verb_end = True
        elif tags[i] == verb_tag:
            verb.append(tokens[i])
            verb_end = True
        elif tags[i] in tails:
            verb.append(tokens[i])
        else:
            pass
        if verb_end:
            verb.reverse()
            verb = ' '.join(verb)
            verbs.append(verb)
            verb_end = False
            verb = []
    return verbs

def separate(s):
    tokens = []
    tags = []
    readings = []
    for triple in s.strip().split():
        if triple == '//补助记号/・':
            tokens.append('/')
            tags.append('补助记号')
            readings.append('・')
            continue
        pieces = triple.split('/')
        if len(pieces) != 3:
            logging.error("Can't divide this into token/tag/reading: [" + triple + "]")
            continue
        token, tag, reading = pieces
        tokens.append(token)
        tags.append(tag)
        readings.append(reading)
    tokens = ' '.join(tokens)
    tags = ' '.join(tags)
    readings = ' '.join(readings)
    return tokens, tags, readings

# We use edit distance from each formality domain as our metric for the formality of a sentence.
# Here edit distance denotes the number of verb conjugation changes required to reach a formality domain.
# A confidence estimation is provided by computing the ratio of the edit distance from different formality domains.

def process(line_raw, mk):
    sentences_raw = tokenise_sentence(line_raw)
    outcome = formality_labels["inconclusive"]
    for sentence_raw in sentences_raw:
        line = mk.getTagsToString(sentence_raw)
        result = [0 for key in maps.keys()]
        # handle escaping of space, slash, backslash
        line = line.replace('\\ /补助记号/UNK', '') # remove spaces
        line = line.replace('\\/', '/') # undo escaping of slash
        line = line.replace('\\ ', '')
        # normalize spaces
        line = re.sub('\s+', ' ', line)
        # separate line into token, tag, reading triples
        tokens, tags, readings = separate(line)
        # search for main verb in tokens
        verbs = search(tokens, tags)
        if len(verbs) > 0:
            verb = verbs[0]
            logging.debug('Found verb: [' + verb + ']')
            # identify conjugation of main verb
            ans = identify(verb)
            if ans > -1:
                result[ans] += 1
        
        if sum(result) > 0:
            if result[1] > 0:
                outcome = formality_labels["formal"]
            elif outcome != formality_labels["formal"]:
                outcome = formality_labels["informal"]

    return outcome

split_criterion = re.compile('。|！|？|\\?|\!|\.\.\.')
def tokenise_sentence(sentence):
    return filter(None, re.split(split_criterion, sentence))

logging.getLogger().setLevel(logging.CRITICAL)
# logging.getLogger().setLevel(logging.DEBUG)
mk = Mykytea.Mykytea("-deftag UNKNOWN!!")
load_pattern_map()

# ************************************ #
# Process align corpus
data_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/autoencoder_output_formality_aware.txt"), "r", encoding="utf-8")

total_accurate = 0
total = 0
for n, line in tqdm(enumerate(data_file), total=16000):
    elements = line.strip().split('||')
    s_output = elements[0]
    s_input = mk.getTagsToString(elements[1]).split()
    formality_ref = int(elements[2])
    processed = process(s_output, mk)
    # if processed != .5:
    # if len(s_input) < 16:
    if True:
        total += 1
        if processed == formality_ref:
            total_accurate += 1

print("Formality accuracy: ", total_accurate / float(total))
