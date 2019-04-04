from stanfordcorenlp import StanfordCoreNLP
from document import *
from ner import *
from nominal import *
from filler import *
from typing import *
import json
import pickle
import sys
from multiprocessing.dummy import Pool as ThreadPool
import time
from threading import Semaphore


LOCK = Semaphore(1)

def run_document(fname, nlp, ontology, decisionsi, out_fname=None, raw=False):
    print('processing {}'.format(fname))
    try:
        if raw:
            sents, doc = read_raw_text(fname, nlp=nlp)
        else:
            sents, doc = read_ltf_offset(fname, nlp=nlp)
        if sents is None or doc is None:
            print('skipped {}'.format(fname))
            return
    except:
        print('error: {}; skipped {}'.format(str(e), fname))
        return True

    try:    
        LOCK.acquire()
        out_doc = []
        for sid, sent in enumerate(sents):
            # nlp_dict = {}
            # try:
            #     nlp_dict['ner'] = nlp.ner(sent.get_text().encode('UTF-8'))
            # except:
            #     nlp_dict['ner'] = None
            # try:
            #     nlp_dict['parse'] = nlp.parse(sent.get_text().encode('UTF-8'))
            # except:
            #     nlp_dict['parse'] = None
            # print(sent.get_text())
            named_ents, ners, feats = extract_ner(sent)
            # print(named_ents)
            nominals = extract_nominals(sent, sent.annotation, ners)
            # print(nominals)
            fillers = extract_filler(sent, sent.annotation, ners)
            # fillers =[]
            # print(fillers)

            # for mention, feat in zip(named_ents, feats):
            #     mention['fineGrainedType'] = ontology.lookup(mention['headword'])
            #     if mention['fineGrainedType'] == 'NULL':
            #         mention['fineGrainedType'] = infer_type(feat, decisions, root=normalize_type(mention['type']))
            # for mention in nominals:
            #     mention['fineGrainedType'] = ontology.lookup(mention['headword'])

            for m_id, mention in enumerate(named_ents + nominals + fillers):
                mention['@id'] = 'LDC2018E01-opera-text-entity-mention-{}-s{}-e{}'.format(os.path.split(fname)[1], sid,  m_id)
                mention['type'] = normalize_type(mention['type'])
            out_doc.append({'docID': os.path.split(fname)[1], 'inputSentence': sent.get_original_string(), 'offset': sent.begin-1, 'namedMentions': named_ents, 'nominalMentions': nominals, 'fillerMentions': fillers})

        if not out_fname:
            out_fname = fname + '.json'
        with open(out_fname, 'w') as f:
            json.dump(out_doc, f, indent=1, sort_keys=True)

        print('processed {}'.format(fname))
    except Exception as e:
        print('error: {}; skipped {}'.format(str(e), fname))
    finally:
        LOCK.release()
        return True


def normalize_type(t):
    if t == 'GPE':
        return 'GeopoliticalEntity'
    if t == 'ORG':
        return 'Organization'
    if t == 'LOC':
        return 'Location'
    if t == 'PER':
        return 'Person'
    if t == 'FAC':
        return 'Facility'
    if t == 'WEA':
        return 'Weapon'
    if t == 'VEH':
        return 'Vehicle'
    if t == 'URL':
        return 'URL'
    if t == 'TITLE':
        return 'Title'
    if t == 'TIME':
        return 'Time'
    if t == 'NUMERICAL':
        return 'NumericalValue'
    return t

def create_nlp_pool(num_threads):
    return [StanfordCoreNLP('http://localhost', port=9000) for _ in range(num_threads)]

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    read_raw = False
    if len(sys.argv) > 3:
        read_raw = bool(sys.argv[3])
    print('Reading from {}; writing to {}'.format(input_dir, output_dir))
    ontology = OntologyType()
    decisions = ontology.load_decision_tree()
    # with StanfordCoreNLP('/home/xianyang/stanford-corenlp-full-2017-06-09/') as nlp:
    # with StanfordCoreNLP('http://localhost', port=9006) as nlp:
    with StanfordCoreNLP('./stanford-corenlp-full-2017-06-09/', memory='8g') as nlp:
        start_time = time.time()
        if read_raw:
            files = os.listdir(input_dir)
        else:
            files = filter(lambda x: x.endswith('.xml'), os.listdir(input_dir))
        # for file in files:
        #     success = run_document(os.path.join(input_dir, file), nlp, ontology, decisions, out_fname=os.path.join(output_dir, file + '.json'))
        pool = ThreadPool(processes=8)
        success = pool.map(lambda file: run_document(os.path.join(input_dir, file), nlp, ontology, decisions, out_fname=os.path.join(output_dir, file + '.json'), raw=read_raw), files)
        pool.close()
        pool.join()
        end_time = time.time()
        print('total elapsed time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
