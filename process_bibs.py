#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zhangxf2@shanghaitech.edu.cn
# Date: May, 17 2024

import os
import re
import glob
import hashlib
import json
import sys
import argparse
import shutil
import datetime
import collections
from tqdm import tqdm

import pandas
import pdf2doi
import pdf2bib
from crossref_commons import retrieval
from src.utils import get_mongodb_client

def get_parser():
    parser = argparse.ArgumentParser(description='Clean bibliographys for XFEL, generate file list',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o','--output',type=str,default='bibs_process.log',help='file name of file list',metavar='')
    parser.add_argument('-p','--file_path',type=str,required=True,help='path of bibliography folder',metavar='')
    parser.add_argument('-bp','--base_path',type=str,default='.',help='base path of bibliography folder',metavar='')
    parser.add_argument('-d','--db_name',type=str,default='test_doi',help='database name in mongodb',metavar='')
    parser.add_argument('-c','--collection',type=str,default='bibs',help='collection name in mongodb',metavar='')
    parser.add_argument('--facility', type=str, default='', help='set facility by manual', metavar='')
    parser.add_argument('--find_dup', action='store_true', help='find duplicated dois and exits')
    parser.add_argument('-bt','--bib_tool',type=str,choices=['pdf2bib','crossref'],default='pdf2bib',help='tools to get bibs',metavar='')
    parser.add_argument('--manual',action='store_true',help='input metadata of bibs by manual, ignore -bt/--bib_tool')
    parser.add_argument('--year_from_path',action='store_true',help='get year of bibs from the path, ONLY valid when use --manual')
    parser.add_argument('--manual_bibs',nargs='*',default='',help='manual metadata for bibs, format in key=value and seperater by space, only valid when use --manual',metavar='')
    return parser

def cal_file_md5(file_name):
    hash_md5 = hashlib.md5()
    try:
        with open(file_name,'rb') as f:
            for chunk in iter(lambda: f.read(8192),b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest().lower() 
    except Exception as e:
        # sometime duplicate file with different names.
        return f'MD5FAILED:{e}'

def get_bibtex_by_crossref(file_name, get_abstract=True):
    bib = {}
    try:
        bib['doi'] = pdf2doi.pdf2doi(file_name).get('identifier', '')
        x = retrieval.get_publication_as_json(bib['doi'])
        bib['title'] = clean_title(x.get('title', '')[0])
        bib['url'] = x.get('URL', '')
        bib['abstract'] = clean_abstract(x.get('abstract', ''))
        bib['journal']= x['container-title'][0]
        bib['year'] = x['issued']['date-parts'][0][0]
    except Exception as e:
        print(e)
    finally:
        return bib

def get_bibtex_by_pdf2bib(file_name, get_abstract=True):
    bib = {}
    try:
        x = pdf2bib.pdf2bib(file_name)
        bib['title'] = clean_title(x['metadata']['title']) # type: ignore
        bib['doi'] = x['metadata']['doi']
        bib['url'] = x['metadata']['url']
        bib['journal'] = x['metadata']['journal']
        bib['year'] = x['metadata']['year']
        if get_abstract:
            abstract = get_zotero_abstract(bib['doi'])
            if abstract != '':
                bib['abstract'] = abstract
            else:
                val = x['validation_info']
                val = json.loads(val)
                bib['abstract'] = clean_abstract(val.get('abstract',''))
    except Exception as e:
        print(e)
    finally:
        return bib

def get_bibtex_by_manual(file_name, manual_info):
    bib = {}
    bib['title'] = os.path.basename(file_name)[:-4]
    bib['doi'] = ''
    bib['url'] = ''
    bib['journal'] = 'Engineering Report'
    bib['year'] = 2024
    bib['abstract'] = ''
    if manual_info != '':
        metas = manual_info.strip().split()
        for meta in metas:
            m = meta.strip().split('=')
            bib[m[0]] = m[1]
    return bib

def get_zotero_abstract(doi):
    abstract = ''
    res = zotero_col.find_one({'doi':doi})
    if res:
        abstract = res['meta'].get('abstractNote', '')
    return abstract

def clean_abstract(abstract):
    res = ''
    if abstract != '':
        sos = '<jats:p>'
        eos = '</jats:p>'
        s = abstract.find(sos)
        e = abstract.find(eos)
        if s != -1 and e != -1:
            res = abstract[s+len(sos):e]
    return res

def clean_title(title):
    # remove some marking substring, e.g. '<sup>', '</sup>'
    pattern1 = r'</[^>]*>'
    pattern2 = r'<[^>]*>'
    title = re.sub(pattern1, ' ', title)
    title = re.sub(pattern2, '', title)
    title = title.replace('/', '-') # remove /
    title = title.replace('\n', ' ') # remove \n
    title = ' '.join(title.split()) # remove redundant space
    return title

def get_existed_md5(col):
    md5_list = []
    doi_list = []
    try:
        docs = col.find({}, {"md5":1, "doi":1, "_id":0})
        for doc in docs:
            md5_list.append(doc['md5'])
            doi_list.append(doc['doi'])
        #md5_list = [doc['md5'] for doc in docs]
        #doi_list = [doc['doi'].lower() for doc in docs]
    except Exception as e:
        print(e)
    return md5_list, doi_list

def move_file(file_name, Type='failed'):
    if Type == 'failed':
        dst = f'{os.path.dirname(args.file_path)}/not_process'
    elif Type == 'duplicated':
        dst = f'{os.path.dirname(args.file_path)}/duplicate'
    elif Type == 'processed':
        dst = f'{os.path.dirname(args.file_path)}/processed'
    dst = file_name.replace(args.file_path, dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(file_name, dst)

def find_duplicate_dois(col, filters={}):
    docs = col.find(filter=filters, projection={'doi':1, '_id':0})
    dois = [x['doi'] for x in docs]
    N = len(dois)
    print(f'Number of docs: {N}')
    counts = collections.Counter(dois)
    u = 0
    for k, v in counts.items():
        if v > 1:
            doc = col.find(filter={'doi':k})
            print(f'\nDuplicated doi: {k}')
            for d in doc:
                if d['dir'] != '':
                    path = f"{d['path']}/{d['dir']}/{d['title']}.pdf"
                else:
                    path = f"{d['path']}/{d['title']}.pdf"
                print(path)
        else:
            u += 1
    print(f'\nNumber of dois: {N}')
    print(f'Number of non-unique dois: {N-u}')

if __name__ == "__main__":
    parser = get_parser()
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    print('+' * 80)
    print(args)

    #if not args.file_path.startswith('/'):
    #    # convert to absolute path
    #    args.file_path = os.path.abspath(args.file_path)
    #    print(f'args.file_path: {args.file_path}')

    mongo_client = get_mongodb_client(db_name=args.db_name)
    db = mongo_client.get_database(args.db_name)
    col = db.get_collection(args.collection)
    if args.find_dup:
        find_duplicate_dois(col)
        mongo_client.close()
        print('Finished')
        sys.exit()
    all_papers = glob.glob(f'{args.file_path}/**/*.pdf', recursive=True)
    zotero_col = db.get_collection('zotero_bibs')
    print('Get MD5 list...')
    md5_list, doi_list = get_existed_md5(col)
    if md5_list == []:
        print('MD5 list is empty')

    n_dup, n_insert, n_no_title, n_no_doi, n_failed, n_no_md5 = 0,0,0,0,0,0
    pdf2bib.config.set('verbose', False)
    pdf2doi.config.set('verbose', False)
    log_dir = f'{os.path.dirname(args.file_path)}/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    fout = open(f'{log_dir}/{args.output}', 'a')
    for f in tqdm(all_papers, desc='processing', unit=' papers'):
        md5 = cal_file_md5(f)
        if md5 in md5_list:
            fout.write(f'Duplicate file: {f}\n')
            # move to duplicate 
            move_file(f, Type='duplicated')
            n_dup += 1
            continue
        if md5.startswith('MD5FAILED'):
            fout.write(f'No MD5 file: {f}\n')
            n_no_md5 += 1
            continue

        if args.manual:
            bib = get_bibtex_by_manual(f, args.manual_bibs)
            if args.year_from_path:
                try:
                    year = int(re.findall(r'\b\d{4}\b', os.path.dirname(f))[0])
                except Exception as e:
                    print(e)
                    print('Failed to find year from the path of bibs')
                    year = 0
                bib['year'] = year
        else:
            if args.bib_tool == 'pdf2bib':
                bib = get_bibtex_by_pdf2bib(f)
            elif args.bib_tool == 'crossref':
                bib = get_bibtex_by_crossref(f)

        title = bib.get('title', '')
        if title == '':
            # move to not processed
            fout.write(f'Failed file: {f}\n')
            move_file(file_name=f, Type='failed')
            n_no_title += 1
            continue

        if not args.manual:
            doi = bib.get('doi', '')
            if doi == '':
                fout.write(f'Failed file: {f}\n')
                move_file(file_name=f, Type='failed')
                n_no_doi += 1
                continue

            if doi.lower() in doi_list:
                fout.write(f'Duplicate file: {f}\n')
                # move to duplicate 
                move_file(f, Type='duplicated')
                n_dup += 1
                continue
            doi_list.append(doi.lower())

        # rename to title
        new_name = f.replace(os.path.basename(f), f'{title}.pdf')
        try:
            os.rename(f, new_name)
        except Exception as e:
            print(f'Original name: {f}')
            print(f'New name: {new_name}')
            print(e)
        data = {}
        if args.facility == '':
            x = new_name[len(args.file_path):]
            if x.startswith('/'):
                x = x[1:]
            data['facility'] = x.split('/', maxsplit=1)[0]
        else:
            data['facility'] = args.facility
        md5_list.append(md5)
        data['md5'] = md5
        data['embedding'] = False
        data['failed'] = False
        data['path'] = f'{os.path.dirname(args.file_path)}/processed'
        data['dir'] = os.path.dirname(new_name)[len(args.file_path)+1:]
        data['creationTime'] = datetime.datetime.now()
        data = {**bib, **data}
        res = col.insert_one(data)
        if res.acknowledged:
            n_insert += 1
            move_file(file_name=new_name, Type='processed')
        else:
            n_failed += 0
            move_file(file_name=new_name, Type='failed')

    fout.close()
    mongo_client.close()
    print('-' * 100)
    print(f'Number of files: {len(all_papers)}')
    print(f'Number of files duplicated: {n_dup}')
    print(f'Number of files no title: {n_no_title}')
    print(f'Number of files no doi: {n_no_doi}')
    print(f'Number of files failed: {n_failed}')
    print(f'Number of files inserted: {n_insert}')
    print(f'Number of files without MD5: {n_no_md5}')