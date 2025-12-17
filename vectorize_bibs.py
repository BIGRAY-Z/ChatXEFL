#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zhangxf2@shanghaitech.edu.cn
# Date: Apr 15, 2024

import argparse
import multiprocessing
import sys
import os
import time
import pathlib
import time
from itertools import chain
from datetime import datetime
from tqdm import tqdm
from src import rag, utils
from pymilvus import (FieldSchema, CollectionSchema, DataType,
                      Collection, connections)
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import pandas as pd
from bson.objectid import ObjectId

def get_parser():
    parser = argparse.ArgumentParser(description='export bibliorgaphys into vector database',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p','--file-path',type=str,default='.',help='path of bibliography folder',metavar='')
    parser.add_argument('-m','--model',choices=['llama2','llama3','bge'],type=str,default='bge',help='embedding model to use',metavar='')
    parser.add_argument('-M','--metric',choices=['L2','IP','Cos'],type=str,default='IP',help='metric type for BGE-M3 model',metavar='')
    parser.add_argument('-vc','--collection',type=str,default='',help='collection name to store vectors',metavar='')
    parser.add_argument('-s','--size',type=int,default=2000,help='size of splitter',metavar='')
    parser.add_argument('-o','--overlap',type=int,default=200,help='overlap of splitter',metavar='')
    parser.add_argument('--new',action='store_true', help='create new collection in Milvus, only valid for bge model')
    parser.add_argument('-i','--introduction',type=str,default='',help='description of new collection. only valid when use --new',metavar='')
    parser.add_argument('--new-only',action='store_true',help='exit after creation of collectio in Milvus, only valid when use --new')
    parser.add_argument('-md','--mongo-db',type=str,default='test_doi',help='database name in mongodb',metavar='')
    parser.add_argument('-mc','--mongo-col',type=str,default='bibs',help='collection name in mongodb',metavar='')
    parser.add_argument('-mf','--filters',type=str,nargs='*',default='',help='filter to query MongoDB in the format: key=value',metavar='')
    parser.add_argument('-S','--start',type=int,default=-1,help='start index for processing',metavar='')
    parser.add_argument('-E','--end',type=int,default=-1,help='start index for processing',metavar='')
    parser.add_argument('-gl','--generate-list',action='store_true',help='generate file list from mongodb, then exit')
    parser.add_argument('-ul','--update-list',action='store_true',help='update the file list from mongodb after embedding')
    parser.add_argument('--use-cpu',action='store_true',help='Only use CPU. Default is use GPU')
    return parser 

def load_and_split_file(file_list, size, overlap):
    result = []
    #for f in tqdm(file_list, desc='processing', unit=' papers'):
    for f in file_list:
        try:
            docs = rag.load(file_name=f)
            texts = rag.split(docs, size=size, overlap=overlap)
            result += texts
        except Exception as e:
            fout = split_file_log
            with open(fout, 'a') as fout:
                fout.write(f)
                fout.write(f'\n{e}\n')
                fout.write(f'\n')
                fout.flush()
    return result

def embedding_by_llm(source, desc='', splitted=True):
    print(f'{time.strftime(time_format)}: Start to embedding documents')
    try:
        if not splitted:
            docs = rag.load(source)
            texts = rag.split(docs, size=args.size, overlap=args.overlap)
        else:
            texts = source
        rag.restore_vector(texts,connection_args=connection_args,col_name=args.collection,
                           embedding=embedding, desc=desc)
        return 'SUCCESS'
    except Exception as e:
        with open(llm_log, 'a') as f:
            f.write(f'{e}\n')
            f.write(f'\n')
            f.flush()
        return 'FAIL'

def embedding_by_bge(source, splitted=True):
    try:
        if not splitted:
            docs = rag.load(source)
            texts = rag.split(docs, size=args.size, overlap=args.overlap)
        else:
            texts = source
        contents = [x.page_content for x in texts]
        metas = [x.metadata for x in texts]
        sources = [os.path.basename(x.get('source'))[:-4] for x in metas]
        pages = [x.get('page') for x in metas]
        indexs = [x.get('start_index') for x in metas]
        dois, journals, years = get_bibtex(col, sources)
        docs = [sources, dois, journals, years, pages, indexs, contents]
        em = embedding.encode_documents(contents)
        docs.append(em['dense'])
        docs.append(em['sparse'])
        keys = ['title', 'doi', 'journal', 'year', 'page', 'start_index', 'text', 'dense_vector', 'sparse_vector']
        items = [{k:v for k, v in zip(keys, doc)} for doc in zip(*docs)]
        res = milvus_client.insert(collection_name=args.collection, data=items)
        return res['insert_count']
    except Exception as e:
        with open(bge_log, 'a') as f:
            f.write(f'{e}\n')
            f.write(f'\n')
            f.flush()
        return 0

def get_bibtex(mongo_col, titles):
    bibs = {}
    for title in titles:
        if title in bibs.keys():
            continue
        query = mongo_col.find(filter={'title':title}, projection={'doi':True, 'journal':True, 'year':True, '_id':False})
        query = list(query)
        if len(query) == 0:
            print(f'WARNING: Failed to find title {title} in database.')
            bibs[title] = {}
        else:
            bibs[title] = query[0]
    dois = [bibs[x].get('doi', '') for x in titles]
    journals = [bibs[x].get('journal', '') for x in titles]
    years = [bibs[x].get('year', '') for x in titles]
    return dois, journals, years

def create_bge_collection(client, collection_name, dim, desc):
    # create schema and add fields
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True, description=desc)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name='doi', datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name='journal', datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name='year', datatype=DataType.INT16)
    schema.add_field(field_name='page', datatype=DataType.INT16)
    schema.add_field(field_name='start_index', datatype=DataType.INT64)
    schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name='pk', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
    # prepare index parameters and add indexs
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name='dense_vector',
        index_type='HNSW',
        metric_type='IP',
        params={'M':8, 'efConstruction':64}
    )
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type='IP',
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level='Session',
    )
    time.sleep(2)
    res = client.get_load_state(collection_name=collection_name)
    return res

def create_bge_collection_by_connection(connection_args, collection_name, dim, desc):
    connections.connect(**connection_args)
    fields = [
            FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name='doi', dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name='year', dtype=DataType.INT16),
            FieldSchema(name='journal', dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name='page', dtype=DataType.INT16),
            FieldSchema(name='start_index', dtype=DataType.INT64),
            FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='pk', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='dense_vector', dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name='sparse_vector', dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
    schema = CollectionSchema(fields, description=desc, enable_dynamic_field=True)
    collection = Collection(collection_name, schema, consistency_level='Session')
    sparse_index = {'index_type':'SPARSE_INVERTED_INDEX', 'metric_type':'IP'}
    collection.create_index('sparse_vector', sparse_index)
    dense_index_params = {'M':8, 'efConstruction':64}
    dense_index = {'index_type':'HNSW', 'metric_type':'IP', 'params':dense_index_params}
    collection.create_index('dense_vector', dense_index)
    collection.load()
    #connections.disconnect(alias=connection_alias)
    return collection.describe

def generate_file_list(col, outfile, filters=''):
    expression = {'embedding':False,'failed':False}
    if filters != '':
        expression = {}
        try:
            for filter_ in filters:
                x = filter_.split('=')
                if x[1].lower() == 'false':
                    expression[x[0]] = False
                elif x[1].lower() == 'true':
                    expression[x[0]] = True
                else:
                    expression[x[0]] = x[1]
        except Exception as e:
            print(e)
            print(f'Failed to parse the filters: {filters}. Default filters will be used')
            expression = {'embedding':False,'failed':False}
    print(f'Query from MongoDB by filters: {expression}')
    query = col.find(filter=expression, projection={'title':True, 'path':True, 'dir':True})
    docs = list(query)
    fout = open(outfile, 'w')
    for doc in docs:
        #title = f"{doc['path']}/{doc['dir']}/{doc['title']}.pdf"
        title = pathlib.Path(doc['path'], doc['dir'], doc['title']).as_posix()+'.pdf'
        fout.write(f"{doc['_id']} ` {title}\n")
    fout.close()
    return len(docs)

def update_embedding_state(col, doc_id, result, n_embedding, n_fail):
    embedding, fail = False, False
    if type(result) == str:
        # embedding by LLM 
        if result == 'SUCCESS':
            embedding, fail = True, False
            n_embedding += 1
        elif result == 'FAIL':
            embedding, fail = False, True
            n_fail += 1
    elif type(result) == int:
        # embedding by BGE-M3
        if result != 0:
            embedding, fail = True, False
            n_embedding += 1
        else:
            embedding, fail = False, True
            n_fail += 1
    update ={'embedding':embedding, 'failed':fail, 'embeddingTime':datetime.now()} 
    col.update_one(filter={'_id':ObjectId(doc_id)}, update={'$set':{'embedding':embedding, 'failed':fail}})
    return n_embedding, n_fail

if __name__ == "__main__":
    parser = get_parser()
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    print('+' * 80)
    print(args)

    #file_list_name = 'bibs/new/bibs_list.csv'
    file_list_name = f'{args.file_path}/bibs_list.csv'
    mongo_client = utils.get_mongodb_client(db_name=args.mongo_db)
    db = mongo_client.get_database(args.mongo_db)
    col = db.get_collection(args.mongo_col)
    if args.generate_list:
        # Generate file list and exit
        N = generate_file_list(col, outfile=file_list_name, filters=args.filters)
        print(f'List of files have been written into file {file_list_name}')
        print(f'Total papers: {N}')
        print('Bybye!')
        mongo_client.close()
        sys.exit()

    connection_args = utils.get_milvus_connection()
    if args.model == 'bge' and args.collection == '':
        print('You must set collection name by parameter: -vc/--collection')
        sys.exit()
    milvus_uri = 'http://10.19.48.181:19530'
    
    # 建议修改为动态获取 (可选):
    # milvus_uri = f'http://{utils.MILVUS_HOST}:{utils.MILVUS_PORT}'
    
    milvus_client = MilvusClient(uri=milvus_uri, **connection_args)
    time_format = '%Y-%m-%d %H:%M:%S'
    if args.model == 'bge':
        if args.new:
            if milvus_client.has_collection(collection_name=args.collection):
                print(f'ERROR: You cannot create an existed collection: {args.collection}')
                if str(milvus_client.get_load_state(collection_name=args.collection)['state']) == 'NotLoad':
                    print(f"\033[31m WARNING: the collection {args.collection} is unloaded\033[0m")   
                milvus_client.close()
                sys.exit()
            else:
                print(f'{time.strftime(time_format)}: Creating new collection: {args.collection}')
                if args.introduction == '':
                    desc = f'BGE-M3; Hybrid vector; size={args.size}; overlap={args.overlap}'
                elif 'size' not in args.introduction and 'overlap' not in args.introduction:
                    desc = f'{args.introduction}; size={args.size}; overlap={args.overlap}'
                else:
                    desc = args.introduction
                #res = create_bge_collection(client=client, collection_name=args.collection, dim=1024, desc=desc)
                res = create_bge_collection_by_connection(connection_args=connection_args,
                                                          collection_name=args.collection,
                                                          dim=1024, desc=desc)
                print(f'{time.strftime(time_format)}: Successfully create collection: {args.collection}')
                print(res)
            if args.new_only:
                sys.exit()
        elif not milvus_client.has_collection(collection_name=args.collection):
            print(f'ERROR: Cannot find collection {args.collection}, please use parameter --new to create the collection')
            milvus_client.close()
            sys.exit()

    if not os.path.exists(file_list_name): 
        print('File list does not exist, please generate file list by parameter -gl/--generate_list first') 
        sys.exit()
    data = pd.read_csv(file_list_name, delimiter='`', names=['id', 'title'])
    file_id = data['id'].to_list()
    file_list = data['title'].to_list()
    #file_list = [x.strip().replace('zhangxf', 'zhangxf2').replace('test', 'processed') for x in file_list]
    #file_list = [x.strip().replace('/bibs/new', '/bibs/processed') for x in file_list]
    file_list = [x.strip() for x in file_list]
    file_id = [x.strip() for x in file_id]
    N = len(file_list)
    if args.start == -1:
        args.start = 0
    if args.end == -1 or args.end > N:
        args.end = N
    if args.start >= args.end:
        print('ERROR: start index should be smaller than end index')
        sys.exit()


    log_dir = f'{args.file_path}/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    bge_log, llm_log, split_file_log = (
        f'{log_dir}/bge_{args.start}_{args.end}.log', 
        f'{log_dir}/llm_{args.start}_{args.end}.log', 
        f'{log_dir}/split_file_{args.start}_{args.end}.log')

    model_name, embedding_name, embedding = None, None, None,  
    #if args.model in ['7b', '13b']:
    #    model_name = f'/data-10gb/data/llm/gguf/llama2-{args.model}-chat-Q8_0.gguf'
    #    embedding = rag.get_embedding(model_name=model_name, n_gpu_layers=1)
    #    embedding_name = 'llama'
    if args.model == 'llama2':
        model_name = f'/data-10gb/data/llm/gguf/llama2-7b-chat-Q8_0.gguf'
        embedding = rag.get_embedding(model_name=model_name, n_gpu_layers=1)
        embedding_name = 'llama2'

    if args.model == 'llama3':
        model_name = f'/data-10gb/data/llm/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
        embedding = rag.get_embedding(model_name=model_name, n_gpu_layers=1)
        embedding_name = 'llama3'

    if args.model == 'bge':
        #model_name = 'BAAI/bge-m3'
        model_name = '/data-10gb/data/llm/bge-m3'
        # normalize_embedding: default is True
        #embedding = BGEM3EmbeddingFunction(model_name=model_name, device='cuda', use_fp16=True)
        if args.use_cpu:
            embedding = BGEM3EmbeddingFunction(model_name=model_name, device='cpu')
        else:
            embedding = BGEM3EmbeddingFunction(model_name=model_name, device='cuda')
        embedding_name = 'bge'

    start_time = time.time()
    print(f'{datetime.fromtimestamp(start_time).strftime(time_format)}: Start to split and embedding documents')
    n_insert, n_split, n_exist = 0, 0, -1
    n_embedding, n_fail = 0, 0
    for ix in range(args.start, args.end):
        texts = load_and_split_file(file_list=[file_list[ix]],size=args.size, overlap=args.overlap)
        n_split += len(texts)
        if embedding_name in ['llama2', 'llama3']:
            result = embedding_by_llm(source=texts, desc=args.introduction)
            if result == 'SUCCESS':
                n_insert += n_split
        elif embedding_name == 'bge':
            if n_exist == -1:
                n_exist = milvus_client.get_collection_stats(args.collection)['row_count']
            result = embedding_by_bge(source=texts)
            n_insert += result
        n_embedding, n_fail = update_embedding_state(col=col, doc_id=file_id[ix], result=result,
                                                     n_embedding=n_embedding, n_fail=n_fail)
    end_time = time.time()
    duration = end_time - start_time
    print(f'\n{datetime.fromtimestamp(end_time).strftime(time_format)}: Finished!')
    milvus_client.close()
    n_left = -1
    if args.update_list:
        n_left = generate_file_list(col, outfile=file_list_name, filters=args.filters)
    mongo_client.close()
    if embedding_name == 'bge':
        print(f'Existed items: {n_exist}')
        print(f'stats of embedding: {n_insert}')
        print(f'Now items should be : {n_exist + n_insert}')
        print(f'''Processed: {args.end-args.start}, successful: {n_embedding}, failed: {n_fail}, duration: {duration:.2f} s,
              Ave: {duration/max(1,n_embedding):.2f} s/papers''')
        print(f'Total papers: {N}, remained papers: {n_left}') 
    for log in [bge_log, llm_log, split_file_log]:
        if os.path.exists(log):
            print(f'Please check log file: {log}')