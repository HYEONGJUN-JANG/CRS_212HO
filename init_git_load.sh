#!/bin/bash
cd data/redial
#wget -O foo.html google.com
#mkdir redial
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/ETObngmhWGVGjzGdfScOSO8BqTt61E0kP_G-XD7mBpdBTg?e=MAvXT4&download=1" -O content_data.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EZsp_ysU1MVCn-K0JKKWEhkBh6oA1tdY9KLwd_WBX3f2Tw?e=TqhYRu&download=1" -O dbpedia_subkg.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EYqp3KZYdJlAmfJsUZqBBt0BnWTfIIUm_u6y09NqtV0I0w?e=Hkw3Ro&download=1" -O entity2id.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EUHR6G5oiUJFs_kYeYzBTvwBXG5hMVlzMSffD_kv2xoXzg?e=kHIa2o&download=1" -O movie_ids.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/ES2gE4R0zmhMu_JCGGS1nJ0BluqYqe_qxs5HP45jaN4UBA?e=oqzm5H&download=1" -O movie2name.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EbMTUMPzG2FJr1nf3KRewxEBUEpY_o1COyOT93dVOybpJA?e=0CfNM5&download=1" -O test_data.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EfdK80BhxchDt0ey7NlwOAwBPP_qUPI3L6TtGJo17r4MNQ?e=Vebng0&download=1" -O train_data.json
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/EbZEu44RZOVLvmv_5OY__N0B2FFhLa29yZoEeZiH1M1tWg?e=3EySVT&download=1" -O valid_data.json


cd ../../saved_model
wget --no-check-certificate "https://hyu-my.sharepoint.com/:u:/g/personal/hirooms2_hanyang_ac_kr/Eawf83T57AlNh9ISAMA73PsB4MDJu6yb_nXNFMfHe8wfrw?e=E6ix9Y&download=1" -O model.pt
