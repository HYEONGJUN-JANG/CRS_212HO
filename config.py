bert_special_tokens_dict = {
    'additional_special_tokens': ['<movie>'],
}
# todo: gpt2 에 <cls> 가 왜 필요?
gpt2_special_tokens_dict = {
    'pad_token': '<pad>',
    # 'cls_token': '<cls>',
    'additional_special_tokens': ['<movie>', '<explain>'],
}

# prompt_special_tokens_dict = {
#     'additional_special_tokens': ['<movie>', '<movieend>'],
# }
