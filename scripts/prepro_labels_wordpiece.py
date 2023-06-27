import json
import h5py
from transformers import PreTrainedTokenizerFast
import argparse


def get_tokenizer():
    from_ = 'wordpiece_tokenizer/wordpiece_tokenizer.json'    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=from_,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    return tokenizer

def get_vocab(tokenizer):
    ix_to_word = {str(val+1):key for key, val in tokenizer.vocab.items()}
    
    return ix_to_word

def get_input_ids(tokenizer, sents):
    input_ids = tokenizer.batch_encode_plus(sents, padding='max_length', max_length=params['max_length'], 
                            truncation=True, add_special_tokens=True)['input_ids']
    return input_ids


def get_imgs_info(imgs, tokenizer):
    all_sents = []
    label_length = []
    label_start_ix = []
    label_end_ix = []
    images = []
    counter = 1
    for img in imgs['images']:
        img_dict = {'id': img['cocoid'], 'split': img['split'], 'file_path': img['filepath'] + '/' + img['filename']}
        sents = [sent_obj['raw'] for sent_obj in img['sentences']]
        label_len = [min(params['max_length'], len(tokenizer.tokenize(sent))) for sent in sents]
        all_sents.extend(sents)
        label_length.extend(label_len)
        label_start_ix.append(counter)
        label_end_ix.append(counter + len(sents) - 1)
        images.append(img_dict)
        counter += len(sents)
    
    return images, all_sents, label_length, label_start_ix, label_end_ix


def save_discretized_captions(input_ids, label_length, label_start_ix, label_end_ix):
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=input_ids)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()
    
    print('Saved discretized captions successfully')

def save_img_info_and_vocab(images, ix_to_word):
    out = {}
    out['images'] = images
    out['ix_to_word'] = ix_to_word
    json.dump(out, open(params['output_json'], 'w'))
    
    print('Saved images information and vocab successfully')

def main():
    
    imgs = json.load(open(params['input_json'], 'r'))
    tokenizer = get_tokenizer()
    images, all_sents, label_length, label_start_ix, label_end_ix = get_imgs_info(imgs, tokenizer)
    input_ids = get_input_ids(tokenizer, all_sents)
    ix_to_word = get_vocab(tokenizer)
    
    save_discretized_captions(input_ids, label_length, label_start_ix, label_end_ix)
    save_img_info_and_vocab(images, ix_to_word)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

  # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data', help='output h5 file')
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
#     parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main()