import argparse
import math
import os
import time
import yaml
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from AliceMind.mPLUG.models.model_vqa_mplug import MPLUG
from AliceMind.mPLUG.models.tokenization_bert import BertTokenizer

def remove_long_samples(input_ids):
    inds_to_remove = []
    for i in range(input_ids.shape[0]):
        if input_ids[i, 512].item() != 0:
            inds_to_remove.append(i)
    return inds_to_remove

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--mplug_backbone', default='base')
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
            
    batch_size = args.batch_size
    output_file_name = args.output_file

    model_path = args.model_path
    with open(args.input_file, 'r') as fp:
        data = json.load(fp)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = yaml.load(open(f'configs/vqa_mplug_{args.mplug_backbone}.yaml', 'r'), Loader=yaml.Loader)
    config["min_length"] = 1
    config["max_length"] = 50
    config["beam_size"] = 5
    config['add_ocr'] = False
    config['add_object'] = False
    config['text_encoder'] = 'bert-base-uncased'
    config['text_decoder'] = 'bert-base-uncased'

    print("Creating model")
    model = MPLUG(config=config, tokenizer=tokenizer)
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    if os.path.isfile(output_file_name):
        with open(output_file_name, 'r') as fp:
            res = json.load(fp)
            print(f'Found existing output file, resuming from sample {len(res)}, {len(data) - len(res)} samples left to reformulate')
    else:
        res = []
    data = data[len(res):]
    batch_start = 0
    batch_ind = 0
    batch_num = math.ceil(len(data)/batch_size)
    t = time.time()
    while batch_start < len(data):
        if batch_ind % 100 == 0:
            print(f'Starting batch {batch_ind} out of {batch_num}, time from prev {time.time() - t}', flush=True)
            t = time.time()
            with open(output_file_name, 'w') as fp:
                fp.write(json.dumps(res))
        batch_end = min(batch_start + batch_size, len(data))
        batch_inds = [i for i in range(batch_start, batch_end)]

        questions = [data[i]['caption'] for i in batch_inds]
        question_input = tokenizer(questions, padding='longest', return_tensors="pt").to(device)
        if question_input['input_ids'].shape[1] > 512:
            inds_to_remove = remove_long_samples(question_input['input_ids'])
            batch_inds = [i for i in batch_inds if i-batch_start not in inds_to_remove]
            questions = [data[i]['caption'] for i in batch_inds]
            question_input = tokenizer(questions, padding='longest', return_tensors="pt").to(device)

        image_paths = [data[i]['image_path'] for i in batch_inds]
        images = torch.cat([transform(Image.open(image_path).convert('RGB')).unsqueeze(0) for image_path in image_paths], dim=0).to(device, non_blocking=True)

        topk_ids, topk_probs = model(images, question_input, answer=None, train=False, k=config['k_test'])
        answers = [tokenizer.decode(topk_ids[i][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip() for i in range(len(topk_ids))]
        res += [{'image_path': data[batch_inds[i]]['image_path'], 'caption': answers[i]} for i in range(len(batch_inds))]

        batch_start = batch_end
        batch_ind += 1

    with open(output_file_name, 'w') as fp:
        fp.write(json.dumps(res))

    print('Finished!')
