from predict import Predictor
import json
import argparse
from tqdm import tqdm
from reformulate import reformulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--reformulation_model', default=None)
    parser.add_argument('--mplug_backbone', default='base')
    parser.add_argument('--reformulation_batch_size', type=int, default=16)
    args = parser.parse_args()

    predictor = Predictor()
    predictor.setup(args.model_path)

    with open(args.input_file, 'r') as fp:
        dataset = json.load(fp)

    print('Dataset size: ' + str(len(dataset)), flush=True)
    res = []
    for sample in tqdm(dataset, dec='Generating captions'):
        image_path = sample['image_path']
        image_id = sample['image_id']
        generated_caption = predictor.predict(image=image_path, use_beam_search=True)
        res.append({'image_id': image_id, 'caption': generated_caption})

    if args.reformulation_model is not None:
        orig_file = f'orig_{args.output_file}'
        with open(orig_file,  'w') as fp:
            json.dump(res, fp)

        print('Reformulating...', flush=True)
        reformulate(orig_file, args.output_file, args.reformulation_model, args.mplug_backbone, args.reformulation_batch_size)
    else:    
        with open(args.output_file,  'w') as fp:
            json.dump(res, fp)

    print('Finished!')
