import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from utils.output_utils import prepare_output
from PIL import Image
import time


data_dir = '../data'
# code will run in gpu if available and if the flag is set to True, else it will run on cpu
use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

# code below was used to save vocab files so that they can be loaded without Vocabulary class
#ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_ingrs.pkl'), 'rb'))
#ingrs_vocab = [min(w, key=len) if not isinstance(w, str) else w for w in ingrs_vocab.idx2word.values()]
#vocab = pickle.load(open(os.path.join(data_dir, 'final_recipe1m_vocab_toks.pkl'), 'rb')).idx2word
#pickle.dump(ingrs_vocab, open('../demo/ingr_vocab.pkl', 'wb'))
#pickle.dump(vocab, open('../demo/instr_vocab.pkl', 'wb'))

ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

t = time.time()
import sys; sys.argv=['']; del sys
args = get_parser()
args.maxseqlen = 15
args.ingrs_only=False
model = get_model(args, ingr_vocab_size, instrs_vocab_size)
# Load the trained model parameters
model_path = os.path.join(data_dir, 'modelbest.ckpt')
model.load_state_dict(torch.load(model_path, map_location=map_loc))
model.to(device)
model.eval()
model.ingrs_only = False
model.recipe_only = False

transf_list_batch = []
transf_list_batch.append(transforms.ToTensor())
transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406), 
                                              (0.229, 0.224, 0.225)))
to_input_transf = transforms.Compose(transf_list_batch)

greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
numgens = len(greedy)

import requests
from io import BytesIO
import random
from collections import Counter
use_urls = False # set to true to load images from demo_urls instead of those in test_imgs folder
show_anyways = False #if True, it will show the recipe even if it's not valid
image_folder = os.path.join(data_dir, 'demo_imgs')

if not use_urls:
    demo_imgs = os.listdir(image_folder)
    random.shuffle(demo_imgs)

demo_urls = ['https://food.fnr.sndimg.com/content/dam/images/food/fullset/2013/12/9/0/FNK_Cheesecake_s4x3.jpg.rend.hgtvcom.826.620.suffix/1387411272847.jpeg',
            'https://www.196flavors.com/wp-content/uploads/2014/10/california-roll-3-FP.jpg']

def igredent(img):
    
    title = ""
    ingredents = ""
    recipes = ""
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)
    
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)
    
#     plt.imshow(image_transf)
#     plt.axis('off')
#     plt.show()
#     plt.close()
    
    num_valid = 1
#     for i in range(numgens):
    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=True, 
                               temperature=temperature, beam=-1, true_ingrs=None)

    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    recipe_ids = outputs['recipe_ids'].cpu().numpy()

    outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)

    if valid['is_valid'] or show_anyways:

#             print ('RECIPE', num_valid)
#             num_valid+=1
        #print ("greedy:", greedy[i], "beam:", beam[i])

        title = outs['title']

        for ingr in outs['ingrs']:
            ingredents = ingredents  + ingr + ','
#         text_return = text_return + '\nInstructions:'
        for recipe in outs['recipe']:
            recipes = recipes + '\n-' + recipe

#         text_return = text_return + '='*20

    else:
        pass
        text_return.append("Not a valid recipe!")
        text_return.append("Reason: ", valid['reason'])
    return title,ingredents,recipes

demo = gr.Interface(fn=igredent, inputs=[gr.Image()], outputs=["text", "text","text"],
        examples=[
        os.path.join("../data/demo_imgs/", "1.jpg"),
        os.path.join("../data/demo_imgs/", "2.jpg"),
        os.path.join("../data/demo_imgs/", "3.jpg"),
        os.path.join("../data/demo_imgs/", "4.jpg"),
    ],)

if __name__ == "__main__":
    demo.launch()