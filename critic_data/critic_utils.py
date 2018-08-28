from torchtext import vocab, data
from torchtext.datasets import language_modeling

class MusicDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for label in ['real', 'fake']:
            for fname in glob(os.path.join(path, label, '*.txt')):
                with open(fname, 'r') as f: text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex): return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        return super().splits(
            root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
    
    
def create_fake_samples(num_sets=10, bs=64, base=""):
    for i in range(num_sets):
        print(str(i), end=" ")        
        random_choice_frequency=random.random()*.5+.5
        trunc_size=random.randint(1,3)
        prompts,results = create_generation_batch(random_choice_frequency=random_choice_frequency, trunc_size=trunc_size, bs=64)
        for j in range(bs):            
            f=open(str(CRITIC_PATH/'train/fake')+'/'+base+str(i)+'_'+str(j)+'.txt', 'w')
            f.write(results[j])
            f.close()
            
def create_real_samples(num_samples=1000, base=""):
    generation=0
    prompt_size=15000       # size in characters, not tokens
    for i in range(num_samples):
        sample=random.randint(0,len(train_prompts)-1) 
        offset=random.randint(0, len(train_prompts[sample])-prompt_size-1) 
        musical_prompt=train_prompts[sample][offset:prompt_size+offset]
        f=open(str(CRITIC_PATH/'train/real')+'/'+base+str(generation)+'.txt', 'w')
        f.write(musical_prompt)
        f.close()
        generation+=1