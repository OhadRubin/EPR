import random

def load_train_dataset(dataset,size=None,listify=True):
    if size is not None:
        p = size
        data = dataset['train']
        total_size = len(data)

        rand = random.Random(x=int(p*total_size))
        index_list = list(range(total_size))
        rand.shuffle(index_list)
        x = data.select(index_list[:int(p*total_size)])

        
    else:
        x = dataset['train']
    if listify:
        return list(x)
    else:
        return x