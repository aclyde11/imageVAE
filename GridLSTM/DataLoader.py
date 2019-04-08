import torch
from torchvision import datasets, transforms
from invert import *
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import cairosvg
import numpy as np


class MoleLoader(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_len=60, num=None):
        super(MoleLoader, self).__init__()

        self.df = df
        self.vocab = list(vocab)
        self.embedding_width = max_len
        self.start_char = '!'
        self.end_char = '?'
        self.vocab.append('!')
        self.vocab.append('?')
        self.vocab.insert(0, ' ')
        self.vocab =  {k: v for v, k in enumerate(self.vocab)}
        self.charset = {k: v for v ,k in self.vocab.items()}



    def __len__(self):
        return self.df.shape[0]

    def make_image(self, mol, molSize=(256, 256), kekulize=True, mol_name=''):
        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,
                                                       scale=1)))
        image.convert('RGB')
        return Invert()(image)

    def get_vocab_len(self):
        return len(self.vocab)


    def from_one_hot_array(self, vec):
        oh = np.where(vec == 1)
        if oh[0].shape == (0,):
            return None
        return int(oh[0][0])

    def decode_smiles_from_indexes(self, vec):
        return "".join(map(lambda x: self.charset[x], vec)).strip()

    def one_hot_array(self, i, n):
        return list(map(int, [ix == i for ix in range(n)]))

    def one_hot_index(self, vec, charset):
        return list(map(lambda x : charset[x], list(vec)))

    def one_hot_encoded_fn(self, row):
        return np.array(list(map(lambda x: self.one_hot_array(x, self.vocab), self.one_hot_index(row, self.vocab))))

    def apply_t(self, x):
        x = str(x) + str(list((''.join([char * (self.embedding_width - len(x)) for char in [' ']]))))
        smi = self.one_hot_encoded_fn(x)
        return smi

    def apply_one_hot(self, ch):
        mapper = list(map(self.apply_t, ch))
        print(mapper)
        return np.array(mapper)

    def __getitem__(self, item):
        smile = self.df.iloc[item, 0]

        embedding = self.apply_one_hot(smile)
        print(embedding)
        embedding = torch.LongTensor(embedding)
        smile_len = len(str(smile))
        image = self.make_image(smile)

        return embedding, transforms.ToTensor()(image), smile_len
