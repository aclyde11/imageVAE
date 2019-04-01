import torch
from torchvision import datasets, transforms
from invert import *
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import cairosvg





class MoleLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        super(MoleLoader, self).__init__()
        self.df = df

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

    def __getitem__(self, item):
        smile = self.df.iloc[item, 0]
        image = self.make_image(smile)

        return smile, transforms.ToTensor()(image)
