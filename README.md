# SPOCKMIP
**S**egmentation **P**recision **O**ptimised with **C**ontinuity **K**nowledge using **M**aximum **I**ntensity **P**rojection

Official code of the paper "SPOCKMIP: Segmentation of Vessels in MRAs with Enhanced Continuity using Maximum Intensity Projection as Loss" ([https://arxiv.org/abs/2006.10802](https://arxiv.org/abs/2407.08655))

## Model Weights
The weights of the models trained during this research have been made publicly available on Huggingface, and they can be found in the collection: [[https://huggingface.co/collections/soumickmj/ds6-66d623af4a69536bcaf3c377]([https://huggingface.co/collections/soumickmj/spockmip-66d82921b61dd11022114c66](https://huggingface.co/collections/soumickmj/spockmip-66d82921b61dd11022114c66))](https://huggingface.co/collections/soumickmj/ds6-66d623af4a69536bcaf3c377). The designations "woDeform" and "wDeform" within the model names indicate that the respective model was trained without (baseline) and with (DS6) deformation-aware learning, respectively. The designations "MIP" and "mMIP" at end of the model names indicate that the respective model was trained with MIP and multi-MIP loss, respectively (SPOCKMIP methods). 

The weights can be directly be used pulling from Huggingface with the updated version of this pipeline, or the weights can be downloaded using the AutoModel class from the transformers package, saved as a checkpoint, and then the path to this saved checkpoint can be supplied to the pipeline using "-load_path" argument.

Here's an example of how to use directly use weights from huggingface:
```bash
-load_huggingface soumickmj/SMILEUHURA_SPOCKMIP_UNet3D_MIP
```
Additional parameter "-load_huggingface" must be supplied along with the other desired paramters. Technically, this paramter can also be used to supply segmentation models other than the models used in DS6. 

Here is an example of how to save the weights locally (must be saved with .pth extension) and then use it with this pipeline:
```python
from transformers import AutoModel
modelHF = AutoModel.from_pretrained("soumickmj/SMILEUHURA_SPOCKMIP_UNet3D_MIP", trust_remote_code=True)
torch.save({'state_dict': modelHF.model.state_dict()}, "/path/to/checkpoint/model.pth")
```
To run this pipeline with these downloaded weights, the path to the checkpoint must then be passed as preweights_path, as an additional parameter along with the other desired parameters:
```bash
-load_path /path/to/checkpoint/model.pth
```

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use codes from this repository, please cite the following in your publications:

> [Radhakrishna, C., Chintalapati, K. V., Kumar, S. C. H. R., Sutrave, R., Mattern, H., Speck, O., Nuernberger, A. & Chatterjee, S. (2024). SPOCKMIP: Segmentation of Vessels in MRAs with Enhanced Continuity using Maximum Intensity Projection as Loss. arXiv preprint arXiv:2407.08655.](https://arxiv.org/abs/2407.08655)

BibTeX entry:

```bibtex
@article{radhakrishna2024spockmip,
  title={SPOCKMIP: Segmentation of Vessels in MRAs with Enhanced Continuity using Maximum Intensity Projection as Loss},
  author={Radhakrishna, Chethan and Chintalapati, Karthikesh Varma and Kumar, Sri Chandana Hudukula Ram and Sutrave, Raviteja and Mattern, Hendrik and Speck, Oliver and N{\"u}rnberger, Andreas and Chatterjee, Soumick},
  journal={arXiv preprint arXiv:2407.08655},
  year={2024}
}

```
Thank you so much for your support.
