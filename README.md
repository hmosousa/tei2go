# Temporal Expression Identification to Go

[![Paper](https://img.shields.io/badge/Paper-557C55)](https://dl.acm.org/doi/10.1145/3583780.3615130)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)

[![HuggingFace German](https://img.shields.io/badge/-German-informational)](https://huggingface.co/hugosousa/de_tei2go)
[![HuggingFace English](https://img.shields.io/badge/-English-informational)](https://huggingface.co/hugosousa/en_tei2go)
[![HuggingFace Spanish](https://img.shields.io/badge/-Spanish-informational)](https://huggingface.co/hugosousa/es_tei2go)
[![HuggingFace Italian](https://img.shields.io/badge/-Italian-informational)](https://huggingface.co/hugosousa/it_tei2go)
[![HuggingFace French](https://img.shields.io/badge/-French-informational)](https://huggingface.co/hugosousa/fr_tei2go)
[![HuggingFace Portuguese](https://img.shields.io/badge/-Portuguese-informational)](https://huggingface.co/hugosousa/pt_tei2go)

Temporal Expression Identification to Go (TEI2GO) is an approach for fast and effective identification of temporal
expressions.
Currently, TEI2GO has models for **six** languages:

- German
- English
- Spanish
- Italian
- French
- Portuguese

However, it can be expanded to other languages. If you intend to expand it to another language feel free to create an issue, fork the repo, and do a pull request.  

## 🤗 HuggingFace Hub

To facilitate the usage, all TEI2GO models were published on [HuggingFace Hub](https://huggingface.co/hugosousa). The code below demonstrates how one can load the French model:

On the command line, run:

```bash
pip install https://huggingface.co/hugosousa/fr_tei2go/resolve/main/fr_tei2go-any-py3-none-any.whl
```

Then the model can be loaded in two ways:

1. Using Spacy

```python
import spacy
nlp = spacy.load("fr_tei2go")

doc = nlp("Bonjour, aujourd'hui nous sommes le 17 décembre 2024.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

2. Importing as a module

``` python
import fr_tei2go
nlp = fr_tei2go.load()

doc = nlp("Bonjour, aujourd'hui nous sommes le 17 décembre 2024.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Development environment

```shell
virtualenv venv --python=python3.8
source venv/bin/activate
pip install -r requirements.txt
```

To assert that everything is working run pytest: `python -m pytest tests`

## Train

```shell
python -m src.run spacy  --data tempeval_3 ph_english --language en
```

### Download Pre-Trained Models

```shell
cd models
sh download.sh
```

### Download Resources

```shell
cd resources
sh download.sh
```

## Meta

Hugo Sousa - <hugo.o.sousa@inesctec.pt>

This framework is part of the [Text2Story](https://text2story.inesctec.pt/) project which is financed by the ERDF –
European Regional Development Fund through the North Portugal Regional Operational Programme (NORTE 2020), under the
PORTUGAL 2020 and by National Funds through the Portuguese funding agency, FCT - Fundação para a Ciência e a Tecnologia
within project PTDC/CCI-COM/31857/2017 (NORTE-01-0145-FEDER-03185)

## Cite

If you use this work, please cite the following [paper](https://dl.acm.org/doi/10.1145/3583780.3615130):

```bibtex
@inproceedings{10.1145/3583780.3615130,
    author = {Sousa, Hugo and Campos, Ricardo and Jorge, Al\'{\i}pio},
    title = {TEI2GO: A Multilingual Approach for Fast Temporal Expression Identification},
    year = {2023},
    isbn = {9798400701245},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3583780.3615130},
    doi = {10.1145/3583780.3615130},
    booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
    pages = {5401–5406},
    numpages = {6},
    keywords = {temporal expression identification, multilingual corpus, weak label},
    location = {Birmingham, United Kingdom},
    series = {CIKM '23}
}
```
