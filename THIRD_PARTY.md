# Third-party sources

## Code

- **Time-Series-Library:** the CSV data loaders and forecasting utilities are adapted from [Time-Series-Library](https://github.com/thuml/Time-Series-Library). Its MIT copyright and license notice are retained in [LICENSE](LICENSE).
- **GluonTS:** `utils/timefeatures.py` contains time-feature code derived from [GluonTS](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/time_feature/_base.py) through Time-Series-Library. It retains the Amazon copyright header and is covered by [Apache License 2.0](licenses/Apache-2.0.txt). The upstream attribution is included in [GluonTS-NOTICE](licenses/GluonTS-NOTICE); local changes are noted in the source file.

## Datasets

Dataset files are downloaded separately and remain subject to their upstream terms. Download links and preparation commands are in [README.md](README.md#2-download-and-preprocess-data).

- **ETT:** [ETDataset](https://github.com/zhouhaoyi/ETDataset). Cite Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. The upstream [dataset license](data/ETT-small/ETDataset-LICENSE) is retained.
- **Weather and Electricity:** use the benchmark CSVs from the THUML [Time-Series-Library dataset distribution](https://huggingface.co/datasets/thuml/Time-Series-Library), as linked in the main README.
- **Appliances:** Candanedo, L. (2017). *Appliances Energy Prediction*. UCI Machine Learning Repository. [DOI: 10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G). The [UCI dataset page](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) specifies CC BY 4.0. See the [preprocessing notes](data/appliances_energy/README.md) for the retained variables and forecasting target.
