# Appliances Energy Prediction

Source: [UCI dataset 374](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), licensed under CC BY 4.0. Citation details are in [THIRD_PARTY.md](../../THIRD_PARTY.md).

Use the root-level `prepare_data.py` for downloads and preprocessing; see the [setup instructions](../../README.md#2-download-and-preprocess-data).

The script removes the artificial random variables `rv1` and `rv2` from `energydata_complete.csv` and writes `appliances_energy.csv`. It preserves all 19,735 observations, their native 10-minute timestamps, and the remaining 26 numeric columns.

The data loader reads these 26 historical channels and selects `Appliances` as the evaluation target (`features=MS`). CAPL uses the target channel's own history to forecast appliance energy consumption.
