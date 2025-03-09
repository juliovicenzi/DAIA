# Data Sources

To get nutrional information, the [Open Food Facts](https://world.openfoodfacts.org/) data is used. You can download the tsv dataset used for this application by running:

```console
wget https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz
gzip -d en.openfoodfacts.org.products.csv.gz
```

For documentation on how to use the table, check their [wiki](https://wiki.openfoodfacts.org/Reusing_Open_Food_Facts_Data#Where_is_the_data.3F).

Alternatively, you can also use their python SDK, that includes an API wrapper and a database downloader:
```
pip install openfoodfacts
```