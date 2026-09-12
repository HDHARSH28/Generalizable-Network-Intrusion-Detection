# Data Directory

Place your **CIC-IDS2017** CSV files here.

## Download CIC-IDS2017

1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the CSV files (GeneratedLabelledFlows)
3. Place the `.csv` files in this `data/` directory

## Expected Format

The CSV files should contain network flow features with a `Label` column.

Example files:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- etc.

## Demo Mode

If no CSV files are placed here, you can still use the system in **demo mode**:

```bash
python train.py --demo
```

⚠️ Demo mode uses synthetic data and must NOT be used for research results.
