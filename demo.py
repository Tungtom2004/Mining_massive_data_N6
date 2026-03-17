from datasets import load_dataset
ds = load_dataset(
    "leonardoblas/us_election_2024_telegram_distilled",
    split="train",
    streaming=True,
    token = ''
)

subset = []

for i, row in enumerate(ds):
    if row.get('message_id') == 6:
        print(row)
        break 

