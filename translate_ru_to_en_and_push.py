import os
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from transformers import pipeline
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "eng_Latn"


def translate_one_shard(
    rank: int,
    world_size: int,
    dataset_name: str,
    split: str,
    out_dir: str,
    batch_size: int,
    max_length: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

    torch.set_grad_enabled(False)
    # Load split, shard it deterministically.
    ds = load_dataset(dataset_name, split=split)
    ds = ds.shard(num_shards=world_size, index=rank)

    # One pipeline per GPU (device=0 because CUDA_VISIBLE_DEVICES maps this worker's GPU to 0).
    pipe = pipeline(
        task="translation",
        model="facebook/wmt19-ru-en",
        device=0,
        dtype=torch.bfloat16,
    )

    def _translate_batch(batch):
        outs = pipe(
            batch["ru"],
            batch_size=batch_size,
            max_length=max_length,
            truncation=True,
            num_beams=2,        # We can't spend too much compute here
            early_stopping=True
        )
        return {"en": [o["translation_text"] for o in outs]}

    translated = ds.map(
        _translate_batch,
        batched=True,
        batch_size=batch_size,
        desc=f"Translating shard {rank}/{world_size} ({split})",
    )
    print('Translated')
    # Save this shard to disk; main process will concatenate and push.
    shard_path = Path(out_dir) / split / f"shard_{rank}"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    translated.save_to_disk(str(shard_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="alexantonov/chuvash_russian_parallel")
    ap.add_argument("--out_dir", default="./translated_tmp")
    ap.add_argument("--repo_id", required=True, help="e.g. username/chuvash_ru_en_parallel")
    ap.add_argument("--splits", default="train", help="Comma-separated, e.g. train,validation,test")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max_shard_size", default="2GB")
    args = ap.parse_args()

    world_size = args.gpus
    out_dir = args.out_dir
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    mp.set_start_method("spawn", force=True)
    print('Translating')
    for split in splits:
        procs = []
        for rank in range(world_size):
            p = mp.Process(
                target=translate_one_shard,
                args=(
                    rank,
                    world_size,
                    args.dataset,
                    split,
                    out_dir,
                    args.batch_size,
                    args.max_length,
                ),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"A worker crashed with exit code {p.exitcode}")
    print('Loading translated dataset')
    # Re-load and concatenate all shards for each split.
    dd = {}
    for split in splits:
        shards = []
        for rank in tqdm(range((world_size))):
            shard_path = Path(out_dir) / split / f"shard_{rank}"
            shards.append(load_from_disk(str(shard_path)))
        dd[split] = concatenate_datasets(shards)

    final = DatasetDict(dd)
    print('Pushing to hub')
    final.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size=args.max_shard_size,
    )

    print(f"Done. Pushed to: {args.repo_id}")


if __name__ == "__main__":
    main()
