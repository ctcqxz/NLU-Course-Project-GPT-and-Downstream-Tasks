'''
Part 2: Train a mini-GPT *from scratch* (no pretrained transformer weights) to
generate text in the style of William Shakespeare.

The model architecture is the same GPT-2 you implemented in Part 1, but it is
randomly initialized and trained only on data/tinyshakespeare.txt.

We split the data 80% / 10% / 10% into train / val / test, select the model with
the best validation perplexity, report the test perplexity, and print samples.

Run two different settings to compare architectures, e.g.:
  python mini_gpt_shakespeare.py --use_gpu --n_layer 4 --n_head 4 --hidden_size 256 --lr 3e-4
  python mini_gpt_shakespeare.py --use_gpu --n_layer 6 --n_head 8 --hidden_size 512 --lr 3e-4
'''

import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from config import GPT2Config
from models.gpt2 import GPT2Model
from optimizer import AdamW


def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def get_batch(data, block_size, batch_size, device):
  """Sample a random batch of (input, target) blocks from a 1-D tensor of token ids."""
  ix = torch.randint(len(data) - block_size - 1, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
  y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix]).to(device)
  mask = torch.ones_like(x)
  return x, y, mask


def compute_logits(model, x, mask):
  hidden = model(x, mask)['last_hidden_state']
  return model.hidden_state_to_token(hidden)


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, device, n_batches=50):
  """Return the average cross-entropy loss over n_batches random blocks."""
  model.eval()
  losses = []
  for _ in range(n_batches):
    x, y, mask = get_batch(data, block_size, batch_size, device)
    logits = compute_logits(model, x, mask)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    losses.append(loss.item())
  model.train()
  return float(np.mean(losses))


@torch.no_grad()
def generate(model, tokenizer, device, prompt="\n", max_new_tokens=200, block_size=128, temperature=0.8, top_k=40):
  model.eval()
  ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
  for _ in range(max_new_tokens):
    ids_cond = ids[:, -block_size:]
    mask = torch.ones_like(ids_cond)
    logits = compute_logits(model, ids_cond, mask)[:, -1, :] / temperature
    if top_k is not None:
      v, _ = torch.topk(logits, top_k)
      logits[logits < v[:, [-1]]] = -float('inf')
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    ids = torch.cat([ids, next_id], dim=1)
  return tokenizer.decode(ids[0].tolist())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, default="downstream-tasks/data/tinyshakespeare.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--use_gpu", action='store_true')
  # Architecture / hyperparameters (change these to compare different settings).
  parser.add_argument("--n_layer", type=int, default=4)
  parser.add_argument("--n_head", type=int, default=4)
  parser.add_argument("--hidden_size", type=int, default=256)
  parser.add_argument("--block_size", type=int, default=128)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--max_steps", type=int, default=2000)
  parser.add_argument("--eval_interval", type=int, default=200)
  args = parser.parse_args()

  seed_everything(args.seed)
  device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
  print(f"Using device: {device}")

  # 1. Load and tokenize the whole corpus with the GPT-2 BPE tokenizer.
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  with open(args.data_path, 'r', encoding='utf-8') as f:
    text = f.read()
  ids = torch.tensor(tokenizer(text)['input_ids'], dtype=torch.long)
  print(f"Total tokens: {len(ids)}")

  # 2. Split 80% / 10% / 10% into train / val / test.
  n = len(ids)
  train_data = ids[:int(0.8 * n)]
  val_data = ids[int(0.8 * n):int(0.9 * n)]
  test_data = ids[int(0.9 * n):]

  # 3. Build a randomly-initialized (from-scratch) GPT-2.
  config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    hidden_size=args.hidden_size,
    num_hidden_layers=args.n_layer,
    num_attention_heads=args.n_head,
    intermediate_size=4 * args.hidden_size,
    max_position_embeddings=args.block_size,
  )
  model = GPT2Model(config).to(device)
  model.train()
  n_params = sum(p.numel() for p in model.parameters())
  print(f"Model parameters: {n_params / 1e6:.2f}M")

  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

  # 4. Training loop with validation-based model selection.
  best_val_loss = float('inf')
  best_state = None
  for step in range(1, args.max_steps + 1):
    x, y, mask = get_batch(train_data, args.block_size, args.batch_size, device)
    logits = compute_logits(model, x, mask)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % args.eval_interval == 0 or step == args.max_steps:
      val_loss = evaluate(model, val_data, args.block_size, args.batch_size, device)
      print(f"step {step:5d} | train loss {loss.item():.3f} | "
            f"val loss {val_loss:.3f} | val ppl {math.exp(val_loss):.2f}")
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

  # 5. Restore the best model and report test perplexity.
  if best_state is not None:
    model.load_state_dict(best_state)
  test_loss = evaluate(model, test_data, args.block_size, args.batch_size, device, n_batches=100)
  print(f"\nBest val ppl: {math.exp(best_val_loss):.2f} | Test loss: {test_loss:.3f} | "
        f"Test ppl: {math.exp(test_loss):.2f}")

  # 6. Save the best model and print a few generated samples.
  ckpt = f"mini_gpt_L{args.n_layer}_H{args.hidden_size}.pt"
  torch.save({'model': model.state_dict(), 'args': args}, ckpt)
  print(f"Saved best model to {ckpt}\n")

  print("===== Generated samples =====")
  for i in range(3):
    print(f"\n----- sample {i + 1} -----")
    print(generate(model, tokenizer, device, prompt="\n",
                   block_size=args.block_size, max_new_tokens=200))


if __name__ == "__main__":
  main()
