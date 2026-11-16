"""Forecast-guided thermal probe placement.

Each board is a 640x640 image holding four thermal strips in time order. The script cuts a board
into its four strips and stacks them as 12 input channels, so the same sensor column lines up
across the four times. A ConvNeXt-V2 Tiny backbone from timm (masked-autoencoder pretraining, then
ImageNet fine-tuning) is fine-tuned on the training boards. Its feature map is pooled over the
vertical axis only, which leaves 40 cells along x. The probes sit at sensor columns 18, 42, 66, 90,
114 and 138, and a strip is 640 pixels wide for 160 columns, so the probes sit six cells apart. One small head with shared
weights reads the eight cells around each probe, together with a summary of all six probes (the
levels are relative to the strongest response of the board). For each probe it outputs one of the
time profiles (the three levels at times 9, 12 and 15) that occur in the training labels. The same
probe vector also feeds an ordinal output, one yes/no per level boundary and future time, used as a
second loss only. Another head reads all six probe vectors and outputs one of the 15 probe pairs.

Training boards are transformed in ways that keep or move the labels exactly: a left-right mirror
(moved by three columns to put the probes back in place, label columns and pair reversed), a shift
by one probe spacing when both edge probes read zero (label columns and pair shifted with it), a
vertical flip and a vertical roll inside each strip (the probes are x positions, so labels do not
change).

The evaluation boards come from other acquisition runs, and measured laser traces stay in the
background. To make the model depend less on the particular traces of the
training runs, each training board is combined with a randomly drawn board of the same batch,
in and around that board's long bright runs. For half the boards the brighter pixel wins; for the
other half the two intensities are summed and the result is redrawn with the colour map, which is
read from training pixels. Short bright regions away from a trace are not copied; one that touches
a trace can be, and it gets no label. The labels stay those of the base board. This only
recombines real training boards.

Seven models are trained with the same settings. Five are fitted on five seeded folds of the
training set, and each is scored on its held-out boards with the task's metric, which gives a
held-out score over all training boards. Two more are fitted on all training boards with other
seeds. The seven are averaged on the test boards. Every prediction averages four views of the
board: as it is, mirrored, and both with the strips upside down. The matrix is the most likely
profile of each probe and the pair is the most likely output of the pair head.

The run is deterministic: fixed seeds for folds, batch order and augmentation.

Usage:
    python train.py DATA_DIR OUT_CSV
    python train.py DATA_DIR OUT_CSV --arch convnextv2_atto.fcmae_ft_in1k --epochs 2 --folds 2 --full-models 0
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("data_dir", help="folder with train.csv, test.csv and images/")
parser.add_argument("out_csv", help="where to write the predictions for test.csv")
parser.add_argument("--arch", default="convnextv2_tiny.fcmae_ft_in22k_in1k", help="any timm backbone")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--folds", type=int, default=5, help="models trained on cross-validation folds")
parser.add_argument("--full-models", type=int, default=2, help="extra models trained on all boards")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

SEED = args.seed
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dict(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda")
EPOCHS = args.epochs
BATCH = args.batch
LR = args.lr
N_FOLDS = args.folds
FULL_MODELS = args.full_models
ARCH = args.arch
DATA_DIR = args.data_dir
OUT_CSV = args.out_csv

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PAIRS = [(a, b) for a in range(6) for b in range(a + 1, 6)]
PAIR_ID = {p: i for i, p in enumerate(PAIRS)}


def read_board(path):
    # 640x640 board -> four strips of 160 rows -> 12 channels (strip * 3 + rgb), 160 x 640
    a = np.asarray(Image.open(os.path.join(DATA_DIR, path)).convert("RGB"), dtype=np.uint8)
    return a.reshape(4, 160, 640, 3).transpose(0, 3, 1, 2).reshape(12, 160, 640)


def read_boards(paths):
    with ThreadPoolExecutor(NUM_WORKERS) as pool:
        return np.stack(list(pool.map(read_board, paths)))


train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_matrix = np.array([json.loads(s) for s in train.response_matrix], dtype=np.int64)
train_pair = np.array([[int(s[0]), int(s[2])] for s in train.probe_pair], dtype=np.int64)
train_pair_id = np.array([PAIR_ID[(a, b)] for a, b in train_pair], dtype=np.int64)

# The time profiles of one probe that occur in the training labels, and each probe's profile id.
PROFILES, inverse = np.unique(train_matrix.transpose(0, 2, 1).reshape(-1, 3), axis=0, return_inverse=True)
train_profile = inverse.reshape(len(train), 6)
K = len(PROFILES)
ZERO_PROFILE = int(np.where((PROFILES == 0).all(1))[0][0])
print(f"{len(train)} train boards, {len(test)} test boards, {K} profiles", flush=True)

train_boards = read_boards(train.image_path.tolist())
test_boards = read_boards(test.image_path.tolist())
MEAN = torch.tensor([0.485, 0.456, 0.406] * 4, device=DEVICE).view(1, 12, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * 4, device=DEVICE).view(1, 12, 1, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = timm.create_model(ARCH, pretrained=True, in_chans=12, features_only=True,
                                      out_indices=(-2, -1), drop_path_rate=0.1)
        c16, c32 = self.body.feature_info.channels()
        self.mix = nn.Sequential(nn.Conv2d(c16 + c32, 256, 1), nn.GELU())
        self.row = nn.Sequential(nn.Conv1d(512, 256, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(256, 256, 3, padding=2, dilation=2), nn.GELU())
        self.local = nn.Sequential(nn.Linear(256 * 8, 256), nn.GELU(), nn.Dropout(0.1))
        self.profile = nn.Sequential(nn.Linear(256 * 3, 256), nn.GELU(), nn.Linear(256, K))
        self.pair = nn.Sequential(nn.Linear(256 * 8, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 15))
        # ordinal view of the same cells: 3 times x 4 level boundaries per probe, a training loss only
        self.ordinal = nn.Sequential(nn.Linear(256 * 3, 256), nn.GELU(), nn.Linear(256, 12))

    def forward(self, x):
        f16, f32 = self.body(x)
        f = self.mix(torch.cat([f16, F.interpolate(f32, size=f16.shape[-2:], mode="nearest")], 1))
        # pool over y only: 40 cells along x at stride 16
        g = self.row(torch.cat([f.mean(2), f.amax(2)], 1))
        # probe p sits at cell 4.6 + 6p (sensor column 18 + 24p, 4 px per column, 16 px per cell):
        # the same head reads cells 6p+1 to 6p+8 for every probe
        h = torch.stack([self.local(g[:, :, 6 * p + 1:6 * p + 9].flatten(1)) for p in range(6)], 1)
        board = torch.cat([h.amax(1), h.mean(1)], 1)
        probe = torch.cat([h, board[:, None].expand(-1, 6, -1)], 2)
        return self.profile(probe), self.pair(torch.cat([h.flatten(1), board], 1)), self.ordinal(probe)


# index of the mirrored pair: probes (a, b) become (5 - b, 5 - a)
PAIR_MIRROR = torch.tensor([PAIR_ID[(5 - b, 5 - a)] for a, b in PAIRS], device=DEVICE)


# pair index after moving both probes one place right or left, -1 when a probe would leave the board
PAIR_RIGHT = torch.tensor([PAIR_ID.get((a + 1, b + 1), -1) for a, b in PAIRS], device=DEVICE)
PAIR_LEFT = torch.tensor([PAIR_ID.get((a - 1, b - 1), -1) for a, b in PAIRS], device=DEVICE)


def intensity(raw):
    # the colour map runs blue -> red -> yellow: R and G rise, B falls
    v = raw.view(len(raw), 4, 3, 160, 640)
    return (v[:, :, 0] + v[:, :, 1] + 1 - v[:, :, 2]) / 3


BRIGHT = 0.5  # intensity above which a pixel counts as bright
RUN = 97  # px: one probe spacing (96 px) plus one; a response blob is narrower than that
RUN_FILL = 0.85  # share of a 9 x RUN window that must be bright
HALO = (33, 161)  # rows, px: one feature cell (16 px) up and down, 80 px left and right of the run
FAINT = 0.2  # the halo is copied down to this intensity


def trace_mask(level):
    # A laser trace is a bright run far longer than a response blob: keep the pixels around a
    # horizontal bright run, out to its faint halo.
    bright = (level > BRIGHT).float()
    core = F.avg_pool2d(F.avg_pool2d(bright, (9, 1), stride=1, padding=(4, 0)), (1, RUN), stride=1,
                        padding=(0, RUN // 2)) > RUN_FILL
    near = F.max_pool2d(F.max_pool2d(core.float(), (HALO[0], 1), stride=1, padding=(HALO[0] // 2, 0)),
                        (1, HALO[1]), stride=1, padding=(0, HALO[1] // 2)) > 0
    return near & (level > FAINT)


def colour_map(boards):
    # Colour of each of 256 intensity steps, read from the pixels of the given training boards.
    # It lets a board be redrawn after two intensities are summed.
    raw = torch.from_numpy(boards).float().div_(255)
    colour = raw.view(len(raw), 4, 3, -1).permute(0, 1, 3, 2).reshape(-1, 3)
    step = (intensity(raw).reshape(-1) * 255).round().long()
    total = torch.zeros(256, 3).index_add_(0, step, colour)
    count = torch.zeros(256).index_add_(0, step, torch.ones(len(step)))
    table = total / count.clamp(min=1)[:, None]
    for i in range(1, 256):
        if count[i] == 0:
            table[i] = table[i - 1]
    return table.to(DEVICE)


COLOURS = colour_map(train_boards[:64])


def overlay_traces(raw, gen):
    """Lay the laser traces of a randomly drawn board of the batch over each board. The drawn
    board is mirrored or flipped at random and its four strips keep their time order. For half the
    boards the brighter pixel wins inside the trace; for the other half the trace is summed onto the
    board on the intensity scale and redrawn with the colour map, so a response under a faint trace
    stays visible. Labels stay those of the base board: they describe its responses."""
    n = len(raw)
    donor = raw[torch.randperm(n, generator=gen, device=DEVICE)]
    r = torch.rand(n, 3, generator=gen, device=DEVICE)
    donor = torch.where((r[:, 0] < 0.5).view(-1, 1, 1, 1), donor.flip(3), donor)
    donor = torch.where((r[:, 1] < 0.5).view(-1, 1, 1, 1), donor.flip(2), donor)
    level, base = intensity(donor), intensity(raw)
    mask = trace_mask(level)
    pasted = torch.where((mask & (level > base)).repeat_interleave(3, 1), donor, raw)
    # the trace counts from the mask floor, so the sum leaves no step at the mask edge
    total = (base + (level - FAINT).clamp(min=0) / (1 - FAINT)).clamp(0, 1)
    redrawn = COLOURS[(total * 255).round().long()].permute(0, 1, 4, 2, 3).reshape(raw.shape)
    summed = torch.where(mask.repeat_interleave(3, 1), redrawn, raw)
    return torch.where((r[:, 2] < 0.5).view(-1, 1, 1, 1), summed, pasted)


def augment(x, profile, pair, gen):
    n = len(x)
    # the probes are x positions, so moving a strip's content up or down changes no label:
    # one vertical roll of up to 40 rows per batch, the same for the four strips
    x = torch.roll(x, int(torch.randint(-40, 41, (1,), generator=gen, device=DEVICE).item()), 2)
    # vertical flip inside each strip
    m = torch.rand(n, generator=gen, device=DEVICE) < 0.5
    x = torch.where(m.view(-1, 1, 1, 1), x.flip(2), x)
    # shift by one probe spacing (24 columns = 96 px) when both edge probes read zero at all three times
    free = (profile[:, 0] == ZERO_PROFILE) & (profile[:, 5] == ZERO_PROFILE)
    r = torch.rand(n, generator=gen, device=DEVICE)
    right = free & (r < 0.25) & (PAIR_RIGHT[pair] >= 0)
    left = free & (r > 0.75) & (PAIR_LEFT[pair] >= 0)
    x = torch.where(right.view(-1, 1, 1, 1), torch.roll(x, 96, 3), x)
    x = torch.where(left.view(-1, 1, 1, 1), torch.roll(x, -96, 3), x)
    profile = torch.where(right.view(-1, 1), torch.roll(profile, 1, 1), profile)
    profile = torch.where(left.view(-1, 1), torch.roll(profile, -1, 1), profile)
    pair = torch.where(right, PAIR_RIGHT[pair], torch.where(left, PAIR_LEFT[pair], pair))
    # mirror: column c goes to 159 - c, so probe column 138 lands on 21: roll 3 columns (12 px) left
    m = torch.rand(n, generator=gen, device=DEVICE) < 0.5
    x = torch.where(m.view(-1, 1, 1, 1), torch.roll(x.flip(3), -12, 3), x)
    profile = torch.where(m.view(-1, 1), profile.flip(1), profile)
    return x, profile, torch.where(m, PAIR_MIRROR[pair], pair)


def to_raw(boards, idx):
    return torch.from_numpy(boards[idx]).to(DEVICE).float().div_(255)


def to_input(boards, idx):
    return ((to_raw(boards, idx) - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def predict(model, boards):
    # Four views of each board: as it is, mirrored, and both with every strip upside down.
    # Mirrored outputs are put back in probe order; a vertical flip moves no probe.
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(**AMP):
        for s in range(0, len(boards), 64):
            x = to_input(boards, np.arange(s, min(s + 64, len(boards))))
            prof_sum, pair_sum = 0.0, 0.0
            for view in (x, x.flip(2)):
                prof, pair, _ = model(view.contiguous(memory_format=torch.channels_last))
                prof2, pair2, _ = model(torch.roll(view.flip(3), -12, 3).contiguous(memory_format=torch.channels_last))
                prof_sum = prof_sum + prof.float().softmax(-1) + prof2.float().softmax(-1).flip(1)
                pair_sum = pair_sum + pair.float().softmax(-1) + pair2.float().softmax(-1)[:, PAIR_MIRROR]
            out.append(torch.cat([(prof_sum / 4).flatten(1), pair_sum / 4], 1).cpu())
    return torch.cat(out).numpy()


def with_traces(boards, seed):
    # held-out boards with the traces of other held-out boards laid over them: a harder check
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    out = []
    for s in range(0, len(boards), 128):
        raw = overlay_traces(to_raw(boards, np.arange(s, min(s + 128, len(boards)))), gen)
        out.append((raw * 255).round().byte().cpu())
    return torch.cat(out).numpy()


def decode(prob):
    prof, pair = prob[:, :6 * K].reshape(-1, 6, K), prob[:, 6 * K:]
    matrix = PROFILES[prof.argmax(-1)].transpose(0, 2, 1)
    return matrix, np.array(PAIRS)[pair.argmax(1)]


def task_score(true_matrix, pred_matrix, true_pair, pred_pair):
    tm, pm = true_matrix.reshape(len(true_matrix), -1), pred_matrix.reshape(len(pred_matrix), -1)
    hit = tm == pm
    weight = 1.0 + 0.4 * tm
    response = 0.70 * hit.all(1).mean() + 0.30 * ((weight * hit).sum(1) / weight.sum(1)).mean()
    overlap = (pred_pair[:, :, None] == true_pair[:, None, :]).any(2).sum(1) / 2.0
    plan = 0.75 * (pred_pair == true_pair).all(1).mean() + 0.25 * overlap.mean()
    return 0.62 * response + 0.38 * plan


def train_model(fit_idx, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    order = torch.Generator()
    order.manual_seed(seed)
    flips = torch.Generator(device=DEVICE)
    flips.manual_seed(seed)
    model = Net().to(DEVICE).to(memory_format=torch.channels_last)
    body = [p for n, p in model.named_parameters() if n.startswith("body.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("body.")]
    opt = torch.optim.AdamW([{"params": body, "lr": LR}, {"params": head, "lr": LR * 5}], weight_decay=0.05)
    steps_per_epoch = len(fit_idx) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, LR * 5], total_steps=EPOCHS * steps_per_epoch,
                                                pct_start=0.1)
    profile_t = torch.from_numpy(train_profile).to(DEVICE)
    pair_t = torch.from_numpy(train_pair_id).to(DEVICE)
    levels_t = torch.from_numpy(PROFILES).to(DEVICE)
    boundaries = torch.arange(4, device=DEVICE)
    for epoch in range(EPOCHS):
        model.train()
        perm = fit_idx[torch.randperm(len(fit_idx), generator=order).numpy()]
        total = 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(perm[s * BATCH:(s + 1) * BATCH])
            x = (overlay_traces(to_raw(train_boards, idx), flips) - MEAN) / STD
            x, y_profile, y_pair = augment(x, profile_t[idx], pair_t[idx], flips)
            with torch.autocast(**AMP):
                prof, pair, ordinal = model(x.contiguous(memory_format=torch.channels_last))
            # is the level of each probe and time above boundary 0, 1, 2, 3
            above = (levels_t[y_profile][..., None] > boundaries).float().flatten(2)
            loss = (F.cross_entropy(prof.float().reshape(-1, K), y_profile.reshape(-1))
                    + 0.5 * F.cross_entropy(pair.float(), y_pair)
                    + F.binary_cross_entropy_with_logits(ordinal.float(), above))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  epoch {epoch}: train loss {total / steps_per_epoch:.4f}", flush=True)
    return model


fold = np.random.RandomState(SEED).permutation(len(train)) % N_FOLDS
n_out = 6 * K + 15
test_prob = np.zeros((len(test), n_out), dtype=np.float64)
held_prob = np.zeros((len(train), n_out), dtype=np.float64)
for k in range(N_FOLDS):
    held_idx, fit_idx = np.where(fold == k)[0], np.where(fold != k)[0]
    print(f"fold {k}: fit on {len(fit_idx)} boards, hold out {len(held_idx)}", flush=True)
    model = train_model(fit_idx, SEED + k)
    held_prob[held_idx] = predict(model, train_boards[held_idx])
    hard_matrix, hard_pair = decode(predict(model, with_traces(train_boards[held_idx], SEED + k)))
    test_prob += predict(model, test_boards) / (N_FOLDS + FULL_MODELS)
    del model
    torch.cuda.empty_cache()
    held_matrix, held_pair = decode(held_prob[held_idx])
    held = task_score(train_matrix[held_idx], held_matrix, train_pair[held_idx], held_pair)
    hard = task_score(train_matrix[held_idx], hard_matrix, train_pair[held_idx], hard_pair)
    print(f"fold {k}: held-out score {held:.4f}, with extra traces laid over the held-out boards {hard:.4f}", flush=True)
held_matrix, held_pair = decode(held_prob)
print(f"held-out score over all training boards {task_score(train_matrix, held_matrix, train_pair, held_pair):.4f}",
      flush=True)

# The same recipe on all training boards: the held-out scores above are its validation.
all_idx = np.arange(len(train))
for r in range(FULL_MODELS):
    print(f"full model {r}: fit on {len(all_idx)} boards", flush=True)
    model = train_model(all_idx, SEED + 100 + r)
    test_prob += predict(model, test_boards) / (N_FOLDS + FULL_MODELS)
    del model
    torch.cuda.empty_cache()

# One row per test id, in the order of test.csv.
matrix, pair = decode(test_prob)
predictions = pd.DataFrame({
    "case_id": test.case_id,
    "response_matrix": [json.dumps(m.tolist(), separators=(",", ":")) for m in matrix],
    "probe_pair": [f"{a}>{b}" for a, b in pair],
})
assert len(predictions) == len(test) and predictions.notna().all().all()
assert predictions.case_id.is_unique and (pair[:, 0] < pair[:, 1]).all()
assert matrix.shape == (len(test), 3, 6) and matrix.min() >= 0 and matrix.max() <= 4
os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
predictions.to_csv(OUT_CSV, index=False, lineterminator="\n")
print(f"wrote {len(predictions)} rows to {OUT_CSV}", flush=True)
