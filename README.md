# BaselinesNeg RLBOA

BaselinesNeg は、3者間の stacked alternating offers protocol (SAOP) 上で
RLBOA を学習・評価するための実験コードです。

RLBOA の本体は、BOA の構成を使って bidding strategy を強化学習で学習します。
現在の実装では、RL エージェント、相手1、相手2の3者を固定順で SAOP に参加させます。
先攻後攻や RL エージェント位置の切り替えは実験軸にしていません。

```text
RLBOA agent
opponent0
opponent1
```

## ディレクトリ構成

```text
.
|-- train.py                  # 学習の実行入口
|-- test_negotiator.py        # 評価の実行入口
|-- envs/
|   |-- env.py                # RLBOAEnv
|   |-- rl_negotiator.py      # RLBOA negotiator / test negotiator
|   |-- observer.py           # RLBOA observation
|   `-- domain_loader.py      # GENIUS domain / utility loader
|-- sao/                      # SAOP mechanism と baseline negotiator
|-- domain/                   # GENIUS 形式の domain / utility XML
|-- tools/                    # 補助スクリプト
`-- results/                  # 学習済み checkpoint と評価結果
```

## Docker

推奨は Docker 実行です。既存の開発イメージを使う場合:

```bash
docker run --rm -it \
  --user $(id -u):$(id -g) \
  --entrypoint /bin/bash \
  -v "$PWD":/work \
  -w /work \
  -e MPLCONFIGDIR=/tmp/mplconfig \
  mipn_negotiator_cpy:dev
```

イメージを作り直す場合:

```bash
docker build -t mipn_negotiator_cpy:dev .
```

`results/` が root 所有になった場合は、次で戻せます。

```bash
docker run --rm --entrypoint /bin/bash \
  -v "$PWD":/work -w /work \
  mipn_negotiator_cpy:dev \
  -lc "chown -R $(id -u):$(id -g) results"
```

## 利用できる名前

`train.py` で学習対象として選べるドメイン:

```text
Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```

`test_negotiator.py` で評価対象として読めるドメイン:

```text
Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
Coffee Camera Lunch SmartPhone Kitchen Travel party
```

`planes` も `domain/` にありますが、2者用 utility しかないため、3者間 RLBOA の
選択肢には入れていません。

`train.py` で学習相手として選べる negotiator:

```text
Boulware Linear Conceder Atlas3
```

`test_negotiator.py` で評価相手として選べる negotiator:

```text
Boulware Linear Conceder TitForTat1 TitForTat2 AgentK HardHeaded Atlas3 AgentGG
```

## 学習方法

PHRI_Negotiator と同じ出力形式に寄せています。

```bash
python3 train.py -a Boulware Conceder -i Laptop
```

短く動作確認する場合:

```bash
python3 train.py \
  -a Boulware Conceder \
  -i Laptop \
  -t 8 \
  -n 1 \
  --eval-episodes 1 \
  --no-noise
```

`-a Boulware` のように相手を1つだけ指定すると、`Boulware-Boulware`
として学習します。

複数ドメイン・複数相手を指定する例:

```bash
python3 train.py \
  -a Boulware Conceder Linear Atlas3 \
  -i Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```

主な学習オプション:

```text
-a, --agents           交渉相手名。1つだけ指定すると self-pair になる
-i, --issue            ドメイン名。複数指定可能
-sp, --save_path       出力先 root。デフォルトは ./results/
-t, --timesteps        学習 timesteps。デフォルトは 100000
-n, --n-envs           並列環境数。デフォルトは 4
--eval-episodes        学習後評価の episode 数。デフォルトは 100
--n-actions            RLBOA の utility bin 数。デフォルトは 10
--no-noise             baseline negotiator の noise を無効化
--processes            複数ジョブ実行時のプロセス数
```

## 一括学習

7ドメイン × 4エージェントの重複あり組み合わせ、合計70実験を実行する
スクリプトがあります。

```bash
./run_train_all.sh
```

対象ドメイン:

```text
Laptop ItexvsCypress IS_BT_Acquisition Grocery thompson Car EnergySmall_A
```

対象エージェント:

```text
Boulware Linear Conceder Atlas3
```

各ドメインについて、次のような重複あり組み合わせを実行します。

```text
Boulware-Boulware
Boulware-Linear
Boulware-Conceder
Boulware-Atlas3
Linear-Linear
Linear-Conceder
Linear-Atlas3
Conceder-Conceder
Conceder-Atlas3
Atlas3-Atlas3
```

実行前にコマンドだけ確認する場合:

```bash
DRY_RUN=1 ./run_train_all.sh
```

短い動作確認として全70件を軽く流す場合:

```bash
TIMESTEPS=8 N_ENVS=1 EVAL_EPISODES=1 NO_NOISE=1 ./run_train_all.sh
```

設定は環境変数で変更できます。

```text
TIMESTEPS       デフォルト 100000
N_ENVS          デフォルト 4
EVAL_EPISODES   デフォルト 100
N_ACTIONS       デフォルト 10
SAVE_ROOT       指定時はこの配下に domain/pair/timestamp ごとの結果を出す
NO_NOISE        1 なら --no-noise を付ける
DRY_RUN         1 なら実行せずコマンドだけ表示
```

## 学習出力

出力先はデフォルトで次の形です。

```text
results/<domain>_<agents>/<timestamp>-TA/RLBOA_Negotiator/
```

例:

```text
results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/
```

中身:

```text
RLBOA_Negotiator/
|-- checkpoint.zip             # Stable-Baselines3 PPO モデル
|-- checkpoint/                # TensorBoard event
|   `-- events.out.tfevents...
`-- result.csv                 # 学習後評価の summary
```

PHRI_Negotiator は独自 PPO のため `checkpoint.pt` を使いますが、
この RLBOA 実装は Stable-Baselines3 の PPO を使うため `checkpoint.zip` です。

## テスト方法

学習済みモデルのディレクトリを `-m` に指定します。

```bash
python3 test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/
```

試しに episode 数を減らす場合:

```bash
python3 test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/ \
  -e 5
```

plot を保存する場合:

```bash
python3 test_negotiator.py \
  -a Boulware Conceder \
  -i Laptop \
  -m ./results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/ \
  -p
```

旧形式の指定も残しています。

```bash
python3 test_negotiator.py \
  --load-path ./results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/ \
  --domain Laptop \
  --opponent1 Boulware \
  --opponent2 Conceder \
  -e 5
```

## テスト出力

評価結果はモデルディレクトリ配下に出力されます。

```text
<model_dir>/csv/<agent0>-<agent1>/<domain>/det=False_noise=False/*.tsv
<model_dir>/img/<agent0>-<agent1>/<domain>/det=False_noise=False/*.png
```

例:

```text
results/Laptop_Boulware-Conceder/20260627-160401-TA/RLBOA_Negotiator/
`-- csv/Boulware-Conceder/Laptop/det=False_noise=False/
    `-- Laptop-Boulware-Conceder-dF-nF.tsv
```

TSV の列:

```text
my_util  opp_util1  opp_util2  social  nash  agreement  step
```

`test_negotiator.py` の episode 数はデフォルトで 100 です。

## 実行確認例

```bash
python3 train.py -a Boulware Conceder -i Laptop -t 8 -n 1 --eval-episodes 1 --no-noise
python3 test_negotiator.py -a Boulware Conceder -i Laptop \
  -m ./results/Laptop_Boulware-Conceder/<timestamp>-TA/RLBOA_Negotiator/ \
  -e 2
```

成功すると次のような出力が出ます。

```text
saved_model:./results/.../RLBOA_Negotiator/checkpoint.zip
loaded_model:results/.../RLBOA_Negotiator/checkpoint.zip
saved_result:results/.../csv/Boulware-Conceder/Laptop/det=False_noise=False/Laptop-Boulware-Conceder-dF-nF.tsv
```

## 補足

- 交渉プロトコルは `MySAOMechanism` による SAOP です。
- 参加順は RLBOA agent, opponent0, opponent1 で固定です。
- `--rl-position` や先攻後攻の実験軸はありません。
- Gym の deprecation warning が出ることがありますが、それだけでは実行失敗ではありません。
- `results/` は `.gitignore` 対象です。
