# 倉庫シミュレーション

## 背景・目的

+ 倉庫の発注戦略を強化学習で学習できないか？を検証したい

## 問題設定

+ 工場・倉庫・顧客が存在し、`顧客 <--> 倉庫 <--> 工場`の流れで商品が流れていくことを想定

+ 商品の流れ
  + 倉庫は顧客から商品を受注する
  + 倉庫は在庫があれば、商品を顧客に納品する
  + 倉庫は在庫がなければ、工場に商品を発注する
  + 倉庫は納期までに工場から商品を受領する

+ 備考
  + 本来、商品は倉庫で加工される可能性があるが今回は加工はないものとする
  + 商品の数量は、ロットなどは考慮せず、1個単位で移動できるものとする

+ 倉庫が取れる行動
  + 発注終了 or 発注 の2つ
  + 備考
    + 顧客に対する納品は、在庫があれば自動的に行う
    + 倉庫が発注の行動を取り続けている間は時間が止まっていると考える
    + 発注終了すると1日(1単位時間)が経過すると考える
    + 顧客からの受注・工場からの受領は、発注終了のタイミングで発生する

+ シミュレーションの流れ
  + 取り扱うアイテムを決める
  + 初期状態が決まる
    + 工場数・倉庫数・顧客数
    + 工場・倉庫・顧客それぞれについて、アイテムごとの在庫数
  + 終了条件に達するまで、倉庫が行動し続ける
  + 終了条件: 365日の経過

+ 報酬について
  + お金を報酬と考える
  + 納品によって、プラスの報酬が与えられる
  + 発注によって、マイナスの報酬が与えられる(コストがかかる)
  + 倉庫の在庫量によって、マイナスの報酬が与えられる(コストがかかる)

  + TD(0)学習の場合
    + 報酬は毎単位時間に発生する
  + モンテカルロ学習の場合
    + 報酬は1回のシミュレーション終了時に発生する

## 学習の実行方法

+ 多層ネットワークで学習

```bash
python main.py train simple_multi_layer
```

+ GradientBoostingで学習

```bash
python main.py train gradient_boosting
```

+ DeepQLearningで学習

```bash
python main.py train deep_q
```
