# モデルの独自学習

こちらでは独自でデータを学習させてモデルを作成する方法を記載しております。

# 学習データ

本システムは1フレームごとにどのような技を出しているのかを判断するため、学習データには最低限以下の情報を1フレームごとに格納する必要があります。

* 攻め手・受け手2人の骨格座標
* 技の種類

を含む必要があります。

## データ形式・テンプレート

学習データはCSV形式です。以下から空のテンプレートがダウンロードができます。

[学習データ テンプレートCSVをダウンロード](https://samurai.bellt.me/data/data_template.csv)

テンプレートにはヘッダーと見本のデータが1行含まれています。

## 各データの説明
テンプレートファイルには以下のヘッダが含まれております。

### 映像メタ情報
座標記録時に使用した映像のメタ情報を格納できるエリアです。予測には関係しませんが、データ整理時に役立ちます。

* **SHUTTER_SPEED**：記録映像のシャッタースピード（1/200秒の場合、200と入力）
* **CLIP**：映像ファイルの整理用番号（整数なら何でもOK・利用者の環境に応じて割り振ってください）
* **TIME_MIN**：座標を記録する瞬間の映像フレームの分数
* **TIME_SEC**：座標を記録する瞬間の映像フレームの秒数
* **TIME_FLAME**：座標を記録する瞬間の映像フレーム番号

例えばシャッタースピードが1500の試合映像を50個撮影し、15個目の2分13秒10フレーム目の骨格座標を記録する場合、以下のようにメタ情報を入力します。

| SHUTTER_SPEED | CLIP | TIME_MIN | TIME_SEC | TIME_FLAME | 
| ------------- | ---- | -------- | -------- | ---------- | 
| 1500          | 15   | 2        | 13       | 10         | 

### 技コード（ATTACK_CODE）
骨格座標を記録しているその瞬間にどの技を出しているのか（または技を出していないのか）を格納するエリアです。

技コードは以下のように数字（1文字）+英字（大文字3文字）で構成されております。

* 1文字目・**技種別**：1…突き技、2…回し蹴り、3…正蹴り、4…技なし
* 2文字目・**攻撃位置**：U…上段、M…中段
* 3文字目・**攻撃方向**：L…左手足、R…右手足
* 4文字目・**有効打の如何**：S…有効、F…無効

例えば先ほどのフレームが右足で中段回し蹴り（有効打）を出している瞬間の場合、技コードを**2MLS**と入力します。

### 骨格座標
攻撃側・受け手の骨格座標を記録するエリアです。 単位はピクセル[px]です。

データ数は［XY座標（2データ）］×［節点数（17ポイント）］×［攻撃側・受け手（2人）］＝68で、以下のような構成となっております。

````
【攻撃側の骨格】
鼻（ATTACKER_NOSE_POSITION_X・ATTACKER_NOSE_POSITION_Y）
左目（ATTACKER_LEFT_EYE_POSITION_X・ATTACKER_LEFT_EYE_POSITION_Y）
右目（ATTACKER_RIGHT_EYE_POSITION_X・ATTACKER_RIGHT_EYE_POSITION_Y）
・・・
左足首（ATTACKER_LEFT_KNEE_POSITION_X・ATTACKER_LEFT_KNEE_POSITION_Y）
右足首（ATTACKER_RIGHT_KNEE_POSITION_X・ATTACKER_RIGHT_KNEE_POSITION_Y）

【受け手の骨格】
鼻（DEFENDER_NOSE_POSITION_X・DEFENDER_NOSE_POSITION_Y）
左目（DEFENDER_LEFT_EYE_POSITION_X・DEFENDER_LEFT_EYE_POSITION_Y）
右目（DEFENDER_RIGHT_EYE_POSITION_X・DEFENDER_RIGHT_EYE_POSITION_Y）
・・・
左足首（DEFENDER_LEFT_KNEE_POSITION_X・DEFENDER_LEFT_KNEE_POSITION_Y）
右足首（DEFENDER_RIGHT_KNEE_POSITION_X・DEFENDER_RIGHT_KNEE_POSITION_Y）
````

ヘッダー名に対する節点の割り当ては以下の通りです。

| ヘッダー名     | 節点の名称 | 
| -------------- | ---------- | 
| NOSE           | 鼻         | 
| LEFT_EYE       | 左目       | 
| RIGHT_EYE      | 右目       | 
| LEFT_EAR       | 左耳       | 
| RIGHT_EAR      | 右耳       | 
| LEFT_SHOULDER  | 左肩       | 
| RIGHT_SHOULDER | 右肩       | 
| LEFT_ELBOW     | 左ひじ     | 
| RIGHT_ELBOW    | 右ひじ     | 
| LEFT_WRIST     | 左手首     | 
| RIGHT_WRIST    | 右手首     | 
| LEFT_HIP       | 左臀部     | 
| RIGHT_HIP      | 右臀部     | 
| LEFT_KNEE      | 左ひざ     | 
| RIGHT_KNEE     | 右ひざ     | 
| LEFT_ANKLE     | 左足首     | 
| RIGHT_ANKLE    | 右足首     | 

上記の表を参考に節点座標を割り当ててください。

### 骨格の精度

記録する骨格には隠れた節点や画像範囲外の節点が現れる可能性があるため、ここで座標の正確性を記録することができます。

学習データに記録できる精度は以下の3つの文字列のみです。

* **CONFIRM**：正確
* **UNCLEAR**：あいまい（節点が背中やモノ等に遮られ、正確に把握できない場合）
* **OUT_OF_RANGE**：範囲外（節点が画像の外にはみ出している場合）

精度データ数は［節点数（17ポイント）］×［攻撃側・受け手（2人）］＝34で、以下のような構成となっております。

````
【攻撃側の骨格】
鼻（ATTACKER_NOSE_STATUS）
左目（ATTACKER_LEFT_EYE_STATUS）
右目（ATTACKER_RIGHT_EYE_STATUS）
・・・
左足首（ATTACKER_LEFT_KNEE_STATUS）
右足首（ATTACKER_RIGHT_KNEE_STATUS）

【受け手の骨格】
鼻（DEFENDER_NOSE_STATUS）
左目（DEFENDER_LEFT_EYE_STATUS）
右目（DEFENDER_RIGHT_EYE_STATUS）
・・・
左足首（DEFENDER_LEFT_KNEE_STATUS）
右足首（DEFENDER_RIGHT_KNEE_STATUS）
````

## データ記録時の注意事項
* データはすべて必ず半角で記録してください。
* CSVファイルのエンコーディングはUTF-8で保存してください。

# 説明変数 変換アーキテクチャ

**（現時点では独自アーキテクチャは未実装です）**

学習データにある座標はそれだけでは予測に使用することはできないため、特定の数値（説明変数）に変換する必要があります。

SAMURAIには予め変換するための以下のアーキテクチャが搭載されております。

* M-FC v1（詳しいアルゴリズムは後日記載します）

なお上記のアーキテクチャだけでなく自身で新たな変換アーキテクチャをコーディングすることもできます。
その場合はJavaScriptに準じて数値を変換する関数をコーディングする必要があります。

# モデル作成
構築するモデルはTensorFlow jsに準じてコーディングをおこなう必要があります。詳しいコーディング方法はTensorFlow jsの公式サイトを参照ください。

またコーディング時は以下のルールをお守りください。
* 作成したモデルは最後にcompileを行い、returnで返してください。
* 入力層数はアーキテクチャ設定、出力層数は目標ラベルセレクタ設定に表示されている数字と必ず一致させてください。（層数が一致しないと学習時にエラーが発生します。）