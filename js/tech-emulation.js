/* モデル学習関数 */

/**
 * 独自で作成したモデルを読み込みます
 * @return {object} 作成したモデル
 */
function MakeCustomModel() {
    let function_str = editor_a.getValue();
    function_str = "function custom_model(){\n" + function_str + "\n}";
    const custom_model_tag = document.createElement('script');
    custom_model_tag.innerHTML = function_str;
    document.body.appendChild(custom_model_tag);
    // noinspection JSUnresolvedFunction
    return custom_model();
}

/**
 * 独自モデルを学習させます
 * @param {string} csv_file CSVファイル形式の学習データ
 * @param {string} csv_file_test CSVファイル形式の評価用テストデータ
 * @return {object} データによって学習済みのモデル
 */
async function UseCustomModel(csv_file, csv_file_test) {

    const wait = ms => new Promise(resolve => setTimeout(() => resolve(), ms)); // ミリ秒でタイムアウトする関数を定義

    _statusText.main.innerHTML = "処理状況：新規モデルの作成を実行します。";
    await wait(1000);
    const model = MakeCustomModel();

    _statusText.main.innerHTML = "処理状況：新規モデルのコンパイルが完了しました。";
    await wait(1000);
    let param = {
        mean: [],
        std: [],
        res: [
            Number(document.getElementById("pointing_res_w").value),
            Number(document.getElementById("pointing_res_h").value)
        ],
        selector: [
            document.getElementById("sel-1").checked ? 1 : 0,
            document.getElementById("sel-2").checked ? 1 : 0,
            document.getElementById("sel-3").checked ? 1 : 0,
            document.getElementById("sel-4").checked ? 1 : 0
        ],
        FC_model_version: 1
    }

    _statusText.main.innerHTML = "処理状況：アップロードされたCSVファイルを読み込んでいます。";
    await wait(1000);
    const data = await ProcessCSVData(csv_file);
    const [x_pos, y] = SplitDataIndex(data);
    let x_key = FeatureConvertAll(x_pos, param);

    const data_test = await ProcessCSVData(csv_file);
    const [x_pos_test, y_test] = SplitDataIndex(data_test);
    let x_key_test = FeatureConvertAll(x_pos_test, param);

    _statusText.main.innerHTML = "処理状況：CSVファイルの処理が完了しました。読み込まれた学習データをもとに平均値・標準偏差を導出しています。";
    await wait(1000);
    [param.mean, param.std] = CalcMeanAndStd(x_key);

    /* デバッグ用 */
    if (document.getElementById("debug-mode").checked) {
        console.log("平均値↓");
        console.log(JSON.parse(JSON.stringify(param.mean)));
        console.log("標準偏差↓");
        console.log(JSON.parse(JSON.stringify(param.std)));
    }

    x_key = KeyStandardization(x_key, param);
    x_key_test = KeyStandardization(x_key_test, param);
    const x_tensor = tf.tensor2d(KeyToValue(x_key));
    const y_tensor = tf.tensor2d(LabelOneHotEncode(y, param));

    _statusText.main.innerHTML = "処理状況：平均値・標準偏差の導出が完了しました。";
    await wait(1000);

    /* デバッグ用 */
    if (document.getElementById("debug-mode").checked) {
        console.log("モデルサマリー↓");
        model.summary();
    }

    _statusText.main.innerHTML = "処理状況：新規モデルの学習準備が整いました。モデルの学習を開始します";
    await wait(1000);

    function onEpochEnd(epoch, logs) {
        _statusText.main.innerHTML = "処理状況：新規モデルを学習中…<br/>・Epoch " + epoch + "<br/>・学習データ精度（Accuracy）" + logs.acc + "<br/>・学習データ損失（Loss）" + logs.loss + "<br/>・テストデータ精度（Val Accuracy）" + logs.val_acc + "<br/>・テストデータ損失（Val Loss）" + logs.val_loss;
        if (document.getElementById("debug-mode").checked) {
            console.log(String(epoch) + '回目の結果↓');
            console.log(logs);
        }
    }

    const info = await model.fit(x_tensor, y_tensor, {
        batchSize: Number(document.getElementById("batch_size").value),
        epochs: Number(document.getElementById("epoch_num").value),
        validationData: [tf.tensor2d(KeyToValue(x_key_test)), tf.tensor2d(LabelOneHotEncode(y_test, param))],
        callbacks: {onEpochEnd}
    });

    /* デバッグ用 */
    if (document.getElementById("debug-mode").checked) {
        console.log("学習結果↓");
        console.log(JSON.parse(JSON.stringify(info)));
    }

    _statusText.main.innerHTML = "処理状況：新規モデルの学習が完了しました。ならびにモデルのダウンロードを解除しました。";
    await wait(1000);
    return [model, param, info];
}

/**
 * 学習モデルを読み込みます（binファイルとstandard.jsonファイルが同一ディレクトリにいることを必ず確認！）
 * @param {object} model_file model.jsonのパス（外部も可）
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function LoadModel(model_file) {
    const dnn_model = await tf.loadLayersModel(model_file);
    const std_file = model_file.split("/").reverse().slice(1).reverse().join("/") + "/standard.json";
    const param_str = await GetDataFromFile(std_file);
    const param = JSON.parse(param_str);
    return [dnn_model, param];
}

/**
 * 学習モデルをjson, bin, standard別個で読み込みます
 * @param {object} json モデルjsonファイルのパス
 * @param {object} weight binファイルのパス
 * @param {object} standard standardファイルのパス
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function LoadModel_Outside(json, weight, standard) {
    const dnn_model = await tf.loadLayersModel(tf.io.browserFiles([json.files[0], weight.files[0]]));
    const DownloadUrl = URL.createObjectURL(standard.files[0]);
    const param_str = await GetDataFromFile(DownloadUrl);
    const param = await JSON.parse(param_str);
    return [dnn_model, param];
}

/**
 * 骨格座標をモデルに準拠させます（リサイズ）
 * @param {object} position 骨格座標（複数可・pose-detectionに準ずる）
 * @param {canvas} canvas 予想時に使用したキャンバス
 * @param {object} param モデルのパラメータ
 * @return {object} リサイズ後の骨格座標
 */
function PosDataResize(position, canvas, param) {
    let scale = param["res"][1] / canvas.height;
    for (let i = 0; i < position.length; i++) {
        for (let j = 0; j < position[i].keypoints.length; j++) {
            position[i].keypoints[j].x *= scale;
            position[i].keypoints[j].y *= scale;
        }
    }
    return position;
}

/**
 * 学習モデルを参考に説明変数を標準化します
 * @param {object} key 標準化前の説明変数配列（FeatureConvertAll関数に準じたデータ形式が必要）
 * @param {object} param モデルのパラメータ
 * @return {object} 標準化後の説明変数
 */
function KeyStandardization(key, param) {
    let result = key;
    // 標準偏差・平均値を代入する
    if (result !== undefined) { // keyが定義されていなければ処理をおこなわない
        if (result[0][0] === undefined) { // 実測時はここを通る（推定変数が１データ分だけなので）
            for (let j = 0; j < result.length; j++) {
                result[j].value = (key[j].value - param["mean"][j]) / param["std"][j];
            }
        } else {
            for (let i = 0; i < result.length; i++) {
                for (let j = 0; j < result[i].length; j++) {
                    result[i][j].value = (key[i][j].value - param["mean"][j]) / param["std"][j];
                }
            }
        }
    }
    return result;
}

/**
 * 学習データを標準化します
 * @param {object} key 標準化前の説明変数配列（FeatureConvertAll関数に準じたデータ形式が必要）
 * @return {object} {平均, 標準偏差}
 */
function CalcMeanAndStd(key) {
    let mean = [], std = [];
    for (let i = 0; i < key[0].length; i++) { // 各説明変数ごとに標準化をおこなう
        mean[i] = 0;
        std[i] = 0;
        for (let j = 0; j < key.length; j++) { // 各データの説明変数を足していく
            mean[i] += key[j][i].value;
        }
        mean[i] /= key.length;
        for (let j = 0; j < key.length; j++) { // 各データの説明変数を足していく
            std[i] += Math.pow(key[j][i].value - mean[i], 2);
        }
        std[i] = Math.sqrt(std[i] / key.length);
    }
    return [mean, std];
}

/**
 * 読み込んだテキストファイル（CSV・JSON）を改行（\n）含めた文字列にして出力します
 * @param {string} URL CSV・JSONファイルのパス（外部でも可）
 * @return {promise} 変換後の文字列
 */
async function GetDataFromFile(URL) {
    return new Promise(resolve => {
        let req = new XMLHttpRequest(); // HTTPでファイルを読み込むためのXMLHttpRrequestオブジェクトを生成
        req.open("get", URL, true); // アクセスするファイルを指定
        req.send(null); // HTTPリクエストの発行
        req.onload = () => {
            resolve(req.responseText);
        }
    });
}

/**
 * 読み込んだテキストファイル（CSV）を（position（x, y, score）×2）（label）（meta）の3オブジェクトにして出力します
 * @param {string} URL CSVファイルのパス（外部でも可）
 * @return {object} 変換後のデータ配列
 */
async function ProcessCSVData(URL) {
    let csv_data = await GetDataFromFile(URL);
    // 文字列をデータ化
    let result = [];
    let tmp = csv_data.split('\n'); // 改行を区切り文字として行を要素とした配列を生成
    for (let i = 0; i < tmp.length; ++i) {
        result[i] = tmp[i].split(','); // 各行ごとにカンマで区切った文字列を要素とした二次元配列を生成
    }

    // データを振り分け
    let data = [];
    for (let i = 0; i < result.length - 2; i++) {
        data[i] = {position: [[], []], meta: [], label: ""}; // 2人分の骨格座標、メタ情報配列、目標ラベル文字列を確保
        for (let j = 0; j < 17; j++) {
            data[i].position[0][j] = {};
            data[i].position[1][j] = {};
            data[i].position[0][j].x = Number(result[i + 1][j * 2 + 6]);
            data[i].position[0][j].y = Number(result[i + 1][j * 2 + 7]);
            data[i].position[1][j].x = Number(result[i + 1][j * 2 + 40]);
            data[i].position[1][j].y = Number(result[i + 1][j * 2 + 41]);
            switch (result[i + 1][j + 74]) {
                case 'CONFIRM':
                    data[i].position[0][j].score = 1.0;
                    break;
                case 'UNCLEAR':
                    data[i].position[0][j].score = 0.7;
                    break;
                case 'OUT_OF_RANGE':
                    data[i].position[0][j].score = 0.5;
            }
            switch (result[i + 1][j + 91]) {
                case 'CONFIRM':
                    data[i].position[1][j].score = 1.0;
                    break;
                case 'UNCLEAR':
                    data[i].position[1][j].score = 0.7;
                    break;
                case 'OUT_OF_RANGE':
                    data[i].position[1][j].score = 0.5;
            }
        }
        for (let j = 0; j < 5; j++) {
            data[i].meta[j] = Number(result[i + 1][j]);
        }
        data[i].label = result[i + 1][5];
    }

    return data;

}

/**
 * 目標ラベルをワンホット符号化します
 * @param {object} label 符号化する目標ラベル（技コードstr × データ数の配列を代入）
 * @param {object} param 符号化する対象（1のセレクタのみ復号化対象）
 * @return {object} 変換後の目標ラベル
 */
function LabelOneHotEncode(label, param) {
    let selector_str = [[1, 2, 3, 4], ['U', 'M'], ['L', 'R'], ['S', 'F']];
    let result = [];
    if (param.selector.toString() === [1, 0, 0, 0].toString()) {
        for (let i = 0; i < label.length; i++) {
            result[i] = [];
            switch (Number(label[i].charAt(0))) {
                case selector_str[0][0]:
                    result[i] = [1, 0, 0, 0];
                    break;
                case selector_str[0][1]:
                    result[i] = [0, 1, 0, 0];
                    break;
                case selector_str[0][2]:
                    result[i] = [0, 0, 1, 0];
                    break;
                case selector_str[0][3]:
                    result[i] = [0, 0, 0, 1];
            }
        }
    } else {
        // 後で頑張れ for 未来の自分
    }
    return result;
}

/**
 * 予測結果に座標の精度を考慮して調整します
 * @param {object} key 特徴量オブジェクト
 * @param {object} result 予測結果（テンソル形式のデータに準拠）
 * @return {object} 変換後の目標ラベル（通常の配列形式）
 */
function AdjustWithScore(key, result) {
    let res = result.reshape([-1]).arraySync();
    const score_hand = (key[0].score + key[1].score + key[2].score + key[3].score) / 4;
    const score_leg = (key[4].score + key[5].score + key[6].score + key[7].score) / 4;
    res[0] *= score_hand;
    res[1] *= score_leg;
    res[2] *= score_leg;
    res[3] *= 1 - (res[0] + res[1] + res[2]);
    return res;
}

/**
 * CSVでインポートしたデータ配列を座標、目標ラベル、メタデータで切り分けます
 * @param {object} data 振り分け前のデータ配列（GetDataFromCSVから得たデータ形式が必要）
 * @param {boolean} use_position_score 切り分け時に精度を保持するか
 * @return {object} 振り分け後のデータ（position(x, y (,score) ×2), label, meta）
 */
function SplitDataIndex(data, use_position_score = true) {
    let position = [];
    let label = [];
    let meta = [];
    for (let i = 0; i < data.length; i++) {
        position[i] = [[], []];
        for (let j = 0; j < data[i].position[0].length; j++) {
            position[i][0][j] = {};
            position[i][1][j] = {};
            position[i][0][j].x = data[i].position[0][j].x;
            position[i][0][j].y = data[i].position[0][j].y;
            position[i][1][j].x = data[i].position[1][j].x;
            position[i][1][j].y = data[i].position[1][j].y;
            if (use_position_score) {
                position[i][0][j].score = data[i].position[0][j].score;
                position[i][1][j].score = data[i].position[1][j].score;
            }
        }
        for (let j = 0; j < data[i].meta.length; j++) {
            meta[i] = data[i].meta;
        }
        label[i] = data[i].label;
    }
    return [position, label, meta];
}

/**
 * 説明変数配列から変数名、精度を取り除きテンソル学習可能な変数に変換します
 * @param {object} key 振り分け前のデータ配列（GetDataFromCSVから得たデータ形式が必要）
 * @return {object} 説明変数 ×　変数数
 */
function KeyToValue(key) {
    let result = [];
    if (key !== undefined) { // keyが定義されていなければ処理をおこなわない
        if (key[0][0] === undefined) { // 実測時はここを通る（推定変数が１データ分だけなので）
            for (let i = 0; i < key.length; i++) {
                result[i] = key[i].value;
            }
        } else {
            for (let i = 0; i < key.length; i++) {
                result[i] = [];
                for (let j = 0; j < key[i].length; j++) {
                    result[i][j] = key[i][j].value;
                }
            }
        }
    }
    return result;
}

/**
 * 説明変数をHTML上に表示します
 * @param {object} key 説明変数を含んだ配列（FeatureConvertに準じた形式が必要）
 * @param {object} status HTML UI内のステータス変数
 */
function PrintParam_Key(key, status) {
    // 変換特徴量を出力
    status.master.innerHTML = "";
    if (key !== undefined) {
        for (let i = 0; i < key.length; i++) {
            status.master.innerHTML += key[i].name + ": " + ((Math.floor(key[i].value * 100)) / 100) + "　(確率" + (Math.floor(key[i].score * 10000) / 100) + "%)<br/>";
        }
    }
}