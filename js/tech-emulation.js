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
 * 目標ラベルをワンホット符号化します
 * @param {object} label 符号化する目標ラベル（技コードstr × データ数の配列を代入）
 * @param {object} param 符号化する対象（1のセレクタのみ復号化対象）
 * @return {object} 変換後の目標ラベル
 */
function LabelOneHotEncode(label, param) {
    let selector_str = [[1, 2, 3, 4], ['U', 'M'], ['L', 'R'], ['S', 'F']];
    let result = [];
    if (param.selector.toString() === [1, 0, 0, 0].toString()) { // toStringを使う理由はJavaScriptでは配列の合同関係を直接検証することができないため（これがないと常にfalseを返してしまう）
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
 * モデルを基に骨格座標を説明変数に変換します
 * （1）SplitDataIndexまたはpose-detectionに基づいたデータ形式が必要
 * （2）複数学習データの一括処理はこの関数ではなくFeatureConvertAll関数を使用
 * @param {object} pose 変換する座標データ（（座標xy, 精度）× 節点数 × 骨格数）
 * @param {object} param モデルのパラメータ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数）を両方のパターンで配列出力（1番目が10、2番目が01）
 */
function FeatureConvert(pose, param) {
    if (pose.length === 1) {
        // 単一フォームの場合
        let key_S;
        if (pose[0].keypoints === undefined) { // CSVの直接インポートだとkeypointsを定義していないので飛ばす
            // FC v1の場合
            //key_S = S_FC_V1(pose[0]);
            key_S = undefined; // 一時しのぎ用
        } else {
            //key_S = S_FC_V1(pose[0].keypoints);
            key_S = undefined; // 一時しのぎ用
        }
        return key_S;
    } else {
        // 対戦形式の場合
        let key_1;
        let key_2;
        if (pose[0].keypoints === undefined) {
            if (param.FC_model_version === 1) {
                // FC v1の場合
                key_1 = M_FC_V1(pose[0], pose[1]).concat(S_FC_V1(pose[0]));
                key_2 = M_FC_V1(pose[1], pose[0]).concat(S_FC_V1(pose[1]));
            } else {
                // v1以外（未設計）
                key_1 = undefined;
                key_2 = undefined;
            }
        } else {
            if (param.FC_model_version === 1) {
                // FC v1の場合
                key_1 = M_FC_V1(pose[0].keypoints, pose[1].keypoints).concat(S_FC_V1(pose[0].keypoints));
                key_2 = M_FC_V1(pose[1].keypoints, pose[0].keypoints).concat(S_FC_V1(pose[1].keypoints));
            } else {
                // v1以外（未設計）
                key_1 = undefined;
                key_2 = undefined;
            }
        }
        return [key_1, key_2];
    }
}

/**
 * 骨格座標を一斉に説明変数に変換します
 * （1）SplitDataIndexに基づいたデータ形式が必要
 * （2）単一データの処理はこの関数ではなくFeatureConvert関数を使用
 * @param {object} pose 変換する座標データ（（座標xy, 精度）× 節点数 × 骨格数 × データ数）※必ず攻撃側を先頭に格納してください
 * @param {object} param モデルのパラメータ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数 × データ数）
 */
function FeatureConvertAll(pose, param) {
    let key = [];
    for (let i = 0; i < pose.length; i++) {
        key[i] = FeatureConvert(pose[i], param)[0]; // key_1だけ取り出す
    }
    return key;
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