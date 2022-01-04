// メイン

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
            // FC v1の場合（今は全部ここに収束）
            //key_S = S_FC_V1(pose[0]);
        } else {
            //key_S = S_FC_V1(pose[0].keypoints);
        }
        key_S = undefined; // 一時用
        return key_S;
    } else {
        // 対戦形式の場合
        let key_1;
        let key_2;
        if (pose[0].keypoints === undefined) {
            if (param.FC_model_version === 1) {
                // FC v1の場合
                key_1 = M_FC_V1(pose[0], pose[1]);
                key_1 = key_1.concat(S_FC_V1(pose[0]));
                key_2 = M_FC_V1(pose[1], pose[0]);
                key_2 = key_2.concat(S_FC_V1(pose[1]));
            } else {
                // v1以外（未設計）
                key_1 = undefined;
                key_2 = undefined;
            }
        } else {
            if (param.FC_model_version === 1) {
                // FC v1の場合
                key_1 = M_FC_V1(pose[0].keypoints, pose[1].keypoints);
                key_1 = key_1.concat(S_FC_V1(pose[0].keypoints));
                key_2 = M_FC_V1(pose[1].keypoints, pose[0].keypoints);
                key_2 = key_2.concat(S_FC_V1(pose[1].keypoints));
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
 * 攻撃側と受け手側を利用した説明変数を作成します
 * @param {object} points1 攻撃側の骨格データ
 * @param {object} points2 受け身側の骨格データ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数）
 */
function M_FC_V1(points1, points2) {
    let key = [];
    for (let i = 0; i <= 7; i++) {
        key[i] = {};
    }
    key[0] = CalcDistance(EncodeMiddleShoulder(points2), points1[9]);
    key[0].name = '左手と肩までの距離';
    key[1] = CalcDistance(EncodeMiddleShoulder(points2), points1[10]);
    key[1].name = '右手と肩までの距離';
    key[2] = CalcDistance(EncodeMiddleMind(points2), points1[9]);
    key[2].name = '左手と上半身下部までの距離';
    key[3] = CalcDistance(EncodeMiddleMind(points2), points1[10]);
    key[3].name = '左手と上半身下部までの距離';
    key[4] = CalcDistance(EncodeMiddleShoulder(points2), points1[15]);
    key[4].name = '左足と肩までの距離';
    key[5] = CalcDistance(EncodeMiddleShoulder(points2), points1[16]);
    key[5].name = '右足と肩までの距離';
    key[6] = CalcDistance(EncodeMiddleMind(points2), points1[15]);
    key[6].name = '左足と上半身下部までの距離';
    key[7] = CalcDistance(EncodeMiddleMind(points2), points1[16]);
    key[7].name = '右足と上半身下部までの距離';
    return key;
}

/**
 * 攻撃側一人のみを利用した説明変数を作成します
 * @param {object} points 攻撃側の骨格データ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数）
 */
function S_FC_V1(points) {
    let key = [];
    for (let i = 0; i <= 9; i++) {
        key[i] = {};
    }
    key[0] = CalcDistance(points[9], points[10]);
    key[0].name = '左手と右手の距離';
    key[1] = CalcDistance(points[5], points[9]);
    key[1].name = '左手と左肩までの距離';
    key[2] = CalcDistance(points[6], points[10]);
    key[2].name = '右手と右肩までの距離';
    key[3] = DifferenceRadian(points[5], points[7], points[9]);
    key[3].name = '左手の伸ばし具合（肩・ひじ・手首が成すラジアン角度）';
    key[4] = DifferenceRadian(points[6], points[8], points[10]);
    key[4].name = '右手の伸ばし具合（肩・ひじ・手首が成すラジアン角度）';
    key[5] = CalcDistance(points[15], points[16]);
    key[5].name = '左足と右足の距離';
    key[6] = CalcDistance(points[11], points[15]);
    key[6].name = '左足と左臀部までの距離';
    key[7] = CalcDistance(points[12], points[16]);
    key[7].name = '右足と右臀部までの距離';
    key[8] = DifferenceRadian(points[11], points[13], points[15]);
    key[8].name = '左足の伸ばし具合（臀部・ひざ・足首が成すラジアン角度）';
    key[9] = DifferenceRadian(points[12], points[14], points[16]);
    key[9].name = '右足の伸ばし具合（臀部・ひざ・足首が成すラジアン角度）';
    return key;
}

// 変換アルゴリズム

// 2点間距離の導出
function CalcDistance(a, b) {
    let result = {};
    let dist_x = a.x - b.x;
    let dist_y = a.y - b.y;
    result.value = Math.sqrt(Math.pow(dist_x, 2) + Math.pow(dist_y, 2));
    result.score = (a.score + b.score) / 2;
    return result;
}

// 3点間の直線がなす角度（ラジアン）を求める
function DifferenceRadian(a, b, c) {
    let rad1 = Math.atan((b.x - a.x) / (b.y - a.y));
    let rad2 = Math.atan((c.x - b.x) / (c.y - b.y));
    let result = {};
    result.value = Math.abs(rad2 - rad1);
    result.score = (a.score + b.score) / 2;
    return result;
}

// 肩の中点を計算
function EncodeMiddleShoulder(points) {
    let result = {};
    result.x = (points[5].x + points[6].x) / 2; // 右肩・左肩のX軸中点
    result.y = (points[5].y + points[6].y) / 2; // 右肩・左肩のY軸中点
    result.score = (points[5].score + points[6].score) / 2; // 右肩・左肩の精度平均
    return result;
}

// 下腹部の重心を計算
function EncodeMiddleMind(points) {
    let result = {};
    result.x = (points[5].x + points[6].x + (points[11].x + points[12].x) * 2) / 6; // 右肩・左肩のX軸中点
    result.y = (points[5].y + points[6].y + (points[11].y + points[12].y) * 2) / 6; // 右肩・左肩のY軸中点
    result.score = (points[5].score + points[6].score + points[11].score + points[12].score) / 4; // 右肩・左肩の精度平均
    return result;
}