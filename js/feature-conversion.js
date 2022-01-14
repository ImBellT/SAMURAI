/**
 * 攻撃側と受け手側を利用した説明変数を作成します
 * @param {object} points1 攻撃側の骨格データ
 * @param {object} points2 受け身側の骨格データ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数）
 */
function M_FC_V1(points1, points2) {
    return [
        {
            value: calcDistance(midShoulderPoint(points2), points1[9]).value,
            score: calcDistance(midShoulderPoint(points2), points1[9]).score,
            name: '右手と肩までの距離'
        }, {
            value: calcDistance(midShoulderPoint(points2), points1[10]).value,
            score: calcDistance(midShoulderPoint(points2), points1[10]).score,
            name: '左手と肩までの距離'
        }, {
            value: calcDistance(abdomenPoint(points2), points1[9]).value,
            score: calcDistance(abdomenPoint(points2), points1[9]).score,
            name: '右手と上半身下部までの距離'
        }, {
            value: calcDistance(abdomenPoint(points2), points1[10]).value,
            score: calcDistance(abdomenPoint(points2), points1[10]).score,
            name: '左手と上半身下部までの距離'
        }, {
            value: calcDistance(midShoulderPoint(points2), points1[15]).value,
            score: calcDistance(midShoulderPoint(points2), points1[15]).score,
            name: '右足と肩までの距離'
        }, {
            value: calcDistance(midShoulderPoint(points2), points1[16]).value,
            score: calcDistance(midShoulderPoint(points2), points1[16]).score,
            name: '左足と肩までの距離'
        }, {
            value: calcDistance(abdomenPoint(points2), points1[15]).value,
            score: calcDistance(abdomenPoint(points2), points1[15]).score,
            name: '右足と上半身下部までの距離'
        }, {
            value: calcDistance(abdomenPoint(points2), points1[16]).value,
            score: calcDistance(abdomenPoint(points2), points1[16]).score,
            name: '左足と上半身下部までの距離'
        }
    ];
}

/**
 * 攻撃側一人のみを利用した説明変数を作成します
 * @param {object} points 攻撃側の骨格データ
 * @return {object} 変換後の説明変数（（数値, 変数名, 精度） × 変数数）
 */
function S_FC_V1(points) {
    return [
        {
            value: calcDistance(points[9], points[10]).value,
            score: calcDistance(points[9], points[10]).score,
            name: '左手と右手の距離'
        }, {
            value: calcDistance(points[5], points[9]).value,
            score: calcDistance(points[5], points[9]).score,
            name: '右手と右肩までの距離'
        }, {
            value: calcDistance(points[6], points[10]).value,
            score: calcDistance(points[6], points[10]).score,
            name: '左手と左肩までの距離'
        }, {
            value: diffRadian(points[5], points[7], points[9]).value,
            score: diffRadian(points[5], points[7], points[9]).score,
            name: '左手の伸ばし具合（肩・ひじ・手首が成すラジアン角度）'
        }, {
            value: diffRadian(points[6], points[8], points[10]).value,
            score: diffRadian(points[6], points[8], points[10]).score,
            name: '右手の伸ばし具合（肩・ひじ・手首が成すラジアン角度）'
        }, {
            value: calcDistance(points[15], points[16]).value,
            score: calcDistance(points[15], points[16]).score,
            name: '左足と右足の距離'
        }, {
            value: diffHeight(points[11], points[15]).value,
            score: diffHeight(points[11], points[15]).score,
            name: '右足と右臀部の高低差（有理数）'
        }, {
            value: diffHeight(points[12], points[16]).value,
            score: diffHeight(points[12], points[16]).score,
            name: '左足と左臀部の高低差（有理数）'
        }
    ];
}

// 2点間距離の導出
function calcDistance(a, b) {
    let result = {};
    let dist_x = a.x - b.x;
    let dist_y = a.y - b.y;
    result.value = Math.sqrt(Math.pow(dist_x, 2) + Math.pow(dist_y, 2));
    result.score = (a.score + b.score) / 2;
    return result;
}

// 3点間の直線がなす角度（ラジアン）を求める
function diffRadian(a, b, c) {
    let rad1 = Math.atan((b.x - a.x) / (b.y - a.y));
    let rad2 = Math.atan((c.x - b.x) / (c.y - b.y));
    let result = {};
    result.value = Math.abs(rad2 - rad1);
    result.score = (a.score + b.score + c.score) / 2;
    return result;
}

// 2点間距離の導出
function diffHeight(a, b) {
    let result = {};
    result.value = a.y - b.y;
    result.score = (a.score + b.score) / 2;
    return result;
}

// 肩の中点を計算
function midShoulderPoint(points) {
    let result = {};
    result.x = (points[5].x + points[6].x) / 2; // 右肩・左肩のX軸中点
    result.y = (points[5].y + points[6].y) / 2; // 右肩・左肩のY軸中点
    result.score = (points[5].score + points[6].score) / 2; // 右肩・左肩の精度平均
    return result;
}

// 下腹部の重心を計算
function abdomenPoint(points) {
    let result = {};
    result.x = (points[5].x + points[6].x + (points[11].x + points[12].x) * 2) / 6; // 右肩・左肩のX軸中点
    result.y = (points[5].y + points[6].y + (points[11].y + points[12].y) * 2) / 6; // 右肩・左肩のY軸中点
    result.score = (points[5].score + points[6].score + points[11].score + points[12].score) / 4; // 右肩・左肩の精度平均
    return result;
}