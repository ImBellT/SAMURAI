// 円の描画
function drawPoint(ctx, y, x, radius, color, scale) {
    ctx.beginPath();
    ctx.arc(x / scale, y / scale, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

// セグメント描画
function drawSegment(a_pos, b_pos, color, scale, ctx) {
    ctx.beginPath(); // 描画準備開始
    ctx.setLineDash([2, 2]);
    ctx.moveTo(a_pos.x / scale, a_pos.y / scale); // 始点
    ctx.lineTo(b_pos.x / scale, b_pos.y / scale); // 終点
    ctx.lineWidth = 2; // 線幅
    ctx.strokeStyle = color; // 線の色
    ctx.stroke(); // 描画実行
}

/**
 * キャンバス上に単一骨格を描画します
 * （1）SplitDataIndexまたはpose-detectionに基づいたデータ形式が必要
 * （2）複数骨格データの描画はこの関数ではなくdrawPoses関数を使用
 * @param {object} points 描画する座標データ（（座標xy, 精度）× 節点数）
 * @param {object} ctx 描画先のコンテキスト
 * @param {object} poseParam 骨格推定の設定パラメータ
 * @param {boolean} main_pose 骨格推定に使用される骨格か否かを設定（falseに設定するとグレーアウトで描画します）
 * @param {number} scale 入力画像と指定解像度の比率（AdjustCanvasToCtxの使用を推奨）
 */
function drawPose(points, ctx, poseParam, scale, main_pose = true) {
    // セグメントペア呼び出し
    let Pair
    if (poseParam.model.value === "pose_net") {
        Pair = poseDetection.SupportedModels.PoseNet;
    } else if (poseParam.model.value === "move_net_lighting" || poseParam.model.value === "move_net_thunder") {
        Pair = poseDetection.SupportedModels.MoveNet;
    }else if(poseParam.model.value === "blaze_pose"){
        Pair = poseDetection.SupportedModels.BlazePose;
    }

    // 線の描画
    poseDetection.util.getAdjacentPairs(Pair).forEach(([i, j]) => {
        const kp1 = points[i];
        const kp2 = points[j];
        // スコアが無ければ1として登録
        const score1 = kp1.score != null ? kp1.score : 1;
        const score2 = kp2.score != null ? kp2.score : 1;
        if (score1 >= poseParam.threshold.value && score2 >= poseParam.threshold.value) {
            if(main_pose === true){
                drawSegment(kp1, kp2, "black", scale, ctx.preview);
            }else{
                drawSegment(kp1, kp2, "gray", scale, ctx.preview);
            }
        }
    });
    // 点の描画
    points.forEach((keypoint) => {
        if (keypoint.score >= poseParam.threshold.value && main_pose === true) {
            drawPoint(ctx.preview, keypoint.y, keypoint.x, 7, "blue", scale);
        } else if(main_pose === true) {
            drawPoint(ctx.preview, keypoint.y, keypoint.x, 7, "red", scale);
        }else{
            drawPoint(ctx.preview, keypoint.y, keypoint.x, 7, "gray", scale);
        }
    });
}

/**
 * キャンバス上に推定した複数骨格を一斉に描画します
 * （1）SplitDataIndexまたはpose-detectionに基づいたデータ形式が必要
 * （2）単一骨格データの描画はこの関数ではなくdrawPose関数を使用
 * @param {object} pose 描画する骨格データ（（座標xy, 精度）× 節点数 × 骨格数）
 * @param {object} ctx 描画先のコンテキスト
 * @param {object} poseParam 骨格推定の設定パラメータ
 * @param {number} scale 入力画像と指定解像度の比率（AdjustCanvasToCtxの使用を推奨）
 */
function drawPoses(pose, ctx, poseParam, scale) {
    for (let i = 0; i < pose.length; i++) {
        if(i === 0){
            pose[i].keypoints.forEach((keypoint) => {
                drawPoint(ctx.preview, keypoint.y, keypoint.x, 12, "yellow", scale);
            });
            drawPose(pose[i].keypoints, ctx, poseParam, scale, true);
        } else if(i === 1){
            pose[i].keypoints.forEach((keypoint) => {
                drawPoint(ctx.preview, keypoint.y, keypoint.x, 12, "white", scale);
            });
            drawPose(pose[i].keypoints, ctx, poseParam, scale, true);
        }else{
            drawPose(pose[i].keypoints, ctx, poseParam, scale, false);
        }
    }
}