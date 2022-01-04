/**
 * 骨格推定モデルを宣言します
 * @param {object} inputVideo 入力映像（videoタグ）
 * @param {object} poseParam 骨格推定の設定パラメータ
 * @return {model} 宣言したモデル
 */
async function CreatePoseModel(inputVideo, poseParam) {
    // 骨格モデル定義（選択肢ごとに定義）
    let netModel;
    if (poseParam.model.value === "pose_net") {
        const detectorConfig = {
            architecture: 'ResNet50',
            outputStride: Number(poseParam.stride.value),
            quantBytes: Number(poseParam.quant.value),
            inputResolution: {
                width: Number(poseParam.input_res.value) * (inputVideo.videoWidth / inputVideo.videoHeight),
                height: Number(poseParam.input_res.value)
            }
        };
        netModel = await poseDetection.createDetector(poseDetection.SupportedModels.PoseNet, detectorConfig);
    } else if (poseParam.model.value === "move_net_lighting") {
        const detectorConfig = {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING, // モデルはLightingを使用
            multiPoseMaxDimension: 512,
            enableTracking: false // トラッキング（追従）を有効化。これで一度とらえた人物を失うまで追い続ける。
        };
        netModel = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    } else if (poseParam.model.value === "move_net_thunder") {
        const detectorConfig = {
            modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER // モデルはThunderを使用
        };
        netModel = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
    } else if(poseParam.model.value === "blaze_pose"){
        const model = poseDetection.SupportedModels.BlazePose;
        const detectorConfig = {
            runtime: 'tfjs',
            enableSmoothing: true,
            modelType: 'heavy'
        };
        netModel = await poseDetection.createDetector(model, detectorConfig);
    }
    return netModel;
}

/**
 * キャンバスのサイズを入力画像（または設定解像度）に合わせてスケーリングします
 * @param {object} video 入力映像（videoタグ）
 * @param {object} canvas リセット対象のキャンバス
 * @param {object} poseParam 骨格推定の設定パラメータ
 * @return {number} 入力画像と指定解像度の比率（PoseNet以外は関係なしに1を出力）
 */
function AdjustCanvasToCtx(video, canvas, poseParam) {
    let scale = 1;
    if (Number(poseParam.input_res.value) < video.videoHeight) {
        scale = Number(poseParam.input_res.value) / video.videoHeight;
    }
    canvas.calc.height = video.videoHeight * scale;
    canvas.calc.width = video.videoWidth * scale;
    canvas.preview.height = video.videoHeight;
    canvas.preview.width = video.videoWidth;
    return scale;
}

/**
 * 技推定を行う骨格を先頭に移動します
 * @param {object} pose 骨格座標
 * @return {object} 順序変更を行った骨格座標
 */
function AdjustPoseIndex(pose){

    let re_pose = [];
    let pose_size1 = [];

    for(let i=0; i < pose.length; i++){ // 全骨格のサイズを導出
        let size1 = CalcDistance(pose[i].keypoints[5], pose[i].keypoints[11]);
        let size2 = CalcDistance(pose[i].keypoints[6], pose[i].keypoints[12]);
        pose_size1[i] = size1.value > size2.value ? size1.value : size2.value;
    }
    const pose_size2 = JSON.parse(JSON.stringify(pose_size1)); // pose_size2にソート前の配列を複製する
    pose_size1.sort(function(a,b){
        if( a > b ) return -1;
        if( a < b ) return 1;
        return 0;
    }); // サイズで配列をソート

    for(let i=0; i < pose.length; i++) {　// 大きい順のインデックスを取得し、骨格データを順々に入れる
        re_pose[i] = pose[pose_size2.indexOf(pose_size1[i])];
    }
    return re_pose;
}

/**
 * 推定座標をHTML上に表示します
 * @param {object} position 骨格座標を含んだ配列（pose-detectionに準じた形式が必要）
 * @param {object} status HTML UI内のステータス変数
 */
function PrintParam_Origin(position, status) {
    // キーポイントの配列
    let keyName = ["鼻", "左目", "右目", "左耳", "右耳", "左肩", "右肩", "左ひじ", "右ひじ", "左手首", "右手首", "左臀部", "右臀部", "左ひざ", "右ひざ", "左足首", "右足首"];
    // オリジン座標を出力
    status.origin.innerHTML = "";
    for (let i = 0; i <= 16; i++) {
        status.origin.innerHTML += keyName[i] + "： x[" + ((Math.floor(position[0].keypoints[i].x * 100)) / 100) + "] y[" + ((Math.floor(position[0].keypoints[i].y * 100)) / 100) + "] (精度: " + ((Math.floor(position[0].keypoints[i].score * 10000)) / 100) + "%)<br/>";
    }
}

/**
 * キャンバスに入力映像のスナップショットを上書きします（骨格描画のリセット用）
 * @param {object} video 入力映像（videoタグ）
 * @param {object} canvas 上書きするキャンバス（calc, previewとも必要）
 * @param {object} ctx 上書きするコンテキスト（calc, previewとも必要）
 */
function ResetAllCanvas(ctx, canvas, video) {
    ctx.calc.clearRect(0, 0, canvas.calc.width, canvas.calc.height);
    ctx.calc.drawImage(video, 0, 0, canvas.calc.width, canvas.calc.height);
    ctx.preview.clearRect(0, 0, canvas.preview.width, canvas.preview.height);
    ctx.preview.drawImage(video, 0, 0, canvas.preview.width, canvas.preview.height);
}

/**
 * 骨格推定座標を格納するためのIndexDBを作成します
 * @param {string} DB_name IndexDBの名前
 * @param {string} storeName オブジェクトストアの名前
 */
function MakePoseDB(DB_name, storeName){
    let openReq  = indexedDB.open(DB_name, 1); // 注意：バージョン番号は小数点NG

    // オブジェクトストアの作成・削除はDBの更新時しかできないので、バージョンを指定して更新
    openReq.onupgradeneeded = function(event){
        let db = event.target.result; // データベースを定義
        db.createObjectStore(storeName, {keyPath : "time_stamp"});　// time_stamp変数をキーとしてオブジェクトストアを作成
    }
}

/**
 * 骨格推定座標をIndexDBに格納します
 * @param {object} position 推定骨格座標（複数可）
 * @param {string} DB_name IndexDBの名前
 * @param {string} storeName オブジェクトストアの名前
 * @param {int} keyNumber キー番号
 */
function InsertPoseDB(position, DB_name, storeName, keyNumber){

    const data = {time_stamp: keyNumber, pose: position};

    let openReq  = indexedDB.open(DB_name, 1);

    openReq.onsuccess = function(event){
        const db = event.target.result;
        const trans = db.transaction(storeName, 'readwrite');
        const store = trans.objectStore(storeName);
        store.put(data);
    }
}

/**
 * 格納した骨格推定座標をIndexDBから取り出します
 * @param {string} DB_name IndexDBの名前
 * @param {string} storeName オブジェクトストアの名前
 * @return {Object} KeyNumberまでの骨格座標
 */
function ResumePoseDB(DB_name, storeName){

    let openReq  = indexedDB.open(DB_name, 1);
    let ret_data;

    openReq.onsuccess = function(event){
        const db = event.target.result;
        const trans = db.transaction(storeName, 'readonly');
        const store = trans.objectStore(storeName);
        const getReq = store.getAll();

        getReq.onsuccess = function(event){
            ret_data = event.target.result;
        }
    }
    return ret_data;
}