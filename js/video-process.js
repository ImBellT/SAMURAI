/**
 * 骨格・技推定を実行します
 * @param {object} canvas キャンバス（calc, previewとも必要）
 * @param {object} ctx コンテキスト（calc, previewとも必要）
 * @param {object} inputVideo 入力映像（videoタグ）
 * @param {object} poseParam 骨格推定の設定パラメータ
 * @return {object} 推定骨格座標・説明変数
 */
async function RunSimulation(canvas, ctx, inputVideo, poseParam) {

    _statusText.main.innerHTML = "処理状況：処理を開始しました。モデルを読み込んでいます。";

    let time_num = 0;
    let position;
    let key;
    let result;

    async function ChoiceModel() {
        let model, param, info, data;
        switch (document.getElementById("dnn_model").value) {
            case "my_model":
                [model, param] = await LoadModel_Outside(document.getElementById("custom_model_file_json"), document.getElementById("custom_model_file_weight"), document.getElementById("custom_model_file_standard"));
                break;
            case "custom":
                const DataFile = URL.createObjectURL(document.getElementById("custom_data_file").files[0]); // 学習データを変数化
                const TestFile = document.getElementById("custom_valid_file").files[0] === null ? URL.createObjectURL(document.getElementById("custom_valid_file").files[0]) : null;
                document.getElementById("shuffle_seed").value = parseInt(document.getElementById("shuffle_seed").value, 10); // 小数点とかはすべて除去
                const config = {
                    debug_mode: document.getElementById("debug-mode").checked,
                    custom_valid: document.getElementById("custom_valid").checked,
                    data_shuffle: document.getElementById("data_shuffle").checked,
                    test_ratio: parseFloat(document.getElementById("valid_ratio").value),
                    seed: parseInt(document.getElementById("shuffle_seed").value, 10)
                };
                [model, param, info, data] = await UseCustomModel(DataFile, TestFile, config); // 学習データを使用してモデル作成
                document.getElementById("save_model").disabled = false; // ダウンロードボタンを有効化（グレーアウトを解除）
                document.getElementById("save_model").title = "作成したモデルをダウンロードします";
                break;
            default:
                [model, param] = await miyabiSelector(document.getElementById("dnn_model").value);
                model.summary();
        }
        if (document.getElementById("dnn_model").value === "custom") {
            return [model, param, info, data];
        } else {
            return [model, param, 0, 0];
        }

    }

    // 骨格座標を検出する関数
    async function EstimatePose() {
        ResetAllCanvas(ctx, canvas, inputVideo); // 計算用・プレビュー用のキャンバスをリセットする
        if (poseParam.model.value === "pose_net") {
            position = await netModel.estimatePoses(canvas.calc, {maxPoses: 4});
        } else {
            position = await netModel.estimatePoses(canvas.calc);
        }
    }

    // 検出座標を変換して技予測を行う関数
    function TechEmulation() {
        position = PosDataResize(position, canvas.calc, param);
        if (position.length !== 0) {
            key = FeatureConvert(position, param);
        }
        // 特徴量定義&標準化
        if (key !== undefined) {
            key = [KeyStandardization(key[0], param), KeyStandardization(key[1], param)];
        } else {
            key = undefined;
        }
        if (key !== undefined) {
            // ちゃんと説明変数が定義されていたら予測を行う
            let result_once1 = model.predict(tf.tensor([KeyToValue(key[0])]));
            result_once1 = AdjustWithScore(key[0], result_once1);
            let result_once2 = model.predict(tf.tensor([KeyToValue(key[1])]));
            result_once2 = AdjustWithScore(key[1], result_once2);

            // 正変数と逆変数を比較し高確率のほうを採用する
            const aryMax = function (a, b) {
                return Math.max(a, b);
            } // 大きい数値をとるmax関数を定義
            const result_max_1 = result_once1.reduce(aryMax); // 配列分繰り返して最大値を取得
            const result_max_2 = result_once2.reduce(aryMax);
            result = result_max_1 >= result_max_2 ? result_once1 : result_once2;
            PrintParam_Key(result_max_1 >= result_max_2 ? key[0] : key[1], _statusText);
        } else {
            result = [0, 0, 0, 0];
        }
    }

    // 骨格描画を行う関数
    function drawIt() {
        if (position.length > 0) {
            position = AdjustPoseIndex(position);
            drawPoses(position, ctx, poseParam, scale); // 結果から描画を実行
            PrintParam_Origin(position, _statusText); // 結果をリアルタイムで出力
        }
    }

    const wait = ms => new Promise(resolve => setTimeout(() => resolve(), ms)); // ミリ秒でタイムアウトする関数を定義
    makeDB("samurai_db"); // 骨格と結果を格納するIndexedDBを作成
    let graph = await graphInitialize('chart_div');
    const netModel = await CreatePoseModel(inputVideo, poseParam); // 骨格モデル定義（選択肢ごとに定義）
    const scale = AdjustCanvasToCtx(inputVideo, canvas, poseParam); // キャンバス設定
    const [model, param, info, data] = await ChoiceModel(); // 設定に準してモデルとパラメータを定義
    model_data = [model, param, info, data]; // グローバル変数にmodelとparamを代入（ダウンロードできるように）
    document.getElementById("model_param").innerHTML = "※本機能は近日公開";

    _statusText.main.innerHTML = "処理状況：モデルの読み込みがすべて終了しました。予測と描画を開始します。";

    ResetAllCanvas(ctx, canvas, inputVideo); // 計算用・プレビュー用のキャンバスをリセットする

    let percentage = 0;
    let max_percentage = document.getElementById("trim_time_end").value === "" ? 99 : Number(document.getElementById("trim_time_end").value) / inputVideo.duration * 100;
    if(max_percentage >= 99) max_percentage = 99;
    inputVideo.currentTime = document.getElementById("trim_time_start").value === "" ? 0 : Number(document.getElementById("trim_time_start").value);
    while (percentage < max_percentage) {
        await wait(20);
        if (inputVideo.readyState > 1) {
            await EstimatePose(); // 骨格予想
            TechEmulation(); // 技予想
            drawIt();
            insertPoseDB(position, "samurai_db", "pose_store", time_num);
            insertPoseDB(result, "samurai_db", "result_store", time_num);
            time_num++;
            inputVideo.currentTime += parseFloat(poseParam.flame_stride.value) / 1000; // 動画を設定値の秒数分進める
            percentage = Math.floor((inputVideo.currentTime / inputVideo.duration) * 10000) / 100
            _statusText.main.innerHTML = "動画を処理中…（" + String(percentage) + "%完了）<br/>・処理フレーム数：" + time_num + "<br/>・経過秒数（スキップ秒数含む）：" + inputVideo.currentTime + "秒";
            addGraphData(graph, result, time_num);
        }
    }
    _statusText.origin.innerHTML = "処理完了";
    _statusText.master.innerHTML = "処理完了";
    _statusText.main.innerHTML = "骨格の推定が完了しました。推定データをダウンロードするにはJSONダウンロードをクリックしてください。";
}