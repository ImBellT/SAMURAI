// 姿勢推定を実行
async function runVideoPoseSimulation() {
    if (_inputVideo.getAttribute("src") === "" && (document.getElementById("dnn_model").value !== "custom" || (document.getElementById("dnn_model").value === "custom" && !document.getElementById("model_only").checked))) {
        alert('動画をアップロードしてください。');
        return;
    }
    if ((document.getElementById("custom_data_file").files[0] || document.getElementById("custom_valid_file").files[0] === undefined) === undefined && document.getElementById("dnn_model").value === "custom") {
        alert('学習データをアップロードしてください。');
        return;
    }
    if (document.getElementById("dnn_model").value === "custom" && document.getElementById("model_only").checked) {
        ChangeInputAvailability(true);
        // モデル学習のみ
        const DataFile = URL.createObjectURL(document.getElementById("custom_data_file").files[0]); // 学習データを変数化
        const TestFile = URL.createObjectURL(document.getElementById("custom_valid_file").files[0]); // テストデータを変数化
        model_data = await UseCustomModel(DataFile, TestFile); // 学習データを使用してモデル作成
        document.getElementById("save_model").disabled = false; // ダウンロードボタンを有効化（グレーアウトを解除）
        document.getElementById("save_model").title = "作成したモデルをダウンロードします";
        ChangeInputAvailability(false);
    } else {
        // 全部やる
        result_data = await RunSimulation(_Canvas, _Ctx, _inputVideo, _poseParam); // 姿勢推定
        ResetPreviewVision();
        ChangeInputAvailability(true);
        document.getElementById("save_result").disabled = false; // ダウンロードボタンを有効化（グレーアウトを解除）
        document.getElementById("save_result").title = "予測した技確率をCSV形式でダウンロードします";
        document.getElementById("save_pose").disabled = false; // ダウンロードボタンを有効化（グレーアウトを解除）
        document.getElementById("save_pose").title = "予測した骨格をJSON形式でダウンロードします";
        ChangeInputAvailability(false);
    }

}

// JSON形式でダウンロード
function DownloadDataJSON(Data, Filename) {
    const dataBlob = new Blob([JSON.stringify(Data, null, '')], {type: 'application\/json'}); // 抽出したデータをJSON形式に変換
    const DownloadUrl = URL.createObjectURL(dataBlob); // JSONデータをURLに変換
    const downloadOpen = document.createElement('a');
    downloadOpen.href = DownloadUrl;
    downloadOpen.download = Filename; // ダウンロード時のファイル名を指定
    downloadOpen.click(); // 疑似クリック
    URL.revokeObjectURL(DownloadUrl); // 作成したURLを解放（削除）
}

// CSV形式でダウンロード
function DownloadDataCSV(Data, Filename) {
    let csv_string = "";
    for (let d of Data) {
        csv_string += d.join(","); // 配列ごとの区切りを「,」をつけて一列化
        csv_string += '\r\n';
    }
    const dataBlob = new Blob([csv_string], {type: "text/csv"}); // 抽出したデータをCSV形式に変換
    const DownloadUrl = URL.createObjectURL(dataBlob); // JSONデータをURLに変換
    const downloadOpen = document.createElement('a');
    downloadOpen.href = DownloadUrl;
    downloadOpen.download = Filename; // ダウンロード時のファイル名を指定
    downloadOpen.click(); // 疑似クリック
    URL.revokeObjectURL(DownloadUrl); // 作成したURLを解放（削除）
}

// ファイルを圧縮してZIP形式でダウンロード
async function downloadModel(models) {

    // モデルからアーティファクトを作成（configはincludeOptimizerとtrainableOnlyのbool値を記載（なくてもいい））
    // tfjsのsave関数から引用
    async function saveModelArtifacts(models, config) {
        // 重みデータを最適化
        const weightDataAndSpecs = await tf.io.encodeWeights(models[0].getNamedWeights(config));

        const returnString = false;
        const unusedArg = null;

        // モデル情報をJSON出力できるように変換する
        const modelConfig = models[0].toJSON(unusedArg, returnString);
        // アーティファクト定義
        const modelArtifacts = {
            modelTopology: modelConfig,
            format: 'layers-model',
            generatedBy: `Bell.T Science`,
            convertedBy: null,
        };
        // オプティマイザ追記
        const includeOptimizer = config == null ? false : config.includeOptimizer;
        if (includeOptimizer && models[0].optimizer != null) {
            modelArtifacts.trainingConfig = this.getTrainingConfig();
            const weightType = 'optimizer';
            const {
                data: optimizerWeightData,
                specs: optimizerWeightSpecs
            } = await tf.io.encodeWeights(await models[0].optimizer.getWeights(), weightType);
            weightDataAndSpecs.specs.push(...optimizerWeightSpecs);
            weightDataAndSpecs.data = tf.io.concatenateArrayBuffers([weightDataAndSpecs.data, optimizerWeightData]);
        }
        // メタデータ追記
        if (models[0].userDefinedMetadata != null) {
            const checkSize = true;
            checkUserDefinedMetadata(models[0].userDefinedMetadata, models[0].name, checkSize);
            modelArtifacts.userDefinedMetadata = models[0].userDefinedMetadata;
        }
        modelArtifacts.weightData = weightDataAndSpecs.data;
        modelArtifacts.weightSpecs = weightDataAndSpecs.specs;
        return modelArtifacts;
    }

    // アーティファクトに重み情報を追加
    function getModelForModelArtifacts(model_artifacts, w_manifest) {
        const data = {
            modelTopology: model_artifacts.modelTopology,
            format: model_artifacts.format,
            generatedBy: model_artifacts.generatedBy,
            convertedBy: model_artifacts.convertedBy,
            weightsManifest: w_manifest
        };
        if (model_artifacts.signature != null) {
            data.signature = model_artifacts.signature;
        }
        if (model_artifacts.userDefinedMetadata != null) {
            data.userDefinedMetadata = model_artifacts.userDefinedMetadata;
        }
        if (model_artifacts.modelInitializer != null) {
            data.modelInitializer = model_artifacts.modelInitializer;
        }
        if (model_artifacts.trainingConfig != null) {
            data.trainingConfig = model_artifacts.trainingConfig;
        }
        return data;
    }

    // アーティファクトからダウンロードデータを作成
    async function SaveModelTF(modelArtifact) {
        if (typeof (document) === 'undefined') {
            throw new Error('この環境上ではdocumentが存在しないため、ブラウザからのダウンロードはサポートされていません。');
        }

        async function modelBlob() {
            const modelJSON = getModelForModelArtifacts(modelArtifact, weightsManifest);
            return new Blob([JSON.stringify(modelJSON)], {type: 'application/json'});
        }

        async function weightBlob() {
            return new Blob([modelArtifact.weightData], {type: 'application/octet-stream'});

        }

        const weightsManifest = [{
            paths: ['./' + "samurai_custom_model_weights.bin"],
            weights: modelArtifact.weightSpecs
        }];
        return [await modelBlob(), await weightBlob()];
    }

    // 取得情報を圧縮
    async function modelZipper() {

        // JSZip定義
        const zip = new JSZip();

        // 圧縮するファイルを定義
        const artifact = await saveModelArtifacts(models);
        const [model, weights] = await SaveModelTF(artifact);
        const std = new Blob([JSON.stringify(models[1], null, '')], {type: 'application\/json'});
        const info = new Blob([JSON.stringify(models[2], null, '')], {type: 'application\/json'});

        // 定義したファイルをアーカイブに追加
        zip.file("samurai_custom_model.json", model);
        zip.file("samurai_custom_model_weights.bin", weights);
        zip.file("samurai_custom_model_standard.json", std);
        zip.file("samurai_custom_model_info.json", info);
        zip.folder("分析用CSVファイル");

        zip.generateAsync({type: "blob"}).then(function (dataBlob) {
            const DownloadUrl = URL.createObjectURL(dataBlob); // BlobデータをURLに変換
            const downloadOpen = document.createElement('a');
            downloadOpen.href = DownloadUrl;
            downloadOpen.download = "samurai_custom_model"; // ダウンロード時のファイル名を指定
            downloadOpen.click(); // 疑似クリック
            URL.revokeObjectURL(DownloadUrl); // 作成したURLを解放（削除）
        });
    }

    let caution = confirm('以下のファイルをZIP形式saveに圧縮し、ダウンロードします。\n\n・モデルの枠組み（JSON）\n・重み（BIN）\n・学習データの平均値や標準偏差等を格納したファイル（JSON）\n・学習精度の履歴（JSON）\n・評価データの分析用データ（CSV）\n\n※ブラウザによっては複数ダウンロードの許可を出す必要がありますのでご注意ください。\n本当に続行しますか？');
    if (caution) {
        await modelZipper(model_data);
    }
}

// デバッグ用関数
async function runDebug() {
    ChangeInputAvailability(true);
    const DataFile = "data/learning_data_ss1500_c11limited.csv"; // 学習データを変数化
    const TestFile = "data/learning_data_ss1500_c11only.csv"; // テストデータを変数化
    model_data = await UseCustomModel(DataFile, TestFile); // 学習データを使用してモデル作成
    document.getElementById("save_model").disabled = false; // ダウンロードボタンを有効化（グレーアウトを解除）
    document.getElementById("save_model").title = "作成したモデルをダウンロードします";
    ChangeInputAvailability(false);
    await downloadModel(model_data);
}

// ページ読み込み時に発火
window.addEventListener('load', async () => {
    ResetStyle(); // アコーディオンを初期化
    ResetPreviewVision(); // プレビュー画面を初期化
});

// ダウンロードボタン押し時に発火
document.getElementById("save_pose").addEventListener('click', () => {
    DownloadDataJSON(ResumePoseDB("samurai_db", "pose_store"), "pose.json");
});
document.getElementById("save_result").addEventListener('click', () => {
    DownloadDataCSV(result_data, "result.csv");
});
document.getElementById("save_model").addEventListener('click', () => {
    downloadModel().then();
});

// 処理実行ボタン押し時に発火
document.getElementById("send_data").addEventListener('click', () => {
    runVideoPoseSimulation().then();
});

// デバッグボタン押し時に発火
document.getElementById("run_test").addEventListener('click', () => {
    runDebug().then();
});