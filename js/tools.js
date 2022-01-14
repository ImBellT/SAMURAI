/**
 * 作成したモデルをダウンロードします
 * @param {object} models UseCustomModelで出力されたモデルデータ
 */
async function downloadModel(models) {

    // モデルからアーティファクトを作成（configはincludeOptimizerとtrainableOnlyのbool値を記載（なくてもいい））
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

    // JSZip定義
    const zip = new JSZip();

    // 圧縮するファイルを定義
    const artifact = await saveModelArtifacts(models);
    const [model, weights] = await SaveModelTF(artifact);
    const std = new Blob([JSON.stringify(models[1], null, '')], {type: 'application\/json'});
    const info_json = new Blob([JSON.stringify({
        "validationData": models[2].validationData,
        "params": models[2].params
    }, null, '')], {type: 'application\/json'});
    const data1 = new Blob([JSON.stringify(models[3][0], null, '')], {type: 'application\/json'});
    const data2 = new Blob([JSON.stringify(models[3][1], null, '')], {type: 'application\/json'});

    const transpose = a => a[0].map((_, c) => a.map(r => r[c]));
    const info_csv_data = transpose([models[2].epoch, models[2].history.acc, models[2].history.val_acc, models[2].history.loss, models[2].history.val_loss]);
    let csv_string = "Epoch（学習回数）,Accuracy（精度）,Val accuracy（評価精度）,Loss（損失）,Val loss（評価損失）\r\n";
    info_csv_data.forEach(d => {
        csv_string += d.join(","); // 配列ごとの区切りを「,」をつけて一列化
        csv_string += '\r\n';
    });
    let bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
    const info_csv = new Blob([bom, csv_string], {type: "text/csv"}); // 抽出したデータをCSV形式に変換

    // 定義したファイルをアーカイブに追加
    zip.file("samurai_custom_model.json", model);
    zip.file("samurai_custom_model_weights.bin", weights);
    zip.file("samurai_custom_model_standard.json", std);
    let analytics = zip.folder("分析用ファイル");
    analytics.file("モデル仕様.json", info_json);
    analytics.file("学習精度・損失.csv", info_csv);
    let l_data = zip.folder("学習に使用したデータ");
    l_data.file("学習データ.json", data1);
    l_data.file("テストデータ.json", data2);

    zip.generateAsync({type: "blob"}).then(function (dataBlob) {
        const DownloadUrl = URL.createObjectURL(dataBlob); // BlobデータをURLに変換
        const downloadOpen = document.createElement('a');
        downloadOpen.href = DownloadUrl;
        downloadOpen.download = "samurai_custom_model"; // ダウンロード時のファイル名を指定
        downloadOpen.click(); // 疑似クリック
        URL.revokeObjectURL(DownloadUrl); // 作成したURLを解放（削除）
    });
}

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
 * @param {string} csv_file CSVファイル形式の学習データ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {string} csv_file_test CSVファイル形式の評価用テストデータ（URLでファイルを指定・Input形式の場合createObjectURLで作る）
 * @param {object} config 設定データ
 * @return {object} データによって学習済みのモデル
 */
async function UseCustomModel(csv_file, csv_file_test, config) {

    const wait = ms => new Promise(resolve => setTimeout(() => resolve(), ms)); // ミリ秒でタイムアウトする関数を定義

    _statusText.main.innerHTML = "処理状況：新規モデルの作成・学習を実行します。";
    const model = MakeCustomModel();
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

    const [data, data_test] = config.custom_valid ? [await ProcessCSVData(csv_file), await ProcessCSVData(csv_file_test)] : splitData(await ProcessCSVData(csv_file), config, param.selector);

    const [x_pos, y] = SplitDataIndex(data);
    let x_key = FeatureConvertAll(x_pos, param);
    const [x_pos_test, y_test] = SplitDataIndex(data_test);
    let x_key_test = FeatureConvertAll(x_pos_test, param);

    [param.mean, param.std] = CalcMeanAndStd(x_key); // 平均値や標準偏差を導出

    // デバッグ用
    if (config.debug_mode) {
        console.log("学習データ↓");
        console.log(data);
        console.log("テストデータ↓");
        console.log(data_test);
        console.log("説明変数↓");
        console.log(x_key);
        console.log("平均値↓");
        console.log(JSON.parse(JSON.stringify(param.mean)));
        console.log("標準偏差↓");
        console.log(JSON.parse(JSON.stringify(param.std)));
    }

    x_key = KeyStandardization(x_key, param);
    x_key_test = KeyStandardization(x_key_test, param);
    const x_tensor = tf.tensor2d(KeyToValue(x_key));
    const y_tensor = tf.tensor2d(LabelOneHotEncode(y, param));

    // デバッグ用
    if (config.debug_mode) {
        console.log("モデルサマリー↓");
        model.summary();
    }

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

    // デバッグ用
    if (config.debug_mode) {
        console.log("学習結果↓");
        console.log(JSON.parse(JSON.stringify(info)));
    }

    _statusText.main.innerHTML = "処理状況：新規モデルの学習が完了しました。ならびにモデルのダウンロードを解除しました。";
    await wait(1000);
    return [model, param, info, [data, data_test]];
}

/**
 * MIYABIモデルを読み込みます（binファイルとstandard.jsonファイルが同一ディレクトリにいることを必ず確認！）
 * @param {string} category モデルの種類
 * @return {object} [Kerasモデル, モデルのパラメータ]
 */
async function miyabiSelector(category){
    const version = Number(category.substr(8,1));
    const optimizer = category.substr(10);
    console.log("/model/miyabi_v" + version + "/" + optimizer + "/model.json");
    return LoadModel("/model/miyabi_v" + version + "/" + optimizer + "/model.json");
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
 * 学習データをtrainとtestで分割します（目標セレクタごと）
 * @param {object} data シャッフルしたい配列
 * @param {object} config 設定データ
 * @param {object} selector 目標セレクタ
 * @return {object} [学習データ, テストデータ]
 */
function splitData(data, config, selector) {
    let selector_str = [[1, 2, 3, 4], ['U', 'M'], ['L', 'R'], ['S', 'F']];
    let tmp, result = [[], []];
    if (config.data_shuffle) data = shuffleMatrix(data, config.seed); // シャッフルの指定があれば事前に行う
    if (selector.toString() === [1, 0, 0, 0].toString()) { // toStringを使う理由はJavaScriptでは配列の合同関係を直接検証することができないため（これがないと常にfalseを返してしまう）
        tmp = [[], [], [], []];
        data.forEach(e => {
            const cat = selector_str[0].indexOf(Number(e.label.charAt(0)));
            tmp[cat] = tmp[cat].concat(e)
        }); // 技種別コード-1の配列に突っ込む
    } else {
        // 後で頑張れ for 未来の自分
    }
    tmp.forEach(e => {
        result[0] = result[0].concat(e.slice(Math.ceil(e.length * config.test_ratio), e.length));
        result[1] = result[1].concat(e.slice(0, Math.ceil(e.length * config.test_ratio)));
    });
    return result;
}

/**
 * 乱数を発生させ、配列をシャッフルします（シード機能あり）
 * @param {number} seed 乱数シード値（NaN時は完全ランダム）
 * @param {object} matrix シャッフルしたい配列
 * @return {object} シャッフル後の配列
 */
function shuffleMatrix(matrix, seed) {

    if (isNaN(seed)) seed = Math.floor(Math.random() * 9);

    // 乱数シードリストを定義
    let seed_list = [123456789, 362436069, 521288629, seed];
    let new_list_index = [];
    let new_matrix = [];

    function make() {
        const t = seed_list[0] ^ (seed_list[0] << 11);
        seed_list[0] = seed_list[1];
        seed_list[1] = seed_list[2];
        seed_list[2] = seed_list[3];
        seed_list[3] = (seed_list[3] ^ (seed_list[3] >>> 19)) ^ (t ^ (t >>> 8));
        const r = Math.abs(seed_list[3]);
        return r % (matrix.length + 1);
    }

    for (let i = 0; i < matrix.length; i++) {
        while (true) {
            const num = make();
            let usable = false;
            new_list_index.forEach(e => {
                if (e === num) usable = true
            });
            if (!usable) {
                new_list_index = new_list_index.concat([[i, num]]);
                break;
            }
        }
    }
    new_list_index.sort((a, b) => a[1] - b[1]);
    matrix.forEach(e => new_matrix[matrix.indexOf(e)] = matrix[new_list_index[matrix.indexOf(e)][0]]);
    return new_matrix;
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